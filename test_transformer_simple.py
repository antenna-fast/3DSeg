"""
from point transformer
Simplified by ANTenna
"""

import os
import sys
import time
import random
import numpy as np
# import pickle
# import collections

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from utils.metric_utils import AverageMeter, intersectionAndUnion
from utils.logger_utils import create_logger
from utils.config import get_parser
from utils.data_utils import get_data_list, input_normalize, data_load


def test(model, criterion, names, is_vis=0):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    # Init Metric Buffer
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    args.batch_size_test = 10

    model.eval()

    # check_makedirs(args.save_folder)
    pred_save, label_save = [], []
    data_list = get_data_list(args)  # get test data list
    num_data_list = len(data_list)
    for idx, data_name in enumerate(data_list):  # file name
        start_time = time.time()
        data_path = os.path.join(args.data_root, args.train_full_folder, data_name + '.npy')

        args.voxel_size = 0  # do NOT use voxelize when load test data
        coord, feat, label, idx_data = data_load(data_path, args)  # idx_data: [[part_1], [part_2], ...]
        # Init predict results of the current whole scene point cloud
        pred = torch.zeros((label.size, args.classes)).cuda()  # N x classes
        idx_list, coord_list, feat_list, offset_list = [], [], [], []

        idx_size = len(idx_data)
        for i, idx_part in enumerate(idx_data):  # for each Part
            coord_part, feat_part = coord[idx_part], feat[idx_part]  # xyz, rgb
            # Split to reduce mem-cost
            if args.voxel_max and coord_part.shape[0] > args.voxel_max:
                coord_p = np.random.rand(coord_part.shape[0]) * 1e-3
                idx_uni = np.array([])
                cnt = 0
                while idx_uni.size != idx_part.shape[0]:
                    init_idx = np.argmin(coord_p)
                    dist = np.sum(np.power(coord_part - coord_part[init_idx], 2), 1)
                    idx_crop = np.argsort(dist)[:args.voxel_max]
                    coord_sub, feat_sub, idx_sub = coord_part[idx_crop], feat_part[idx_crop], idx_part[idx_crop]
                    dist = dist[idx_crop]
                    delta = np.square(1 - dist / np.max(dist))
                    coord_p[idx_crop] += delta
                    # Normalize
                    coord_sub, feat_sub = input_normalize(coord_sub, feat_sub)
                    # For Down stream usage
                    idx_list.append(idx_sub)
                    coord_list.append(coord_sub)
                    feat_list.append(feat_sub)
                    offset_list.append(idx_sub.size)

                    idx_uni = np.unique(np.concatenate((idx_uni, idx_sub)))
            else:  # Inference all points at once
                coord_part, feat_part = input_normalize(coord_part, feat_part)
                idx_list.append(idx_part)
                coord_list.append(coord_part)
                feat_list.append(feat_part)
                offset_list.append(idx_part.size)
        # Parse to mini-Batch Inference: 把一个完整的点云拆分为batch size
        batch_num = int(np.ceil(len(idx_list) / args.batch_size_test))  # 向上取整
        for i in range(batch_num):  # to batch processing
            s_i, e_i = i * args.batch_size_test, min((i + 1) * args.batch_size_test, len(idx_list))
            idx_part = idx_list[s_i:e_i]
            coord_part = coord_list[s_i:e_i]
            feat_part = feat_list[s_i:e_i]
            offset_part = offset_list[s_i:e_i]
            idx_part = np.concatenate(idx_part)  # 2D list to 1D index vector
            coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)
            feat_part = torch.FloatTensor(np.concatenate(feat_part)).cuda(non_blocking=True)
            offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)

            with torch.no_grad():
                # pred_part = model([coord_part, feat_part, offset_part])  # (n, k)
                pred_part = model([coord_part, feat_part, feat_part])  # (n, k)
            torch.cuda.empty_cache()
            pred[idx_part, :] += pred_part  # Accumulate all parts' inference result

        # Sample level metric
        loss = criterion(pred, torch.LongTensor(label).cuda(non_blocking=True))  # for reference
        pred = pred.max(1)[1].data.cpu().numpy()  # Get predicted class index by argmax (top 1)

        # Metric: Each scene, Class Level metric
        intersection, union, target = intersectionAndUnion(pred, label, args.classes, args.ignore_label)
        accuracy = sum(intersection) / (sum(target) + 1e-10)

        # Update metric buffer in class level
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        batch_time.update(time.time() - start_time)

        # Scene Level Metric
        logger.info('Test:[{}/{}] | NumPoints:{} | BatchTime:{batch_time.val:.3f}s | Accuracy {accuracy:.4f}.'.
                    format(idx + 1, num_data_list, label.size, batch_time=batch_time, accuracy=accuracy))
        pred_save.append(pred)
        label_save.append(label)
        # np.save(pred_save_path, pred)
        # np.save(label_save_path, label)

    # Save all scenes' prediction results
    # with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
    #     pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(args.save_folder, "label.pickle"), 'wb') as handle:
    #     pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Overall Metric
    # Calculation 1  Overall Test set Class Level metric
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # Calculation 2  Overall Test set Class Level metric
    intersection, union, target = intersectionAndUnion(np.concatenate(pred_save), np.concatenate(label_save), args.classes, args.ignore_label)
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)

    logger.info('Val0 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    logger.info('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))

    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.
                    format(i, iou_class[i], accuracy_class[i], names[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    # Load Config
    code_root = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    config_file_path = os.path.join(code_root, 'config/s3dis/s3dis_antenna.yaml')  # YAML config file
    args = get_parser(desc='ANTenna3DSeg', config_file=config_file_path)

    selected_epoch = 12

    # Logger
    data_root = args.data_root
    test_logger_path = os.path.join(data_root, args.test_log)
    logger = create_logger(test_logger_path)
    logger.info(args)
    # Writer

    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    random.seed(123)
    np.random.seed(123)

    # Model architecture
    if args.arch == 'pointNN':
        from model.pointNN import NN as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))

    model = Model(in_dim=9, hidden_dim=64, out_dim=13).cuda()
    logger.info('model: \n'.format(model))

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    names = [line.rstrip('\n') for line in open(os.path.join(args.data_root, args.names_path))]  # part names

    # Checkpoints
    if args.model_path:  # default=None
        # Load selected epoch weight
        model_root_path = os.path.join(data_root, args.save_path, args.model_path)
        model_path_list = os.listdir(model_root_path)
        model_path = os.path.join(data_root, args.model_root_path, [p for p in model_path_list if 'train_epoch_{}'.format(selected_epoch) in p][0])
        if os.path.isfile(model_path):
            logger.info("=> loading weight '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(model_path))
        else:
            logger.info("=> no weight found at '{}'".format(model_path))
            raise KeyError
    else:  # return if the weights are not given
        raise KeyError

    test(model, criterion, names)

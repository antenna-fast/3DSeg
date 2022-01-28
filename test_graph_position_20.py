"""
from point transformer
Simplified by ANTenna

Test: 3D Semantic Segmentation
"""

import os
import sys
import time
import random
import numpy as np
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.getcwd())
from utils.metric_utils import AverageMeter, intersectionAndUnion
from utils.logger_utils import create_logger
from utils.config import get_parser
from utils.data_utils import get_data_list, input_normalize, data_load

from utils.vis_utils import show_inference
from matplotlib import cm

color_map = cm.get_cmap('tab20').colors


def test(model, criterion, writer, args=0):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    names = [line.rstrip('\n') for line in open(os.path.join(args.data_root, args.names_path))]  # part names 

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

    # Test on all data
    for idx, data_name in enumerate(data_list):  # file name
        start_time = time.time()
        # Load data
        data_path = os.path.join(args.data_root, args.train_full_folder, data_name + '.npy')

        # When loading, split the whole scene into several parts 
        # coord: Nx3 xyz, feat: Nx3 rgb, label: N,
        # idx_data: [[part_1], [part_2], ...]
        coord, feat, label, idx_data = data_load(data_path, args)
        num_points = len(coord)
        point_idxs = np.arange(num_points)

        # sample
        sample_num = 20000
        selected_point_idxs = np.random.choice(point_idxs, sample_num, replace=False)
        coord = torch.tensor(coord[selected_point_idxs])
        feat = torch.tensor(feat[selected_point_idxs])
        label = torch.tensor(label[selected_point_idxs], dtype=torch.long).unsqueeze(0)
        x = torch.concat([coord, feat], dim=-1).unsqueeze(0).to(device)

        pred = model(x)
        pred = pred.permute(0, 2, 1)  # to [batch, class, num_samples]
        loss = criterion(pred, label.cuda(non_blocking=True))

        pred = pred.max(1)[1].cpu().numpy()[0]  # Get predicted result: to [batch, N], to numpy
        label = label.cpu().numpy()[0]
        # Scene Level Metric
        intersection, union, target = intersectionAndUnion(pred, label, args.classes, args.ignore_label)
        accuracy = sum(intersection) / (sum(target) + 1e-10)
        mIoU_test_scene = np.mean(intersection / (union + 1e-10))
        mAcc_test_scene = np.mean(intersection / (target + 1e-10))

        # Update metric buffer in Class level
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        batch_time.update(time.time() - start_time)

        # Scene Level Log
        log_idx = idx + 1
        test_loss = loss.item()
        logger.info('Test:[{}/{}] | Scene:{} | NumPoints:{} | Time:{batch_time:.4f}s | '
                    'Loss:{loss:.5f} | Acc:{acc:.5f} | mIoU:{miou:.5f} | mAcc:{macc:.5f}.'.
                    format(log_idx, num_data_list, data_name, label.size, batch_time=batch_time.val,
                           loss=test_loss, acc=accuracy, miou=mIoU_test_scene, macc=mAcc_test_scene))
        # For Overall Metric        
        pred_save.append(pred)
        label_save.append(label)
        # np.save(pred_save_path, pred)
        # np.save(label_save_path, label)

        if args.is_vis:
            # Visualize Labeled Point Cloud  
            color_list_gt = []
            color_list_pred = []  # time consuming... can replace with  map function
            for c in label:
                color_list_gt.append(np.array(color_map[int(c)]) * 255)
            for c in pred:
                color_list_pred.append(np.array(color_map[int(c)]) * 255)
            color_gt_np = np.array(color_list_gt)
            color_pred_np = np.array(color_list_pred)

            # Show on tensorboard
            # tb_color_gt = torch.tensor(color_gt_np, dtype=torch.int).unsqueeze(0)
            # tb_color_pred = torch.tensor(color_pred_np, dtype=torch.int).unsqueeze(0)
            # writer.add_mesh(tag='TestVisPointsGT', vertices=torch.from_numpy(coord).unsqueeze(0), colors=tb_color_gt, global_step=idx)
            # writer.add_mesh(tag='TestVisPointsPred', vertices=torch.from_numpy(coord).unsqueeze(0), colors=tb_color_pred, global_step=idx)

            # Save Labeled Point Cloud
            pcd_save_root = os.path.join(args.save_root, args.pcd_save_path, args.arch)
            show_inference(coord, color_pred_np, is_show=0, is_save=1, is_norm=1,
                           save_path=os.path.join(pcd_save_root, 'pred_scene_{}.ply'.format(idx)))
            show_inference(coord, color_gt_np, is_show=0, is_save=1, is_norm=1,
                           save_path=os.path.join(pcd_save_root, 'gt_scene_{}.ply'.format(idx)))
            show_inference(coord, feat, is_show=0, is_save=1, is_norm=1,
                           save_path=os.path.join(pcd_save_root, 'color_scene_{}.ply'.format(idx)))

    # Save all scenes' prediction results
    # with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
    #     pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(args.save_folder, "label.pickle"), 'wb') as handle:
    #     pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Overall Metric
    # Calculation 1: Class Level
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    # Overall mean
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # Calculation 2  Overall Test set
    # # Class Level
    # intersection, union, target = intersectionAndUnion(np.concatenate(pred_save), np.concatenate(label_save), args.classes, args.ignore_label)
    # iou_class = intersection / (union + 1e-10)
    # accuracy_class = intersection / (target + 1e-10)
    # # Overall mean
    # mIoU = np.mean(iou_class)
    # mAcc = np.mean(accuracy_class)
    # allAcc = sum(intersection) / (sum(target) + 1e-10)

    # Class Level Metric Log
    for i in range(args.classes):
        logger.info('Class_{} | Name: {} | Result: IoU:{:.4f} | Accuracy:{:.4f}.'.
                    format(i, names[i], iou_class[i], accuracy_class[i]))

    # Overall Metric Log
    # logger.info('Val0 result: mIoU:{:.4f} | mAcc:{:.4f} | allAcc:{:.4f}.'.format(mIoU, mAcc, allAcc))
    logger.info('Val1 result: mIoU:{:.4f} | mAcc:{:.4f} | allAcc:{:.4f}.'.format(mIoU1, mAcc1, allAcc1))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    # Load Config
    code_root = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    config_file_path = os.path.join(code_root, 'config/s3dis/s3dis_antenna_graph_position_20.yaml')  # YAML config file
    args = get_parser(desc='ANTenna3DSeg', config_file=config_file_path)

    data_root = args.data_root
    save_root = args.save_root

    # Logger 
    log_path = os.path.join(args.save_root, args.log_path, args.arch)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    test_logger_path = os.path.join(log_path, args.test_log)
    logger = create_logger(test_logger_path)
    logger.info(args)
    assert args.classes > 1

    logger.info("Classes: {}".format(args.classes))

    # Writer
    writer_path = os.path.join(args.save_root, args.writer_path, args.arch)
    test_writer_path = os.path.join(writer_path, args.test_tensorboard_path)
    if os.path.exists(test_writer_path):  # delete old one
        shutil.rmtree(test_writer_path)
        os.makedirs(test_writer_path)
    writer = SummaryWriter(log_dir=test_writer_path)

    # Predict result
    pcd_save_root = os.path.join(args.save_root, args.pcd_save_path, args.arch)
    if not os.path.exists(pcd_save_root):
        os.makedirs(pcd_save_root)

    # Device 
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    if args.cuda:
        device = torch.device("cuda")
        import torch.backends.cudnn as cudnn
    else:
        device = torch.device("cpu")

    # Init
    if args.manual_seed is not None:
        manual_seed = args.manual_seed
        if device == torch.device("cuda"):
            cudnn.benchmark = False
            cudnn.deterministic = True
            torch.cuda.manual_seed_all(manual_seed)
        random.seed(manual_seed)
        np.random.seed(manual_seed)
        torch.manual_seed(manual_seed)

    # Model architecture
    arch = args.arch
    if arch == 'pointNN':
        from model.pointNN import NN as Model
    elif arch == 'pointNN_graph':
        from model.pointNN_graph import NN as Model
    elif arch == 'pointNN_graph_position_20':
        from model.pointNN_graph_position_20 import NN as Model
    else:
        raise Exception('Architecture {} NOT supported'.format(args.arch))

    logger.info("=> Creating model ...")
    model = Model(in_dim=args.feature_dim, hidden_dim=64, out_dim=args.classes)
    logger.info('model: \n'.format(str(model)))

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    # To Device 
    if device == torch.device("cuda"):
        model = nn.DataParallel(model)
        criterion = criterion.cuda()
        logger.info("Let's use {} GPU!".format(torch.cuda.device_count()))

    # Checkpoints
    checkpoints_path = os.path.join(args.save_root, args.checkpoints_path, args.arch)
    # Load selected epoch weight
    model_path_list = os.listdir(checkpoints_path)
    model_path = os.path.join(checkpoints_path,
                              [p for p in model_path_list if 'train_epoch_{}'.format(args.selected_epoch) in p][0])
    if os.path.isfile(model_path):
        logger.info("=> loading weight '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])  # TODO:
        logger.info("=> loaded weight '{}'".format(model_path))
    else:
        logger.info("=> no weight found at '{}'".format(model_path))
        raise KeyError

    test(model, criterion, writer=writer, args=args)

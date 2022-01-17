"""
Author: ANTenna on 2022/1/12
aliuyaohua@gmail.com

Description:
3D Semantic Test
"""

import os
import time
import shutil
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from dataset.s3dis import S3DIS  # dataset
from model.pointNN import NN as Model  # networks
# from utils.util import AverageMeter, intersectionAndUnionGPU
from utils.metric_utils import AverageMeter, intersectionAndUnion
from utils.logger_utils import create_logger  # log
from utils.config import get_parser
from utils import transform
# from utils.vis_utils import show_inference

import matplotlib.pyplot as plt
from matplotlib import cm

if torch.cuda.is_available():
    device = 'gpu'
    import torch.backends.cudnn as cudnn
else:
    device = 'cpu'


color_map = cm.get_cmap('tab20').colors


def test(model, test_loader, criteration, logger, writer, device, args):
    batch_time = AverageMeter()  # TODO: did what ???
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()

    num_batch = len(test_loader)
    running_loss = 0
    epoch_loss = 0
    for i, data in enumerate(test_loader):
        batch_start_time = time.time()
        x, target = data
        if device == 'gpu':  # TODO: judge from model.device
            x, target = x.to('cuda'), target.to('cuda')

        # forward -> loss
        output = model(x)  # batch, batch_size, class
        pred_label = output.argmax(-1)
        assert pred_label.shape == target.shape

        true_num = torch.sum(pred_label == target)
        all_points = target.shape[0] * target.shape[1]
        logger.info("batch precision [{}/{}]={:3f}".format(true_num, all_points, true_num / all_points))

        output = output.permute(0, 2, 1)
        # Loss
        loss = criteration(output, target)
        batch_time = time.time() - batch_start_time

        # Batch metric
        # output = output.max(1)[1]
        loss_item = loss.item()

        tb_p = x[:, :, 0:3][0].unsqueeze(0)
        # tb_color = torch.tensor(x[:, :, 3:6][0].unsqueeze(0) * 255, dtype=torch.int)
        # tb_color = torch.tensor(np.tile([255, 0, 0], (len(tb_p[0]), 1)), dtype=torch.int).unsqueeze(0)
        class_idx_gt = target.cpu().numpy()[0]
        class_idx_pred = pred_label.cpu().numpy()[0]

        color_list_gt = []
        color_list_pred = []
        
        for c in class_idx_gt:
            color_list_gt.append(np.array(color_map[c]) * 255)
        for c in class_idx_pred:
            color_list_pred.append(np.array(color_map[c]) * 255)

        tb_color_gt = torch.tensor(color_list_gt, dtype=torch.int).unsqueeze(0)
        tb_color_pred = torch.tensor(color_list_pred, dtype=torch.int).unsqueeze(0)
        
        # show on tensorboard
        writer.add_mesh(tag='TestVisPointsGT', vertices=tb_p, colors=tb_color_gt, global_step=i)
        writer.add_mesh(tag='TestVisPointsPred', vertices=tb_p, colors=tb_color_pred, global_step=i)
        print()

        """
        # intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersectionAndUnion(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), x.size(0))
        batch_time.update(time.time() - start_batch_time)       
        """
        running_loss += loss_item
        epoch_loss += loss_item
        if i % args.print_freq == 0 and i != 0:
            """
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.
                        format(epoch + 1, args.epochs, i + 1, len(test_loader),
                               batch_time=batch_time, data_time=data_time,
                               loss_meter=loss_meter,
                               accuracy=accuracy
                               ))
            """
            logger.info('BATCH:[{}/{}] Running loss:{} time:{:.4f}'.
                        format(i + 1, num_batch, running_loss, batch_time))
            writer.add_scalar('BatchLoss/Test', running_loss, i)
            running_loss = 0

    # Epoch metric
    # writer.add_scalar('EpochLoss/Test', epoch_loss, epoch)

    """
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Test result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.
                format(epoch + 1, args.epochs, mIoU, mAcc, allAcc))
    """
    # return loss_meter.avg, mIoU, mAcc, allAcc
    return epoch_loss


if __name__ == '__main__':
    # Load Config
    code_root = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    config_file_path = os.path.join(code_root, 'config/s3dis/s3dis_antenna.yaml')  # YAML config file
    args = get_parser(desc='ANTenna3DSeg', config_file=config_file_path)

    selected_epoch = 12

    # Init
    if args.manual_seed is not None:
        if device == 'cuda':
            cudnn.benchmark = False
            cudnn.deterministic = True
            torch.cuda.manual_seed_all(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
    if len(args.test_gpu) == 1:
        args.sync_bn = False

    # Apply parameters
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)

    # Logger
    data_root = args.data_root
    test_logger_path = os.path.join(data_root, args.test_log)
    ckpt_path = '/'.join(test_logger_path.split('/')[:-1])
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    logger = create_logger(test_logger_path, write_mode='w')
    logger.info(args)
    # Writer
    test_writer_path = os.path.join(data_root, args.test_tensorboard_path)
    if os.path.exists(test_writer_path):  # delete old one
        shutil.rmtree(test_writer_path)
        os.makedirs(test_writer_path)
    writer = SummaryWriter(test_writer_path)

    # launch tensorboard
    # os.system("tensorboard --logdir 'checkpoints' --port 9999 --bind_all")  # can not launch within docker

    # Dataset
    # Dataset - train
    train_transform = transform.Compose([transform.ToTensor()])
    train_data = S3DIS(split='train', data_root=os.path.join(data_root, args.train_full_folder),
                       num_point=args.num_point, test_area=args.test_area, block_size=args.block_size,
                       sample_rate=args.sample_rate, transform=train_transform, logger=logger)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True,
                                               num_workers=args.train_workers, pin_memory=True)

    # Dataset - val/test
    # test_transform = transform.Compose([transform.ToTensor()])
    # test_data = S3DIS(split='eval', data_root=os.path.join(data_root, args.test_full_folder),
    #                   num_point=args.num_point, test_area=args.test_area, block_size=args.block_size,
    #                   sample_rate=args.sample_rate, transform=test_transform, logger=logger)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True,
    #                                           num_workers=args.test_workers, pin_memory=True)

    # Model
    model = Model(in_dim=9, hidden_dim=64, out_dim=13)
    # Model parameter init

    criterion = nn.CrossEntropyLoss()

    if device == 'gpu':
        model.cuda()
        criterion.cuda()

    # Checkpoints
    if args.model_path:  # default=None
        model_root_path = os.path.join(data_root, args.model_path)
        model_path_list = os.listdir(model_root_path)
        # get selected epoch weight
        model_path = os.path.join(data_root, args.model_path, [p for p in model_path_list if 'train_epoch_{}'.format(selected_epoch) in p][0])
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

    # Test
    # save_path = os.path.join(data_root, args.save_path)
    # epoch_loss = test(model=model, test_loader=test_loader, criteration=criterion,
    epoch_loss = test(model=model, test_loader=train_loader, criteration=criterion,
                      logger=logger, writer=writer, device=device, args=args)
    #   filename = '{}/test_epoch_{}_loss_{:.3f}.pth'.format(save_path, epoch_log, epoch_loss)
    #   logger.info('Saving checkpoint to: ' + filename)
    #   torch.save({'epoch': epoch_log,
    #               'state_dict': model.state_dict(),
    #               'optimizer': optimizer.state_dict(),
    #               'scheduler': scheduler.state_dict()}, filename)
    print('finished testing')

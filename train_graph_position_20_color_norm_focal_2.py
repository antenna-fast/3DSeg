"""
Author: ANTenna on 2022/1/12 7:08 下午
aliuyaohua@gmail.com

Description:
Train: 3D Semantic Segmentation
"""

import os
import time
import shutil

import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from dataset.s3dis_short_knn import S3DIS  # dataset

from utils.metric_utils import AverageMeter, intersectionAndUnionGPU
from utils.logger_utils import create_logger
from utils.config import get_parser
from utils import transform


def train_one_epoch(model, train_loader, criteration, optimizer, epoch, logger, writer, device, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()

    num_batch = len(train_loader)
    running_loss = 0
    epoch_loss = 0
    for i, data in enumerate(train_loader):
        batch_start_time = time.time()
        x, target = data[0].to(device), data[1].to(device)

        # Forward -> loss -> backward -> optimize
        optimizer.zero_grad()
        output = model(x)  # output: [batch, samples, classes]
        output = output.permute(0, 2, 1)  # To [batch, classes, samples] for CrossEntropyLoss
        loss = criteration(output, target)  # Loss
        loss.backward()  # Backward
        optimizer.step()  # Optimizer

        # Batch metric
        output = output.max(1)[1]  # [batch, NumPoints]
        loss_item = loss.item()
        overall_batch = epoch * num_batch + i  # for tensorboard writer

        # Training time Metric
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        mIoU_train_batch = np.mean(intersection / (union + 1e-10))  # Mean on all class
        mAcc_train_batch = np.mean(intersection / (target + 1e-10))

        # Update Metric Buffer in class leval
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        loss_meter.update(loss.item(), x.size(0))
        batch_time.update(val=time.time() - batch_start_time)
        
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)  # Overall acc

        # Training Logger
        running_loss += loss_item
        epoch_loss += loss_item
        log_batch = i + 1
        if log_batch % args.print_freq == 0 and i != 0:
            logger.info('EPOCH:[{cur_e}/{max_e}] | Batch:[{cur_b}/{max_b}] | RunningLoss:{loss:.5f} | '
                        'AllAcc:{acc:.5f} | mIoU:{miou:.5f} | mAcc:{macc:.5f} | Time:{batch_time:.4f}'.
                        format(cur_e=epoch+1, max_e=args.epochs, cur_b=log_batch, max_b=num_batch,
                               loss=running_loss, acc=accuracy, miou=mIoU_train_batch,
                               macc=mAcc_train_batch, batch_time=batch_time.val))
            writer.add_scalar('TrainBatch/Loss', running_loss, overall_batch)
            writer.add_scalar('TrainBatch/Metric/mIoU', mIoU_train_batch, overall_batch)
            writer.add_scalar('TrainBatch/Metric/mAcc', mAcc_train_batch, overall_batch)
            writer.add_scalar('TrainBatch/Metric/AllAcc', accuracy, overall_batch)
            running_loss = 0

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # Epoch Logger
    epoch_log = epoch + 1
    logger.info('Train Result: Epoch:[{cur_e}/{max_e}] | EpochLoss:{eloss:5f} |'
                ' mIoU: {miou:.4f} | mAcc: {macc:.4f} | allAcc: {acc:.4f} .'.
                format(cur_e=epoch_log, max_e=args.epochs, eloss=epoch_loss, miou=mIoU, macc=mAcc, acc=allAcc))
    # Epoch Writer
    writer.add_scalar('TrainEpoch/Loss', epoch_loss, epoch_log)
    writer.add_scalar('TrainEpoch/Metric/mIoU', mIoU, epoch_log)
    writer.add_scalar('TrainEpoch/Metric/mAcc', mAcc, epoch_log)
    writer.add_scalar('TrainEpoch/Metric/AllAcc', allAcc, epoch_log)

    return epoch_loss


if __name__ == '__main__':
    # Load Config
    config_file_path = 'config/s3dis/s3dis_antenna_graph_position_20_color_norm_focal_2.yaml'  # YAML config file
    args = get_parser(desc='ANTenna3DSeg', config_file=config_file_path)
    
    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)    
    if args.cuda:
        device = torch.device("cuda")
        import torch.backends.cudnn as cudnn 
    else:
        device = torch.device("cpu")

    # Init
    if args.manual_seed is not None:
        if device == torch.device("cuda"):
            cudnn.benchmark = False
            cudnn.deterministic = True
            torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
    if len(args.train_gpu) == 1:
        args.sync_bn = False

    # Logger
    log_path = os.path.join(args.save_root, args.log_path, args.arch)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    train_logger_path = os.path.join(log_path, args.train_log)
    logger = create_logger(train_logger_path, write_mode='w')
    logger.info(args)

    # Writer
    writer_path = os.path.join(args.save_root, args.writer_path, args.arch)
    if os.path.exists(writer_path):  # delete ole one
        shutil.rmtree(writer_path)
        os.makedirs(writer_path)
    else: 
        os.makedirs(writer_path)
    train_writer_path = os.path.join(writer_path, args.train_tensorboard_path)
    writer = SummaryWriter(train_writer_path)

    # launch tensorboard in-code
    # os.system("tensorboard --logdir 'checkpoints' --port 9999 --bind_all")  # can not launch within docker

    # Dataset
    data_root = args.data_root
    # Dataset: train
    train_transform = transform.Compose([transform.ToTensor()])
    train_data = S3DIS(split='train', data_root=os.path.join(data_root, args.train_full_folder), num_point=args.num_point,
                       test_area=args.test_area, sample_rate=args.sample_rate, transform=train_transform, logger=logger)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True,
                                               num_workers=args.train_workers, pin_memory=True)
    # Dataset: val/test

    # Model
    arch = args.arch
    if arch == 'pointNN_graph':
        from model.pointNN_graph import NN as Model
    elif arch == 'pointNN':
        from model.pointNN import NN as Model
    elif arch == 'pointNN_graph_position':
        from model.pointNN_graph_position import NN as Model
    elif arch == 'pointNN_graph_position_20':
        from model.pointNN_graph_position_20 import NN as Model
    elif arch == 'pointNN_graph_position_20_color_norm':
        from model.pointNN_graph_position_20_color_norm import NN as Model
    elif arch == 'pointNN_graph_position_20_color_norm_focal_2':
        from model.pointNN_graph_position_20_color_norm import NN as Model
    else:
        raise Exception("Architecture {} NOT implemented!".format(arch))
    model = Model(in_dim=args.feature_dim, hidden_dim=64, out_dim=args.classes)

    loss_function = args.loss_function
    if loss_function == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif loss_function == 'FocalLoss':
        from model.losses import FocalLoss
        criterion = FocalLoss(gamma=args.focal_gamma)
    else:
        raise Exception('Loss Function {} NOT implemented!'.format(loss_function))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_epoch, gamma=args.multiplier)

    # To Device
    if device == torch.device("cuda"):
        model = nn.DataParallel(model)
        criterion.cuda()
        logger.info("Let's use {} GPU!".format(torch.cuda.device_count()))
    
    # Checkpoints path
    checkpoints_path = os.path.join(args.save_root, args.checkpoints_path, args.arch)
    if args.weight:  # default=None
        weight_path = os.path.join(checkpoints_path, args.weight)
        if os.path.isfile(weight_path):
            logger.info("=> loading weight '{}'".format(weight_path))
            checkpoint = torch.load(weight_path)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(weight_path))
        else:
            logger.info("=> no weight found at '{}'".format(weight_path))
    # else init Model parameter manually

    # Load Checkpoint
    if args.resume:
        resume_path = os.path.join(checkpoints_path, args.resume)
        if os.path.isfile(resume_path):
            logger.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(resume_path))
    else:  # if checkpoint not given
        if args.is_restart:  # Delete the old checkpoints
            if os.path.exists(checkpoints_path):
                shutil.rmtree(checkpoints_path)
            if not os.path.exists(checkpoints_path):
                os.makedirs(checkpoints_path)
         
    # Train All Epoch
    for epoch in range(args.start_epoch, args.epochs):
        # Train One Epoch
        epoch_loss = train_one_epoch(model=model, train_loader=train_loader, criteration=criterion, optimizer=optimizer,
                                     epoch=epoch, logger=logger, writer=writer, device=device, args=args)
        scheduler.step()
        epoch_log = epoch + 1  # start from 1

        # Save Checkpoints
        if epoch_log % args.save_freq == 0:  
            filename = '{}/train_epoch_{}_loss_{:.3f}.pth'.format(checkpoints_path, epoch_log, epoch_loss)
            logger.info('Saving checkpoint to: {}'.format(filename))
            torch.save({'epoch': epoch_log,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}, filename)
    logger.info('Finished training!')

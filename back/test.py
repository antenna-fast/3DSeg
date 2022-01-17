"""
Author: ANTenna on 2022/1/13 3:05 下午
aliuyaohua@gmail.com

Description:

"""

import os
import time
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter


def test(dataloader, model, logger):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    model.eval()
    # metric


if __name__ == '__main__':
    # parameters
    args = 0

    # logger
    logger = 0
    writer = SummaryWriter(log_dir='')

    # dataset
    testset = 0
    dataloader = 0

    # model
    model = 0

    # checkpoints
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    test()

"""
Author: ANTenna 
Date: 2022/1/26
E-mail: aliuyaohua@gmail.com

Description:
Loss Functions
"""

import torch
import torch.nn as nn


# FocalLoss
class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        a = 0

    def forward(self):
        b = 0
        return 0


# DiceLoss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        input, target = input.cuda(), target.cuda()
        smooth = 1

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()  # number of mask TODO: for binary mask? or, need to trans to class level?
        dice_coef = (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
        return 1 - dice_coef


if __name__ == '__main__':
    print('unit test code')

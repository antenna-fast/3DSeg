"""
Author: ANTenna 
Date: 2022/1/26
E-mail: aliuyaohua@gmail.com

Description:
Loss Functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        # gamma:l
        # alpha: defines the class imbalance [0, 1]
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        # if isinstance(alpha, (float, int, long)):
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        # input pre-process to adapt standard 1-dim loss function
        if input.dim() > 2:  # [N, C, *]
            # N: batch size
            # C: class number
            # *: W, H, ..
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W;
            input = input.transpose(1, 2).contiguous()  # N,C,H*W(*) => N,H*W(*),C
            input = input.view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        # target pre-process
        target = target.view(-1, 1)  # [N, *] -> [Nx*, 1], -1 dim indicate the class index

        logpt = F.log_softmax(input)  # log(softmax(x))
        logpt = logpt.gather(1, target)  # gather values by target indexes
        logpt = logpt.view(-1)  # N,
        pt = Variable(logpt.data.exp())  # to (0, 1)

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))  # TODO:?
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


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

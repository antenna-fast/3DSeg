"""
Author: ANTenna 
Date: 2022/1/17
E-mail: aliuyaohua@gmail.com

Description:
copy from common_util.py
"""

import numpy as np
import torch


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    """
    :param output: Model inference result after argmax
    :param target: Label of each point
    :param K: Class number
    :param ignore_index: Default is 255
    :return: Class Level Intersection and Union
    """
    # 'K' classes, output and target sizes are N or N * L or N * H * W,
    # each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    # output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output = output.copy()
    target = target.copy()
    output[np.where(target == ignore_index)[0]] = ignore_index  # Set the ignore_index is equal in both target and output
    intersection_idx = np.where(output == target)[0]
    intersection = output[intersection_idx]  # Intersection points' label(TP + TN)
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))  # Return the number of each class
    area_output, _ = np.histogram(output, bins=np.arange(K+1))  # K+1是考虑到了其实点为0
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


if __name__ == '__main__':
    print('unit test code')

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    # x: [batch, dim, num_points]
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)  # [batch, dim, num_samples]
    
    # get KNN using input feature (dynanmic)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)  # 每一行，包含k个近邻
    
    device = torch.device('cuda')
    # idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points  # expand dimension
    idx = idx + idx_base
    idx = idx.view(-1)  # to 1D
    
    _, num_dims, _ = x.size()   # [batch, feat_dim, num_sample]
    
    x = x.transpose(2, 1).contiguous()  # [batch, num_points, dim] 
    feature = x.view(batch_size*num_points, -1)[idx, :]  # 相同的view方式，根据idx拿出来对应的sample
    feature = feature.view(batch_size, num_points, k, num_dims)  # 每个point，都带有k个nn sample的features
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # expand dim, and repeat
    # x: [batch, num_point, k, feat_dim]    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # cat edge feature and itself
    # return: [batch, feat_dim x 2, num_point, k]
    return feature 


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        # Conv2D: channel-wise pooling
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        # Conv1D
        # 512 = cat[256, 128, 64, 64]
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
 
    def forward(self, x):
        # x: [batch, feat_dim, num_points]
        batch_size = x.size(0)
        
        # dynamic knn feature
        # input: [x, k, idx]
        # return: [batch, feat_dim * 2, num_point, k]
        x = get_graph_feature(x, k=self.k)  
        x = self.conv1(x)  # 2D Conv with kernel_size=1: channel-wise pooling: 6 -> 64
        x1 = x.max(dim=-1, keepdim=False)[0]  # [batch, feat_dim, num_point]  把邻居pooling掉了[??] 确定x的dim

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)  # Multi-layer feature: cat on feature_dim
        # x: [batch, cat_feat, n_point]
        x = self.conv5(x)  # Conv1D, return: [batch, feat_dim, n_point]
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        # Shared 3 layer MLP
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x  # [batch, ]


if __name__ == "__main__":
    print("model test")

    test_data = torch.tensor([[0, 0],
                            [1, 1],
                            [3, 3],
                            [4, 4]]).unsqueeze(0).transpose(2, 1)
    print(test_data)
    x = get_graph_feature(test_data, k=2)
    print(x)
    print()

"""
Author: ANTenna on 2022/1/13 3:48 下午
aliuyaohua@gmail.com

Description:
Model for 3D semantic segmentation

Introduce DGCNN for segmentation
without color information
k=5
"""

import torch
import torch.nn as nn
import faiss
import numpy as np


class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # Define model elements here
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.top_k = 5

        # Faiss KNN
        self.res = faiss.StandardGpuResources()  # use a single GPU
        self.index_cpu = faiss.IndexFlatL2(3)  # build a flat(CPU) L2
        self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.index_cpu)  # To GPU [res, gpu_id, index(cpu)]
        print("index is trained: ", self.gpu_index.is_trained)

        # Shared MLP for point cloud learning
        # self.shared_mlp = nn.Sequential(nn.Linear(self.in_dim, self.hidden_dim), nn.ReLU(),
        self.shared_mlp = nn.Sequential(nn.Linear(64, self.hidden_dim), nn.ReLU(),
                                        nn.Linear(self.hidden_dim, 128), nn.ReLU(),
                                        nn.Linear(128, 256), nn.ReLU(),
                                        nn.Linear(256, 128), nn.ReLU(),
                                        nn.Linear(128, self.hidden_dim), nn.ReLU(),
                                        nn.Linear(self.hidden_dim, self.out_dim),
                                        )
        # edge feature linear projection
        self.edge_projection = nn.Linear(3, 64)

        # channel pooling
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(13)

        # Conv2D: channel-wise pooling
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=1, bias=False),
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

    # Model Architecture
    def forward(self, x):
        assert x.shape[-1] == self.in_dim, \
            'ERROR Input Data Shape:{}, Expected feature dimension:{} .. '.format(x.shape, self.in_dim)

        points = x[:, :, 0:3]  # .transpose(2, 1)  # [batch, dim, num_points]
        colors = x[:, :, 3:6]

        # get knn for batch samples
        batch_edge_feature = []
        for sample_idx, p in enumerate(points):
            p = np.array(p.cpu(), dtype=np.float32)
            num_points = len(p)

            # faiss
            gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.index_cpu)  # To GPU [res, gpu_id, index(cpu)]
            # print("index is trained: ", gpu_index.is_trained)

            gpu_index.add(p)  # add vector to the index
            sims, nbrs = gpu_index.search(p, k=self.top_k)
            # print(f'batch search finished: [{sample_idx+1}/{batch_size}]')
            # get knn features by index
            nbrs_flat = nbrs.reshape(-1)
            points_knn_flat = p[nbrs_flat].reshape(num_points, -1, 3)  # [N, k, 3]
            points_knn_flat = torch.tensor(points_knn_flat)
            p_center = torch.tensor(p).view(num_points, 1, 3).repeat(1, self.top_k, 1)
            knn_edge = points_knn_flat - p_center  # [N, k, 3]
            batch_edge_feature.append(knn_edge.numpy())

        batch_edge_feature = torch.tensor(np.array(batch_edge_feature)).cuda()  # [batch, N, k, 3]
        trans_edge_feature = batch_edge_feature.permute(0, 3, 1, 2).contiguous()  # [batch, feat, N, K]
        x = self.conv1(trans_edge_feature)  # [batch, feat_dim, N, K]
        x1 = x.max(dim=-1, keepdim=False)[0]  # [batch, feat_dim, N]
        x1 = x1.permute(0, 2, 1)  # [batch, N, feat_dim]

        x = self.shared_mlp(x1)
        return x


if __name__ == '__main__':
    test_data = torch.ones(2, 100, 3)
    test_label = torch.ones(2, 100)
    model = NN(3, 64, 10)
    output = model(test_data)

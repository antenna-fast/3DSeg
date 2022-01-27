"""
Author: ANTenna on 2022/1/13 3:48 下午
aliuyaohua@gmail.com

Description:
Model for 3D semantic segmentation

Introduce DGCNN for segmentation
"""

import torch
import torch.nn as nn
import faiss
import numpy as np


# Borrowed from DGCNN
def knn(x, k):
    # return: [batch_size, num_points, k]
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    # x: [batch, dim, num_points]
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)  # [batch, dim, num_samples]

    # get KNN using input feature (dynanmic)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)  # 每一行，包含k个近邻

    device = torch.device('cuda')
    # idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points  # expand dimension
    idx = idx + idx_base
    idx = idx.view(-1)  # to 1D idx

    _, num_dims, _ = x.size()  # [batch, feat_dim, num_sample]

    x = x.transpose(2, 1).contiguous()  # [batch, xum_points, dim] 
    feature = x.view(batch_size * num_points, -1)[idx, :]  # 相同的view方式，根据idx拿出来对应的sample
    feature = feature.view(batch_size, num_points, k, num_dims)  # 每个point，都带有k个nn sample的features
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # expand dim, and repeat

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature  # [batch, num_point, k, feat_dim]


class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # Define model elements here
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.top_k = 20

        # faiss KNN
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
        # self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=1, bias=False),
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

    # Model Architecture
    def forward(self, x):
        assert x.shape[-1] == self.in_dim, \
            'ERROR Input Data Shape:{}, Expected feature dimension:{} .. '.format(x.shape, self.in_dim)

        points = x[:, :, 0:3]  # [batch, num_points, dim]
        colors = x[:, :, 3:6]

        # get knn for batch samples
        batch_edge_feature = []
        batch_knn_colors = []

        for sample_idx, (p, c) in enumerate(zip(points, colors)):
            p = np.array(p.cpu(), dtype=np.float32)
            c = np.array(c.cpu(), dtype=np.float32)
            num_points = len(p)

            # Faiss
            gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.index_cpu)  # To GPU [res, gpu_id, index(cpu)]
            # print("index is trained: ", gpu_index.is_trained)

            gpu_index.add(p)  # add vector to the index
            sims, nbrs = gpu_index.search(p, k=self.top_k)
            # print(f'batch search finished: [{sample_idx+1}/{batch_size}]')
            # get knn features by index
            # nbrs: [N, k]
            nbrs_flat = nbrs.reshape(-1)  # to Nxk
            points_knn_flat = p[nbrs_flat].reshape(num_points, -1, 3)  # reshape: [N, k, 3]
            points_knn_flat = torch.tensor(points_knn_flat)

            colors_knn = c[nbrs_flat].reshape(num_points, -1, 3)
            batch_knn_colors.append(colors_knn)

            p_center = torch.tensor(p).view(num_points, 1, 3).repeat(1, self.top_k, 1)
            p_center[:, 0, :] = torch.tensor([0, 0, 0], dtype=torch.float)  # to zero, preserve itself
            knn_edge = points_knn_flat - p_center  # [N, k, 3], [position, knn_edges]
            batch_edge_feature.append(knn_edge.numpy())

        batch_edge_feature = torch.tensor(np.array(batch_edge_feature)).cuda()  # [batch, N, k, 3]
        batch_edge_colors = torch.tensor(np.array(batch_knn_colors)).cuda()
        batch_edge_feature = torch.concat([batch_edge_feature, batch_edge_colors], dim=-1)  # [batch, N, k, 6]

        trans_edge_feature = batch_edge_feature.permute(0, 3, 1, 2).contiguous()  # [batch, feat, N, K]
        x = self.conv1(trans_edge_feature)  # [batch, feat_dim, N, K]
        x1 = x.max(dim=-1, keepdim=False)[0]  # [batch, feat_dim, N]  # max pooling
        x1 = x1.permute(0, 2, 1)  # [batch, N, feat_dim]

        x = self.shared_mlp(x1)
        return x


if __name__ == '__main__':
    test_data = torch.ones(2, 100, 3)
    test_label = torch.ones(2, 100)
    model = NN(3, 64, 10)
    output = model(test_data)

"""
PointNet++
PointWeb
PAConv

We, do NOT sample!
We, use FULL data to train and test!
"""

import os
import sys
import numpy as np
import random
import time
import torch
from torch.utils.data import Dataset


class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=None,
                 test_area=5, block_size=1.0, sample_rate=1.0, transform=None, logger=None):
        super().__init__()
        # self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        for room_name in rooms_split:  # For each room
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgb, label, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # N*6: xyz rgb; N,: label
            self.room_points.append(points)  # 缓存了所有的点！xyz rgb
            self.room_labels.append(labels)
            # Bbox corner
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min)
            self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)  # Number of points

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]  # N * 6
        labels = self.room_labels[room_idx]  # N
        return points, labels

    def __len__(self):
        return len(self.room_idxs)


if __name__ == '__main__':
    data_root = '/Users/aibee/Downloads/Paper/Point Cloud/Semantic Segmentation/Dataset/s3dis/trainval_fullarea'

    # parameters
    stage = 'train'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = S3DIS(split=stage, data_root=data_root, num_point=num_point, test_area=test_area,
                       block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)

    # set manual seed, to make sure the result is reproducible
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)

    train_loader = torch.utils.data.DataLoader(point_data, batch_size=2, shuffle=True,
                                               num_workers=2, pin_memory=True,
                                               worker_init_fn=worker_init_fn)

    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()

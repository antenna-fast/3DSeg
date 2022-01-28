"""
PointNet++
PointWeb
PAConv
"""

# TODO: in init, only get file path list, and do not load real data

import os
import sys
import numpy as np
import random
import time
import torch
from torch.utils.data import Dataset


class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=10000,
                 test_area=5, sample_rate=1.0, transform=None, logger=None):
        super().__init__()

        if logger:
            self.logger = logger
            self.logger.info('S3DIS init ... ')
        self.num_point = num_point
        self.transform = transform
        self.split = split
        
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if self.split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        label_weights = np.zeros(13)  # get prior of each class

        for room_name in rooms_split:  # For each room
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyz rgb label, N*7  # TODO: move to get_item
            points, labels = room_data[:, 0:6], room_data[:, 6]  # N*6: xyz rgb; N,: label
            self.room_points.append(points)
            self.room_labels.append(labels)
            # get class weights prior
            tmp, _ = np.histogram(labels, range(14))  # [0, 1), [1, 2), ... [12, 13), label:[0, .., 12]
            label_weights += tmp

            num_point_all.append(labels.size)  # Number of points
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min)
            self.room_coord_max.append(coord_max)  # bbox corner
        # label weights
        label_weights = label_weights.astype(np.float32)
        label_weights = label_weights / np.sum(label_weights)
        # amax / label_weight: how many times
        # x^-3: re-scale weights
        self.label_weights = np.power(np.amax(label_weights) / label_weights, 1 / 3.0)
        print('label_weights: '.format(self.label_weights))

        sample_prob = num_point_all / np.sum(num_point_all)  # NumPoints / points num, ratio of each scene
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)

        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        logger.info("Totally {} samples in {} set.".format(len(self.room_idxs), split))
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]  # N * 6
        labels = self.room_labels[room_idx]  # N, 

        # data pre-processing both for training and testing

        # sampling
        N_points = points.shape[0]  # N
        point_idxs = np.arange(N_points)
        if (N_points > self.num_point) and self.split == "train":
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)  # may cause random?
            points = points[selected_point_idxs]
            labels = labels[selected_point_idxs]

        # normalize

        # centralize  # however, if we use the whole points? is it necessary？

        # to tensor
        return torch.from_numpy(points).float(), torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.room_idxs)


if __name__ == '__main__':
    data_root = '/data0/texture_data/yaohualiu/PublicDataset/s3dis/trainval_fullarea'
    stage = 'test'
    num_point, test_area, sample_rate = 4096, 5, 0.01

    point_data = S3DIS(split=stage, data_root=data_root, num_point=num_point, test_area=test_area,
                       sample_rate=sample_rate, transform=None)
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

    train_loader = torch.utils.data.DataLoader(point_data, batch_size=2, shuffle=True, num_workers=2,
                                               pin_memory=True, worker_init_fn=worker_init_fn)

    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()

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

import open3d as o3d


def show_points(points_xyzrgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyzrgb[:, :3])
    pcd.paint_uniform_color([1, 0.706, 0])
    # pcd.colors = o3d.utility.Vector3dVector(points_xyzrgb[:, 3:6] / 255)
    pcd.colors = o3d.utility.Vector3dVector(points_xyzrgb[:, 3:6])
    print('points: {}'.format(len(pcd.points)))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=10))
    o3d.visualization.draw_geometries([pcd])


class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=10000, test_area=5,
                 block_size=1.0, sample_rate=1.0, transform=None, logger=None):
        super().__init__()
        self.logger = logger
        self.logger.info('S3DIS init ... ')
        self.num_point = num_point
        self.block_size = block_size
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
        for room_name in rooms_split:  # For each room
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgb, label, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # N*6: xyz rgb; N,: label
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points)
            self.room_labels.append(labels)
            self.room_coord_min.append(coord_min)
            self.room_coord_max.append(coord_max)  # bbox corner
            num_point_all.append(labels.size)  # Number of points
        sample_prob = num_point_all / np.sum(num_point_all)  # NumPoints / points num, ratio of each scene
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)

        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        logger.info("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]  # N * 6
        labels = self.room_labels[room_idx]  # N, 
        N_points = points.shape[0]
        
        point_idxs = np.arange(N_points) 

        if (N_points > self.num_point) and self.split == "train":
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
            points = points[selected_point_idxs]
            labels = labels[selected_point_idxs]
            
        return torch.from_numpy(points).float(), torch.from_numpy(labels).long()

        # NOT USE BELOW
        # while True:
        #     # TODO: random sample as center?
        #     center = points[np.random.choice(N_points)][:3]
        #     block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
        #     block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
        #     point_idxs = np.where((points[:, 0] >= block_min[0]) &  # xy在范围内的点的索引
        #                           (points[:, 0] <= block_max[0]) &
        #                           (points[:, 1] >= block_min[1]) &
        #                           (points[:, 1] <= block_max[1]))[0]
        #     if point_idxs.size > 1024:
        #         break
        # Sample
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)
        selected_points = points[selected_point_idxs, :]  # NumPoints x 6

        # current_points[6:9]: Normalized xyz
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]

        # selected_points[0:6]: Centralize xy+z, Normalize color
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]  # centralized xyz
        selected_points[:, 3:6] /= 255.0  # color Normalize

        # current_points[0:6]: [centralized xyz, normalized rgb]
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]

        # Transform
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        # current_points: Nx9: [Centralized xyz, Normalized rgb, Normalized xyz]
        # current_labels: N, [label of each point]
        return current_points, current_labels

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

"""
Author: ANTenna on 2022/1/13 5:26 下午
aliuyaohua@gmail.com

Description:

"""

import os
import sys
import numpy as np


def data_prepare(room_path, args):
    room_data = np.load(room_path)
    points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
    coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
    stride = args.block_size * args.stride_rate
    grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - args.block_size) / stride) + 1)
    grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - args.block_size) / stride) + 1)
    data_room, label_room, index_room = np.array([]), np.array([]), np.array([])
    for index_y in range(0, grid_y):
        for index_x in range(0, grid_x):
            s_x = coord_min[0] + index_x * stride
            e_x = min(s_x + args.block_size, coord_max[0])
            s_x = e_x - args.block_size
            s_y = coord_min[1] + index_y * stride
            e_y = min(s_y + args.block_size, coord_max[1])
            s_y = e_y - args.block_size
            point_idxs = np.where((points[:, 0] >= s_x - 1e-8) & (points[:, 0] <= e_x + 1e-8) & (points[:, 1] >= s_y - 1e-8) & (points[:, 1] <= e_y + 1e-8))[0]
            if point_idxs.size == 0:
                continue
            num_batch = int(np.ceil(point_idxs.size / args.num_point))
            point_size = int(num_batch * args.num_point)
            replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
            point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
            point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
            np.random.shuffle(point_idxs)
            data_batch = points[point_idxs, :]
            normlized_xyz = np.zeros((point_size, 3))
            normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
            normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
            normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
            data_batch[:, 0] = data_batch[:, 0] - (s_x + args.block_size / 2.0)
            data_batch[:, 1] = data_batch[:, 1] - (s_y + args.block_size / 2.0)
            data_batch[:, 3:6] /= 255.0
            data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
            label_batch = labels[point_idxs]
            data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
            label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
            index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
    assert np.unique(index_room).size == labels.size
    return data_room, label_room, index_room, labels


def data_load(data_path, args):
    data = np.load(data_path)  # xyz rgb l, N*7
    coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]  # xyz, rgb, label

    idx_data = []
    # Voxelize
    if args.voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        idx_sort, count = voxelize(coord, args.voxel_size, mode=1)
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            idx_data.append(idx_part)
    else:  # NOT Voxelize
        idx_data.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_data


# load Test data
def get_data_list(args):
    if args.data_name == 's3dis':
        data_list = sorted(os.listdir(os.path.join(args.data_root, args.train_full_folder)))
        # [:-4] is xx in xx.npy, data_list: [Area_5*]
        data_list = [item[:-4] for item in data_list if 'Area_{}'.format(args.test_area) in item]
    else:
        raise Exception('dataset not supported yet'.format(args.data_name))

    print("Totally {} samples in val set.".format(len(data_list)))
    return data_list


def input_normalize(coord, feat):
    coord_min = np.min(coord, 0)
    coord -= coord_min  # make sure all points' coordinate > 0
    feat = feat / 255.  # color normalize
    return coord, feat


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


# Voxelize
def voxelize(coord, voxel_size=0.05, hash_type='fnv', mode=0):
    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    if mode == 0:  # train mode
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_unique = idx_sort[idx_select]
        return idx_unique
    else:  # val mode
        return idx_sort, count


if __name__ == '__main__':
    print('Unit test code ...')

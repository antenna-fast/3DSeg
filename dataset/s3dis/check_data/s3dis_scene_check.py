"""
Author: ANTenna 
Date: 2022/1/27
E-mail: aliuyaohua@gmail.com

Description:
Check the different part in s3dis: they use relative coordinate or absolute coordinate
method:
Stack different part, and visualize it
"""

import os
import numpy as np


if __name__ == '__main__':
    print('unit test code')
    vis_area = 5
    selected_num = 8
    res_path = '/data0/texture_data/yaohualiu/PublicDataset/s3dis/check_data'

    data_root = '/data0/texture_data/yaohualiu/PublicDataset/s3dis'
    train_full_folder = 'trainval_fullarea'
    data_list = sorted(os.listdir(os.path.join(data_root, train_full_folder)))
    data_list = [item[:-4] for item in data_list if 'Area_{}'.format(vis_area) in item][:selected_num]
    num_data = len(data_list)

    data_rooms = np.array([])
    label_rooms = np.array([])
    for i, data_name in enumerate(data_list):
        print('processing [{}/{}]'.format(i+1, num_data))
        data_path = os.path.join(data_root, train_full_folder, data_name + '.npy')
        data = np.load(data_path)  # xyz rgb l, N*7
        coord, color, label = data[:, :3], data[:, 3:6], data[:, 6]  # xyz, rgb, label
        # Accumulate
        coord_color = np.concatenate((coord, color), axis=1)
        data_rooms = np.vstack([data_rooms, coord_color]) if data_rooms.size else coord_color
        label_rooms = np.hstack([label_rooms, label]) if label_rooms.size else label

    # assert np.unique(index_room).size == labels.size
    print('saving stacked room ... ')
    np.save(os.path.join(res_path, 'stack_xyzrgb.npy'), data_rooms)
    np.save(os.path.join(res_path, 'stack_labels.npy'), label_rooms)
    print('done.')

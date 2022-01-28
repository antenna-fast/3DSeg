import os
import open3d as o3d
import numpy as np


if __name__ == '__main__':
    data_path = '/Users/aibee/PycharmProjects/pythonProject/3DSemantic/ANTennaSeg3D/show_inference/check_data/stack_xyzrgb.npy'
    data = np.load(data_path)  # xyz rgb l, N*7
    # coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]  # xyz, rgb, label
    coord, feat = data[:, :3], data[:, 3:6]/255  # xyz, rgb, label

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(feat)
    o3d.visualization.draw_geometries([pcd], window_name='3D')

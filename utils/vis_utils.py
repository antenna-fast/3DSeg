"""
Author: ANTenna on 2022/1/15 1:54 下午
aliuyaohua@gmail.com

Description:
For visualization
"""

import open3d as o3d

# DEBUG UTILS


def show_inference(x, label):
    # x: [batch, points, features]
    # label: [batch, points, classes]
    points = x[:, 0:3]
    colors = x[:, 3:6]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    print('unit test code')

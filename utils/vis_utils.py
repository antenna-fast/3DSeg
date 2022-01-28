"""
Author: ANTenna on 2022/1/15 1:54 下午
aliuyaohua@gmail.com

Description:
For visualization
"""

import open3d as o3d


def show_points_xyz(points_xyz, win_name='ANTenna3D'):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    print('points: {}'.format(len(pcd.points)))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=10))
    o3d.visualization.draw_geometries([pcd], window_name=win_name)


def show_points_xyzrgb(points_xyzrgb, is_color_norm=0, win_name='ANTenna3D'):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyzrgb[:, :3])
    print('points: {}'.format(len(pcd.points)))
    pcd.colors = o3d.utility.Vector3dVector(points_xyzrgb[:, 3:6] / 255) if is_color_norm else o3d.utility.Vector3dVector(points_xyzrgb[:, 3:6])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=10))
    o3d.visualization.draw_geometries([pcd], window_name=win_name)


def show_inference(points, colors, is_norm, is_show=0, is_save=0, save_path=None):
    # points: num_points x 3
    # label: num_points x 3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if is_norm:
        colors = colors / 255
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    if is_show:
        o3d.visualization.draw_geometries([pcd])

    if is_save:
        o3d.io.write_point_cloud(save_path, pcd)


if __name__ == '__main__':
    print('unit test code')

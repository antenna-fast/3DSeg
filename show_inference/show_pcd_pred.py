import os
import open3d as o3d


def show_pointcloud(pcd_path, win_name='ANTenna3D'):
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(pcd_path)
    print('points: ', len(pcd.points))
    print('colors: ', len(pcd.colors))
    o3d.visualization.draw_geometries([pcd], window_name=win_name)


if __name__ == '__main__':
    method = 'pointNN_graph_position_20_color_norm'

    # mode = 'gt'
    mode = 'pred'
    # mode = 'color'
    scene = '1'
    pcd_path = 'pred_pcd/{}/{}_scene_{}.ply'.format(method, mode, scene)
    show_pointcloud(pcd_path=pcd_path, win_name=mode)

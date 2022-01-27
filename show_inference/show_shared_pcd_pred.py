import os
import open3d as o3d


def show_pointcloud(pcd_path, win_name='ANTenna3D'):
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(pcd_path)
    print('points: ', len(pcd.points))
    print('colors: ', len(pcd.colors))
    o3d.visualization.draw_geometries([pcd], window_name=win_name)


def shared_view(pcd_list, win_name=['1', '2']):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=win_name[0], width=960, height=540, left=0, top=0)
    vis.add_geometry(pcd_list[0])

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name=win_name[1], width=960, height=540, left=960, top=0)
    vis2.add_geometry(pcd_list[1])

    while True:
        vis.update_geometry(pcd_list[0])
        if not vis.poll_events():
            break
        vis.update_renderer()

        vis2.update_geometry(pcd_list[1])
        if not vis2.poll_events():
            break
        vis2.update_renderer()

    vis.destroy_window()
    vis2.destroy_window()


if __name__ == '__main__':
    method = 'pointNN_graph_position_20_color_norm'
    scene = 2
    pred_pcd_path = 'pred_pcd/{}/{}_scene_{}.ply'.format(method, 'pred', scene)
    pred_pcd = o3d.io.read_point_cloud(pred_pcd_path)

    gt_pcd_path = 'pred_pcd/{}/{}_scene_{}.ply'.format(method, 'gt', scene)
    gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)

    shared_view(pcd_list=[pred_pcd, gt_pcd], win_name=['pred', 'gt'])

    # show_pointcloud(pcd_path=pcd_path, win_name=mode)

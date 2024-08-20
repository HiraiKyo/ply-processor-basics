import numpy as np
import open3d as o3d
import pytest

from ply_processor_basics.points.convex_hull import detect_plane_edge
from ply_processor_basics.points.ransac import detect_plane


@pytest.mark.parametrize("plypath", ["data/samples/sample_circle.ply"])
@pytest.mark.visual
def test_visualize(plypath):
    pcd = o3d.io.read_point_cloud(plypath)
    points = np.asarray(pcd.points)
    inliers, plane_model = detect_plane(points, 1.0)
    edge_inliers = detect_plane_edge(points[inliers], plane_model)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points[inliers][edge_inliers])
    pcd.paint_uniform_color([0, 0, 0])
    pcd2.paint_uniform_color([0, 1, 0])

    # 点群とConvex Hullの可視化
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Convex Hull Visualization", width=800, height=600)
    vis.add_geometry(pcd)
    vis.add_geometry(pcd2)

    # Convex Hullの頂点のサイズを大きくする
    opt = vis.get_render_option()
    opt.point_size = 5  # 通常の点のサイズ

    # カメラの位置を調整
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)

    vis.run()
    vis.destroy_window()


@pytest.mark.parametrize("plypath", ["data/samples/sample_circle.ply"])
def test_success(plypath):
    pcd = o3d.io.read_point_cloud(plypath)
    points = np.asarray(pcd.points)
    inliers, plane_model = detect_plane(points)
    edge_inliers = detect_plane_edge(points[inliers], plane_model)

    assert len(points[inliers][edge_inliers]) > 0


@pytest.mark.parametrize("plypath", ["data/stained/segmented.ply"])
@pytest.mark.skip
def test_visualize_realdata(plypath):
    pcd = o3d.io.read_point_cloud(plypath)
    points_raw = np.asarray(pcd.points)
    points = points_raw
    for i in range(3):
        inliers, plane_model = detect_plane(points, 1.0)
        pcd__ = o3d.geometry.PointCloud()
        pcd__.points = o3d.utility.Vector3dVector(points[inliers])
        o3d.visualization.draw_geometries([pcd__])

        outliers = np.setdiff1d(np.arange(points.shape[0]), inliers)
        pcd_ = o3d.geometry.PointCloud()
        pcd_.points = o3d.utility.Vector3dVector(points[outliers])
        o3d.visualization.draw_geometries([pcd_])

        points = points[outliers]

    inliers, plane_model = detect_plane(points, 1.0)
    points = points[inliers]
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd_])
    edge_inliers = detect_plane_edge(points, plane_model)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points[edge_inliers])
    pcd.paint_uniform_color([0, 0, 0])
    pcd2.paint_uniform_color([0, 1, 0])

    # 点群とConvex Hullの可視化
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Convex Hull Visualization", width=800, height=600)
    vis.add_geometry(pcd)
    vis.add_geometry(pcd2)

    # Convex Hullの頂点のサイズを大きくする
    opt = vis.get_render_option()
    opt.point_size = 8  # 通常の点のサイズ

    # カメラの位置を調整
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)

    vis.run()
    vis.destroy_window()

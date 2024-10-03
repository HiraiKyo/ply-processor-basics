import numpy as np
import open3d as o3d
import pytest

from ply_processor_basics.points.detect_hole import detect_hole_in_plane
from ply_processor_basics.points.ransac import detect_plane

test_pcd = o3d.io.read_point_cloud("data/samples/hole.ply")
test_points = np.asarray(test_pcd.points)
test_radius = 17.5


@pytest.mark.parametrize("points", [test_points])
@pytest.mark.parametrize("hole_radius", [test_radius])
@pytest.mark.visual
def test_visualize(points, hole_radius):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

    plane_inliers, plane_model = detect_plane(points, threshold=0.1, minPoints=10000)
    pcd_plane = o3d.geometry.PointCloud()
    pcd_plane.points = o3d.utility.Vector3dVector(points[plane_inliers])
    o3d.visualization.draw_geometries([pcd_plane])

    hole_inliers, hole_center, hole_normal, radius = detect_hole_in_plane(
        points[plane_inliers], plane_model, hole_radius
    )

    # 結果を描画
    pcd.paint_uniform_color([0, 0, 0])
    hole_pcd = o3d.geometry.PointCloud()
    hole_pcd.points = o3d.utility.Vector3dVector(points[plane_inliers[hole_inliers]])
    hole_pcd.paint_uniform_color([0, 1, 0])
    # 穴の軸を描画
    axis = o3d.geometry.PointCloud()
    axis.paint_uniform_color([1, 0, 0])
    t = np.linspace(-100, 100, 100)
    line_points = hole_center.reshape(1, -1) + np.outer(t, hole_normal)
    axis.points = o3d.utility.Vector3dVector(line_points)
    o3d.visualization.draw_geometries([pcd, hole_pcd, axis])


@pytest.mark.parametrize("points", [test_points])
@pytest.mark.parametrize("hole_radius", [test_radius])
def test_success(points, hole_radius):
    plane_inliers, plane_model = detect_plane(points, threshold=0.5, minPoints=10000)
    hole_inliers, hole_center, hole_normal, radius = detect_hole_in_plane(
        points[plane_inliers], plane_model, hole_radius
    )
    assert np.allclose(hole_center, [50, 30, 50], atol=1e-0)

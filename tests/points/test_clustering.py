import numpy as np
import open3d as o3d
import pytest

from ply_processor_basics.points import plane_clustering
from ply_processor_basics.points.ransac import detect_plane


@pytest.mark.parametrize("plypath", ["data/samples/sample_clustering.ply"])
def test_success(plypath):
    pcd = o3d.io.read_point_cloud(plypath)
    points = np.asarray(pcd.points)

    inliers, plane_model = detect_plane(points, threshold=1.0)
    # EPS=10.0でサンプルデータ点群のクラスタリングが正常動作した事を確認
    clusters_indices = plane_clustering(points[inliers], eps=10.0)
    # クラスタが2つ認識されることを期待
    assert len(clusters_indices) == 2


@pytest.mark.parametrize("plypath", ["data/samples/sample_clustering.ply"])
@pytest.mark.visual
def test_visualize(plypath):
    pcd = o3d.io.read_point_cloud(plypath)
    points = np.asarray(pcd.points)

    inliers, plane_model = detect_plane(points, threshold=1.0)
    # サンプルデータはZ軸方向に平面の法線が存在するので座標変換は行わない
    clusters_indices = plane_clustering(points[inliers], eps=10.0)  # EPS=10.0mmくらいで適切な
    pcds = []
    for cluster in clusters_indices:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[inliers][cluster])
        pcd.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)

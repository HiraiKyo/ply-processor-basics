import numpy as np
import open3d as o3d

from ply_processor_basics.points.ransac import detect_plane


def test_success():
    pcd = o3d.io.read_point_cloud("data/samples/sample.ply")
    points = np.asarray(pcd.points)
    inliers, model = detect_plane(points, threshold=1.0, minPoints=100, maxIteration=1000)
    assert inliers is not None
    assert model is not None
    assert len(inliers) > 10000
    assert len(model) == 4
    # およそZ軸方向に平面の法線ベクトルが向いていることを確認
    normalized = model[:3] / np.linalg.norm(model[:3])
    # normalizedが[0, 0, 1]もしくは[0, 0, -1]に近いことを確認
    assert np.allclose(normalized, [0, 0, 1], atol=0.1) or np.allclose(normalized, [0, 0, -1], atol=0.1)


def test_failed_to_detect():
    pcd = o3d.io.read_point_cloud("data/samples/sample.ply")
    points = np.asarray(pcd.points)
    inliers, model = detect_plane(points, threshold=0.1, minPoints=100000, maxIteration=1000)
    assert inliers is None
    assert model is None

import datetime

import numpy as np
import open3d as o3d
import pytest

from ply_processor_basics.points.ransac import detect_line

np.random.seed(datetime.datetime.now().microsecond)
test_p = np.asarray([0, 0, 0])
test_v = np.asarray([-1, 1, 5])
test_v = test_v / np.linalg.norm(test_v)
test_line_model = (test_p, test_v)  # p+tv=0
test_length = 20
test_start_point = test_line_model[0] + test_line_model[1] * (test_length / 2)
test_end_point = test_line_model[0] + test_line_model[1] * (-test_length / 2)
diff = 0.1
# 線分上に誤差diffの範囲で点を生成
t = np.random.uniform(-test_length / 2, test_length / 2, 100)
noise = np.random.uniform(-diff, diff, (100, 3))
test_points = test_line_model[0] + np.outer(t, test_line_model[1]) + noise

# ランダムノイズを追加
noise_points = np.random.uniform(-diff * 100, diff * 100, (100, 3))
test_points_2 = np.concatenate([noise_points, test_points])


@pytest.mark.parametrize("points", [test_points, test_points_2])
@pytest.mark.visual
def test_visualize(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

    inliers, line_model = detect_line(points, 0.1)
    pcd.points = o3d.utility.Vector3dVector(points[inliers])
    o3d.visualization.draw_geometries([pcd])


@pytest.mark.parametrize("points", [test_points, test_points_2])
def test_success(points):
    inliers, line_model = detect_line(points, 0.1)
    assert np.allclose(line_model[1], test_line_model[1], atol=1e-1) or np.allclose(
        line_model[1], test_line_model[1] * -1, atol=1e-1
    )

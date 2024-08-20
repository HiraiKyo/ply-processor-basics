import numpy as np
import open3d as o3d
import pytest

from ply_processor_basics.points.convex_hull import detect_line
from ply_processor_basics.points.convex_hull.detect_line import ramer_douglas_peucker
from ply_processor_basics.points.ransac import detect_plane


@pytest.mark.visual
@pytest.mark.parametrize("plypath", ["data/samples/sample.ply"])
def test_visualize(plypath):
    pcd = o3d.io.read_point_cloud(plypath)
    points = np.asarray(pcd.points)
    inliers, plane_model = detect_plane(points, 1.0)
    inliers2, lines, line_models = detect_line(points[inliers], plane_model)

    # 凸包の線分を描画
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points[inliers])
    line_set.lines = o3d.utility.Vector2iVector(lines)
    o3d.visualization.draw_geometries([line_set])


@pytest.mark.parametrize("plypath", ["data/samples/sample.ply"])
def test_success(plypath):
    pcd = o3d.io.read_point_cloud(plypath)
    points = np.asarray(pcd.points)
    inliers, plane_model = detect_plane(points, 1.0)
    inliers2, lines, line_models = detect_line(points[inliers], plane_model)


@pytest.mark.parametrize("plypath", ["data/samples/sample.ply"])
def test_lines_expected(plypath):
    pcd = o3d.io.read_point_cloud(plypath)
    points = np.asarray(pcd.points)
    inliers, plane_model = detect_plane(points, 1.0)
    inliers2, lines, line_models = detect_line(points[inliers], plane_model)

    print("")
    for line in line_models:
        print(line)

    # 直線のどれかが以下である事を期待する
    # (0, 0, 10)を通り、(1, 0, 0)の方向を持つ

    # (0, 0, 10)を通り、(0, 1, 0)の方向を持つ
    # (0, 122, 10)を通り、(1, 0, 0)の方向を持つ


def is_line_present(lines, line):
    for l in lines:
        if np.allclose(l, line):
            return True
    return False


def test_douglas_peucker():
    # 直線状の点はすべて排除
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]])
    inliers = np.arange(len(points))
    epsilon = 0.5
    results = ramer_douglas_peucker(points, inliers, epsilon)
    assert len(results) == 2
    assert np.allclose(points[results[0]], [0, 0, 0])
    assert np.allclose(points[results[1]], [5, 0, 0])

    # 全て残る
    points = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]])
    inliers = np.arange(len(points))
    epsilon = 0.5
    results = ramer_douglas_peucker(points, inliers, epsilon)
    assert len(results) == 4
    assert np.allclose(points[results[0]], [0, 0, 0])
    assert np.allclose(points[results[-1]], [0, 2, 0])

    # 一部が排除
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 0], [4, 0, 0], [5, 0, 0]])
    inliers = np.arange(len(points))
    epsilon = 0.5
    results = ramer_douglas_peucker(points, inliers, epsilon)
    assert len(results) == 3
    assert np.allclose(points[results[0]], [0, 0, 0])
    assert np.allclose(points[results[-1]], [5, 0, 0])

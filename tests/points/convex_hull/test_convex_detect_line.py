import datetime

import numpy as np
import open3d as o3d
import pytest

from ply_processor_basics.points.convex_hull import detect_line
from ply_processor_basics.points.convex_hull.detect_plane_edge import ramer_douglas_peucker
from ply_processor_basics.points.ransac import detect_plane

sample_pcd = o3d.io.read_point_cloud("data/samples/sample.ply")
sample_points = np.asarray(sample_pcd.points)


np.random.seed(datetime.datetime.now().microsecond)
test_points = np.random.uniform(-100, 100, (50000, 2))
test_points = np.hstack((test_points, np.zeros((50000, 1))))
test_indices = np.asarray([[100, 100, 0], [-100, -100, 0], [-100, 100, 0], [100, -100, 0]])
test_vectors = np.asarray([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])


@pytest.mark.visual
@pytest.mark.parametrize("points", [test_points])
@pytest.mark.parametrize("edge_density", [1, 2, 5, 10])
def test_visualize(points, edge_density):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
    plane_inliers, plane_model = detect_plane(points, 1.0)
    lines = detect_line(points[plane_inliers], plane_model, edge_density=edge_density)

    # 凸包の線分を描画
    line_inliers = np.array([], dtype=np.intp)
    for inliers, segments, line_model in lines:
        line_inliers = np.concatenate([line_inliers, plane_inliers[inliers]])
    line_pcd = o3d.geometry.PointCloud()
    line_pcd.points = o3d.utility.Vector3dVector(points[line_inliers])
    o3d.visualization.draw_geometries([line_pcd])


@pytest.mark.parametrize("plypath", ["data/samples/sample.ply"])
def test_success(plypath):
    pcd = o3d.io.read_point_cloud(plypath)
    points = np.asarray(pcd.points)
    inliers, plane_model = detect_plane(points, 1.0)
    lines = detect_line(points[inliers], plane_model)


@pytest.mark.parametrize("points", [test_points])
@pytest.mark.parametrize("expected_indices", [test_indices])
@pytest.mark.parametrize("expected_vectors", [test_vectors])
def test_lines_expected(points, expected_indices, expected_vectors):
    plane_inliers, plane_model = detect_plane(points, 1.0)
    lines = detect_line(points[plane_inliers], plane_model, expected_edges=4, edge_density=10)
    for inliers, segments, line_model in lines:
        print(segments, line_model)

        assert (
            np.allclose(line_model[1], expected_vectors[0], atol=1e-1)
            or np.allclose(line_model[1], expected_vectors[1], atol=1e-1)
            or np.allclose(line_model[1], expected_vectors[2], atol=1e-1)
            or np.allclose(line_model[1], expected_vectors[3], atol=1e-1)
        )

        # assert (
        #     np.allclose(segments[0], expected_indices[0], atol=1e-0)
        #     or np.allclose(segments[1], expected_indices[0], atol=1e-0)
        #     or np.allclose(segments[0], expected_indices[1], atol=1e-0)
        #     or np.allclose(segments[1], expected_indices[1], atol=1e-0)
        #     or np.allclose(segments[0], expected_indices[2], atol=1e-0)
        #     or np.allclose(segments[1], expected_indices[2], atol=1e-0)
        #     or np.allclose(segments[0], expected_indices[3], atol=1e-0)
        #     or np.allclose(segments[1], expected_indices[3], atol=1e-0)
        # )


def test_douglas_peucker():
    # 直線状の点はすべて排除
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]])
    inliers = np.arange(len(points))
    epsilon = 1
    results = ramer_douglas_peucker(points, inliers, epsilon)
    assert len(results) == 2
    assert np.allclose(points[results[0]], [0, 0, 0])
    assert np.allclose(points[results[1]], [5, 0, 0])

    # 正方形は全て残る
    points = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]])
    inliers = np.arange(len(points))
    epsilon = 0.5
    results = ramer_douglas_peucker(points, inliers, epsilon)
    assert len(results) == 4

    # 一部が排除
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 0], [4, 0, 0], [5, 0, 0]])
    inliers = np.arange(len(points))
    epsilon = 0.5
    results = ramer_douglas_peucker(points, inliers, epsilon)
    assert len(results) == 3

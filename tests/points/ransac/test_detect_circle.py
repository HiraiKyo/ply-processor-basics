import numpy as np
import open3d as o3d
import pytest
from scipy import stats

from ply_processor_basics.points.ransac import detect_circle, detect_plane


@pytest.mark.parametrize("plypath", ["data/samples/sample_circle.ply"])
@pytest.mark.parametrize("expected_center", [[49.0, 52.0, 50.0]])
@pytest.mark.parametrize("expected_radius", [17.5])
@pytest.mark.visual
def test_visualize(plypath, expected_center, expected_radius):
    pcd = o3d.io.read_point_cloud(plypath)
    points = np.asarray(pcd.points)
    inliers_plane, plane_model = detect_plane(points)
    inliers, center, radius = detect_circle(points[inliers_plane], plane_model)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[inliers_plane][inliers])
    # 円を描画
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=0.01)
    direction = plane_model[:3] / np.linalg.norm(plane_model[:3])

    # 方向ベクトルに基づいて回転行列を計算
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, direction)
    rotation_angle = np.arccos(np.dot(z_axis, direction))
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)

    # 円筒を回転させ、指定された方向に合わせる
    cylinder.rotate(rotation_matrix, center=(0, 0, 0))

    # 円筒を指定された中心位置に移動
    cylinder.translate(center)

    # 座標軸の作成
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    # パラメータの表示
    print("Plane model: ", plane_model)
    print("Measured: ", center, "r: ", radius)
    print("Expected: ", expected_center, "r: ", expected_radius)
    o3d.visualization.draw_geometries([pcd, cylinder, coordinate_frame])


@pytest.mark.parametrize("plypath", ["data/samples/sample_circle.ply"])
def test_success(plypath):
    pcd = o3d.io.read_point_cloud(plypath)
    points = np.asarray(pcd.points)
    inliers_plane, plane_model = detect_plane(points)
    inliers, center, radius = detect_circle(points[inliers_plane], plane_model)
    print(center, radius)


@pytest.mark.parametrize("plypath", ["data/samples/sample_circle.ply"])
@pytest.mark.parametrize("expected_center", [[49.0, 52.0, 50.0]])
@pytest.mark.parametrize("expected_radius", [17.5])
@pytest.mark.parametrize("tolerance", [10.0, 1.0])
def test_strict(plypath, expected_center, expected_radius, tolerance):
    pcd = o3d.io.read_point_cloud(plypath)
    points = np.asarray(pcd.points)
    inliers_plane, plane_model = detect_plane(points)
    inliers, center, radius = detect_circle(points[inliers_plane], plane_model)
    assert np.allclose(center, expected_center, atol=tolerance)
    assert np.allclose(radius, expected_radius, atol=tolerance)


@pytest.mark.parametrize("plypath", ["data/samples/sample_circle.ply"])
@pytest.mark.parametrize("expected_center", [[49.0, 52.0, 50.0]])
@pytest.mark.parametrize("expected_radius", [17.5])
@pytest.mark.parametrize("num_samples", [100])
@pytest.mark.parametrize("tolerance", [0.5])
def test_variance(plypath, expected_center, expected_radius, num_samples, tolerance):
    pcd = o3d.io.read_point_cloud(plypath)
    points = np.asarray(pcd.points)
    inliers_plane, plane_model = detect_plane(points)
    centers = []
    radiuses = []
    for i in range(num_samples):
        inliers, center, radius = detect_circle(points[inliers_plane], plane_model)
        centers.append(center)
        radiuses.append(radius)

    # 95%の信頼区間を計算
    center_errors = [np.linalg.norm(np.array(expected_center) - np.array(center)) for center in centers]
    radius_errors = [abs(expected_radius - radius) for radius in radiuses]

    center_ci = stats.t.interval(
        0.95, len(center_errors) - 1, loc=np.mean(center_errors), scale=stats.sem(center_errors)
    )
    radius_ci = stats.t.interval(
        0.95, len(radius_errors) - 1, loc=np.mean(radius_errors), scale=stats.sem(radius_errors)
    )

    # アサーション
    assert center_ci[1] < 0.5, "中心座標の誤差が許容範囲を超えています"
    assert radius_ci[1] < 0.5, "半径の誤差が許容範囲を超えています"

import datetime
import math
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ply_processor_basics.points import transform_to_plane_coordinates


def detect_circle(
    points: NDArray[np.floating],
    plane_model: NDArray[np.floating],
    density_threshold: float = 0.8,
    voxel_size: float = 1.0,
    max_iteration: int = 10000,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], float]:
    """
    平面上の点群から最大円をRANSACで検出する関数

    :param points: 点群(N, 3)
    :param plane_model: 平面の方程式(4, )
    :param density_threshold: 円内に含まれる点の密度閾値
    :param voxel_size: 格子点のサイズ
    :param max_iteration: RANSACの最大繰り返し回数
    :return: 検出した円の点ポインタ(N, ), 円中心座標(3, ), 円法線(3, ), 円半径
    """
    # 方針: 平面上の点群をXY平面に射影し、RANSACで円を検出する
    # 1. 平面上の点群をXY平面に射影し、格子点にする
    # 2. 3点をランダムサンプリングし、円の方程式を求める
    # 3. 円内に含まれる点の密度を計算し、密度閾値以上である最大円を算出する
    # 4. 最大円の内部に含まれる点を返す
    assert abs(plane_model[2]) > 1e-6
    origin = np.asarray([0, 0, -plane_model[3] / plane_model[2]])
    points_rotated, inv_matrix = transform_to_plane_coordinates(points, origin, plane_model[:3])
    points_xy = points_rotated[:, :2]

    # 格子化
    grid_size = np.floor(points_xy.ptp(axis=0) / voxel_size + 1).astype(np.int32)
    grid = np.zeros(grid_size, dtype=np.int32)
    for point in points_xy:
        x, y = np.floor((point - points_xy.min(axis=0)) / voxel_size).astype(np.int32)
        grid[x, y] = 1
    grid_points = np.argwhere(grid == 1) * voxel_size + points_xy.min(axis=0)

    # 3点をランダムサンプリング
    best_radius = 0
    best_center = np.zeros(2)
    for i in range(max_iteration):
        np.random.seed(datetime.datetime.now().microsecond)
        sample_points = grid_points[np.random.choice(len(grid_points), 3, replace=False)]
        # 3点から円の方程式を求める
        p1 = sample_points[0]
        p2 = sample_points[1]
        p3 = sample_points[2]
        a = np.array([[2 * (p2[0] - p1[0]), 2 * (p2[1] - p1[1])], [2 * (p3[0] - p1[0]), 2 * (p3[1] - p1[1])]])
        b = np.array(
            [p2[0] ** 2 + p2[1] ** 2 - p1[0] ** 2 - p1[1] ** 2, p3[0] ** 2 + p3[1] ** 2 - p1[0] ** 2 - p1[1] ** 2]
        )
        try:
            center = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            continue
        radius = np.sqrt(np.sum((p1 - center) ** 2))

        # 円内に含まれる格子点の密度計算
        inliers = np.linalg.norm(grid_points - center[:2], axis=1) < radius
        density = (len(grid_points[inliers]) * voxel_size**2) / (math.pi * radius**2)
        if density > density_threshold and radius > best_radius:
            best_radius = radius
            best_center = center[:2]

    # # visualize
    # fig, ax = plt.subplots()
    # ax.scatter(grid_points[:, 0], grid_points[:, 1])
    # circle = plt.Circle(best_center, best_radius, fill=False)
    # ax.add_artist(circle)
    # inliers = np.linalg.norm(grid_points - best_center[:2], axis=1) < radius
    # density = (len(grid_points[inliers]) * voxel_size**2) / (math.pi * best_radius**2)
    # plt.title(f"density={density}")
    # plt.show()

    # 3次元座標に戻して、座標系変換の逆変換を行う
    best_center = np.hstack([best_center, np.zeros(1)])
    best_center = np.dot(inv_matrix, np.hstack([best_center, 1]).T).T

    # 中心点から半径距離にある点を抽出
    best_inliers = np.linalg.norm(points - best_center[:3], axis=1) < best_radius

    normal = plane_model[:3] / np.linalg.norm(plane_model[:3])
    return best_inliers, best_center[:3], normal, best_radius

import datetime
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .detect_plane_edge import detect_plane_edge


def detect_circle(
    points: NDArray[np.floating], plane_model: NDArray[np.floating], iterations: int = 100, tolerance: float = 1.0
) -> Tuple[NDArray[np.intp], NDArray[np.floating], NDArray[np.floating], float]:
    """
    ConvexHullを用いて円検出

    :param points: 点群(N, 3)
    :param plane_model: 平面モデル(4,)
    :param iterations: RANSACのイテレーション数
    :return: 円の中心(3,), 法線ベクトル(3,), 半径
    """

    inliers = detect_plane_edge(points, plane_model)
    centers = []
    normals = []
    radiuses = []
    # 点を3点取得して、その3点を通る円を求める
    for i in range(iterations):
        np.random.seed(datetime.datetime.now().microsecond)
        idx = np.random.choice(inliers, 3, replace=False)
        center, normal, radius = fit_circle(points[idx])
        centers.append(center)
        normals.append(normal)
        radiuses.append(radius)
    center = np.median(centers, axis=0)
    normal = np.median(normals, axis=0)
    radius = np.median(radiuses)

    inliers = np.array([i for i in range(points.shape[0]) if np.linalg.norm(points[i] - center) < radius + tolerance])
    return inliers, center, normal, radius


def fit_circle(points: NDArray[np.floating]):
    assert points.shape[0] == 3

    p1 = points[0]
    p2 = points[1]
    p3 = points[2]

    normal = np.cross(p2 - p1, p3 - p1)
    normal /= np.linalg.norm(normal)

    A = np.array([p2 - p1, p3 - p1])
    if np.linalg.matrix_rank(A) < 2:
        raise ValueError("Three points in a line.")

    # 2. 平面の法線ベクトルを計算
    normal = np.cross(p2 - p1, p3 - p1)
    normal = normal / np.linalg.norm(normal)  # 正規化

    # 3. 円の中心を求める
    # 3点の中点を計算
    mid1 = (p1 + p2) / 2
    mid2 = (p1 + p3) / 2

    # 中点を通り、p1p2とp2p3に垂直な2つのベクトルを計算
    v1 = np.cross(p2 - p1, normal)
    v2 = np.cross(p3 - p1, normal)

    # 2つの直線の交点を求める
    # mid1 + t*v1 = mid2 + s*v2 を解く
    # np.linalgが正方行列のみを扱うので、2次元にスライス FIXME: 特異点対応
    A = np.array([v1, -v2]).T
    A = A[:2, :2]
    b = mid2 - mid1
    b = b[:2]
    t, s = np.linalg.solve(A, b)
    center = mid1 + t * v1

    # 4. 半径を計算
    radius = np.linalg.norm(center - p1)
    return center, normal, radius

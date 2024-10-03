import datetime
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .clustering import plane_clustering
from .transformer import transform_to_plane_coordinates


def detect_hole_in_plane(
    points: NDArray,
    plane_model: NDArray,
    hole_radius: float,
    cluster_index: int = 1,
    density_percentile: float = 95,
    KDTree_min_points: int = 50,
) -> Tuple[NDArray, NDArray, NDArray, float]:
    """
    低密度クラスタ探索による平面内の穴検出

    :param points: 点群(N, 3)
    :param plane_model: 平面モデルax+by+cz+d=0(4,)
    :param hole_radius: 穴の半径
    Returns:
        Tuple: 穴のエッジ点インデックス(N, ), 穴の中心(3, ), 法線(3, ), 検出円半径r

    Caution:
      Z軸に平行な平面は穴検出に対応していない(カメラの視線方向と平行な平面は利用が想定されていない)
    """
    # Z軸に平行な平面は穴検出に対応していない
    assert plane_model[2] > 1e-6
    # 2次元平面投影
    plane_normal = plane_model[:3]
    plane_origin = np.asarray([0, 0, -plane_model[3] / plane_model[2]])
    coordinated_points, inv_matrix = transform_to_plane_coordinates(points, plane_origin, plane_normal)
    projected_points = coordinated_points[:, :2]

    # 低密度点を抽出
    tree = cKDTree(projected_points)
    densities = tree.query(projected_points, k=KDTree_min_points)[0][:, -1]
    low_density_inliers = np.where(densities > np.percentile(densities, density_percentile))[0]

    print(len(points), len(low_density_inliers))
    clusters = plane_clustering(points[low_density_inliers], eps=5, min_samples=5)
    print(len(clusters))
    assert len(clusters) > cluster_index, "cluster_index is out of range"
    hole_inliers = low_density_inliers[clusters[cluster_index]]

    centers = []
    # 点を3点取得して、その3点を通る円を求める
    for i in range(100):
        np.random.seed(datetime.datetime.now().microsecond)
        idx = np.random.choice(hole_inliers, 3, replace=False)
        center, normal = fit_circle(points[idx])
        centers.append(center)
    center = np.median(centers, axis=0)

    # 法線は引数の平面法線と同じ方向を返すのみにし、補正は行わない
    # FIXME: 固定半径を考慮に入れる
    return hole_inliers, center, plane_normal, hole_radius


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

    return center, normal

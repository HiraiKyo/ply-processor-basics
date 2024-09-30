from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from ply_processor_basics.points.ransac import detect_line
from ply_processor_basics.vector import normalize

from .detect_plane_edge import detect_plane_edge


def detect_edge_as_line(
    points: NDArray[np.floating],
    plane_model: NDArray[np.floating],
    threshold: float = 0.5,
    expected_edges: int = 4,
    edge_density: int = 5,
) -> List[Tuple[NDArray[np.intp], NDArray[np.floating], NDArray[np.floating]]]:
    """
    ConvexHullを用いて平面の外形直線検出

    :param points: 点群(N, 3)
    :param plane_model: 平面モデル(4,)
    :param expected_edges: 期待されるエッジ数
    :param edge_density: エッジ点の密度(詳細はREADME.md参照)
    Returns:
        List[Tuple]: 各エッジに関する以下の情報を含むタプルのリスト
            1. エッジ点のポインタ: shape (N,) の numpy 配列
            2. エッジの線分: shape (2, 3) の numpy 配列
            3. エッジの直線方程式 p+tv=0: shape (2, 3) の numpy 配列
    """
    return_list: List[Tuple[NDArray[np.intp], NDArray[np.floating], NDArray[np.floating]]] = []

    # エッジ点を抽出
    edge_inliers = np.array([], dtype=np.intp)
    for _ in range(edge_density):
        outliers = np.where(np.logical_not(np.isin(np.arange(len(points)), edge_inliers)))[0]
        inliers, _ = detect_plane_edge(points[outliers], plane_model)
        edge_inliers = np.concatenate([edge_inliers, outliers[inliers]])

    # 抽出点から直線を検出
    for i in range(expected_edges):
        tmp_inliers, line_model = detect_line(points[edge_inliers], threshold=threshold)
        inliers = edge_inliers[tmp_inliers]

        # 線分検出失敗時はこれまでの計算結果のみを返す
        if line_model is None:
            return return_list

        # ノイズ対応、直線上の線分端点を取得
        # FIXME: 点数が少なすぎてDBSCANクラスタリングが使えない
        # clusters = line_clustering(points[inliers], min_samples=1)
        # if len(clusters) == 0:
        #     continue
        # cluster = clusters[0]
        cluster = inliers

        p, v = line_model
        projected_points = np.dot(points[cluster] - p, v)  # 点pから直線上の点への射影距離
        sorted_indices = np.argsort(projected_points)
        sorted_points = points[cluster[sorted_indices]]
        start_point: NDArray[np.floating] = sorted_points[0]
        end_point: NDArray[np.floating] = sorted_points[-1]

        return_list.append((cluster, np.array([start_point, end_point]), line_model))

        # 検出した線分をエッジ点から消去
        edge_inliers = edge_inliers[np.where(np.logical_not(np.isin(edge_inliers, cluster)))[0]]

    return return_list


def point_line_distance(point, start, end):
    """
    点と直線の距離
    """
    matrix = np.array([start - point, end - point])
    if np.linalg.matrix_rank(matrix) < 2:
        return 0
    u = point - start
    v = normalize(end - start)
    vt = np.inner(u, v) * v
    norm = np.linalg.norm(u - vt)
    return norm

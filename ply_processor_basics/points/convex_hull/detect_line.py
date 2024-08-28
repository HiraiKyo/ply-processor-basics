from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from ply_processor_basics.vector import normalize

from .detect_plane_edge import detect_plane_edge


def detect_line(
    points: NDArray[np.floating], plane_model: NDArray[np.floating], epsilon: float = 1.0
) -> Tuple[NDArray[np.intp], NDArray[np.intp], List]:
    """
    ConvexHullを用いて平面上の直線検出

    :param points: 点群(N, 3)
    :param plane_model: 平面モデル(4,)
    :param epsilon: Douglas-Peuckerのepsilon
    :return: エッジ点のポインタ(N, ), エッジの線分ポインタ(N, 2), 直線の方程式p+tv(p:点, v:方向ベクトル)
    """
    inliers, lines = detect_plane_edge(points, plane_model)

    # 輪郭線を可能な限り角の少ない多角形に近似
    simplified_indices = np.array(ramer_douglas_peucker(points, inliers, epsilon))

    simplified_lines = []
    for i in range(len(simplified_indices) - 1):
        simplified_lines.append([simplified_indices[i], simplified_indices[i + 1]])

    # 直線の方程式を算出する
    line_models = []
    for line in simplified_lines:
        p1 = points[line[0]]
        p2 = points[line[1]]
        v = p2 - p1
        v = normalize(v)
        line_models.append((p1, v))
    return simplified_indices, np.array(simplified_lines), line_models


def ramer_douglas_peucker(points: NDArray[np.floating], inliers: NDArray[np.intp], epsilon: float) -> NDArray[np.intp]:
    """
    Ramer-Douglas-Peuckerアルゴリズム
    """
    points_raw = points[inliers]

    def recursive(start_index, end_index):
        dmax = 0
        index = start_index
        for i in range(start_index + 1, end_index):
            d = point_line_distance(points_raw[i], points_raw[start_index], points_raw[end_index])
            if d > dmax:
                index = i
                dmax = d

        if dmax > epsilon:
            results = recursive(start_index, index) + recursive(index, end_index)[1:]
        else:
            results = [inliers[start_index], inliers[end_index]]

        return results

    return np.array(recursive(0, len(inliers) - 1))


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

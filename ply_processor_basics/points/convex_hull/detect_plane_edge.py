from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

from ply_processor_basics.points import get_distances_to_line, transform_to_plane_coordinates


def detect_plane_edge(
    points: NDArray[np.floating], plane_model: NDArray[np.floating]
) -> Tuple[NDArray[np.intp], NDArray[np.intp]]:
    """
    ConvexHullを用いて平面上のエッジ点を抽出する

    :param points: 点群(N, 3)
    :param plane_model: 平面モデル(4,)
    :return: エッジ点のポインタ(N, ), エッジの線分(N, 2)
    """

    assert abs(plane_model[2]) > 1e-6
    origin = np.asarray([0, 0, -plane_model[3] / plane_model[2]])
    points_rotated, inv_matrix = transform_to_plane_coordinates(points, origin, plane_model[:3])
    points_xy = points_rotated[:, :2]

    # Find the convex hull of the points
    hull = ConvexHull(points_xy)

    inliers = hull.vertices
    lines = hull.simplices
    return inliers, lines


def ramer_douglas_peucker(
    points_raw: NDArray[np.floating], inliers: NDArray[np.intp], epsilon: float
) -> NDArray[np.intp]:
    """
    Ramer-Douglas-Peuckerアルゴリズム
    """
    points = points_raw[inliers]

    def recursive(start_index, end_index):
        dmax = 0
        index = start_index
        for i in range(start_index + 1, end_index):
            p = points[start_index]
            v = points[end_index] - points[start_index]
            d = get_distances_to_line(np.asarray([points[i]]), p, v)[0]
            if d > dmax:
                index = i
                dmax = d

        if dmax > epsilon:
            results = recursive(start_index, index) + recursive(index, end_index)[1:]
        else:
            results = [inliers[start_index], inliers[end_index]]

        return results

    return np.array(recursive(0, len(inliers) - 1))

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

from ply_processor_basics.points import transform_to_plane_coordinates


def detect_plane_edge(
    points: NDArray[np.floating], plane_model: NDArray[np.floating]
) -> Tuple[NDArray[np.intp], NDArray[np.intp]]:
    """
    ConvexHullを用いて平面上のエッジ点を抽出する

    :param points: 点群(N, 3)
    :param plane_model: 平面モデル(4,)
    :return: エッジ点のポインタ(N, )
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

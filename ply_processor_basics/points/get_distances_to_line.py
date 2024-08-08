import numpy as np
from numpy.typing import NDArray

from ply_processor_basics.vector import normalize


def get_distances_to_line(
    points: NDArray[np.floating],
    line_point: NDArray[np.floating],
    line_vector: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    点群と直線の距離を求める。

    :param points: 点群 (N, 3)
    :param line_point: 直線上の1点 (3,)
    :param line_vector: 直線の方向ベクトル (3,)
    :return: 点群と直線の距離 (N,)
    """
    u = points - line_point
    v = normalize(line_vector)
    vt: NDArray[np.floating] = np.inner(u, v).reshape(-1, 1).dot(v.reshape(-1, 3))
    norms: NDArray[np.floating] = np.linalg.norm(u - vt, axis=1)
    return norms

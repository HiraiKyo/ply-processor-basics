import numpy as np
from numpy.typing import NDArray


def get_distances_to_plane(
    points: NDArray[np.floating],
    plane_model: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    平面と点の距離を求める

    :param points: 点群 (N, 3)
    :param plane_model: 平面の方程式の係数 (4,)
    :return: 点と平面の距離 (N,)
    """
    a, b, c, d = plane_model

    # 平面の法線ベクトル
    normal = np.array([a, b, c])
    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError("Invalid plane model")

    distances: NDArray[np.floating] = np.abs(np.dot(points[:, :3], normal) + d) / norm
    return distances

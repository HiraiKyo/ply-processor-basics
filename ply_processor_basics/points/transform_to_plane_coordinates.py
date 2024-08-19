# ply_processor/matrix/coordinate_transform.py

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ply_processor_basics.matrix import get_rotation_from_vectors


def transform_to_plane_coordinates(
    points: NDArray[np.floating], origin: NDArray[np.floating], normal: NDArray[np.floating]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    点群を平面上の1点を原点とし、法線方向をZ軸とする座標系に変換する。

    :param points: 変換する点群 (N, 3)
    :param origin: 新しい座標系の原点 (3,)
    :param normal: 平面の法線ベクトル (3,)
    :return: 変換後の点群 (N, 3), 逆変換行列 (4, 4)
    """
    # 法線ベクトルを正規化
    normal = normal / np.linalg.norm(normal)

    # Z軸方向の単位ベクトル
    z_axis = normal

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, 3] = origin
    transformation_matrix[:3, :3] = get_rotation_from_vectors(np.array([0, 0, 1]), z_axis[:3])
    transformation_matrix_inv = np.linalg.inv(transformation_matrix)

    # アフィン変換のために4x1に変換
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = np.dot(transformation_matrix, points.T).T

    return points[:, :3], transformation_matrix_inv

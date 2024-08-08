import math

import numpy as np
from numpy.typing import NDArray


def rotate_euler(points: NDArray[np.floating], radians: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    オイラー角で点群を回転する関数

    :param points: 点群の座標(N, 3)
    :param radians: オイラー角の回転量(3,)
    :return: 回転後の点群の座標(N, 3)
    """

    theta_x = radians[0]
    theta_y = radians[1]
    theta_z = radians[2]
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta_x), -math.sin(theta_x)],
            [0, math.sin(theta_x), math.cos(theta_x)],
        ]
    )

    rot_y = np.array(
        [
            [math.cos(theta_y), 0, math.sin(theta_y)],
            [0, 1, 0],
            [-math.sin(theta_y), 0, math.cos(theta_y)],
        ]
    )

    rot_z = np.array(
        [
            [math.cos(theta_z), -math.sin(theta_z), 0],
            [math.sin(theta_z), math.cos(theta_z), 0],
            [0, 0, 1],
        ]
    )

    rot_matrix = rot_z.dot(rot_y.dot(rot_x))
    rot_pointcloud: NDArray[np.floating] = rot_matrix.dot(points.T).T
    return rot_pointcloud

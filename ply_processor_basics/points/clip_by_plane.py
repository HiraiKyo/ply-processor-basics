import numpy as np
from numpy.typing import NDArray

from .transform_to_plane_coordinates import transform_to_plane_coordinates


def clip_by_plane(
    points_raw: NDArray[np.floating], plane_eq: NDArray[np.floating], invert: bool = False
) -> NDArray[np.floating]:
    """
    Clip point cloud by a plane.

    :param pcd: Point cloud to clip. (N, 3)
    :param plane_eq: Plane equation in the form of (4,).
    :param invert: Invert the clipping. Defaults to False.

    :return: Clipped point cloud. (N, 3)
    """
    points = points_raw
    a, b, c, d = plane_eq
    mean = np.mean(points, axis=0)
    origin = np.array([mean[0], mean[1], 0])
    if c != 0:
        origin = np.array(
            [
                mean[0],
                mean[1],
                (d - mean[0] * a - mean[1] * b) / c,
            ]
        )
    points, _ = transform_to_plane_coordinates(points, origin, plane_eq[:3])

    # 点を多く含む側を返す
    points_intp = np.where(points[:, 2] >= 0.0)[0]
    points_intp_inv = np.where(points[:, 2] < 0.0)[0]
    if len(points_intp) < len(points_intp_inv):
        tmp = points_intp
        points_intp = points_intp_inv
        points_intp_inv = tmp

    if invert:
        return points_raw[points_intp_inv]
    else:
        return points_raw[points_intp]

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ply_processor_basics.vector import normalize


def get_rotation_from_vectors(vec1: NDArray[np.float32], vec2: NDArray[np.float32]):
    """_summary_

    Args:
        vec1: The vector from.
        vec2: The vector to.

    Returns:
        The rotation matrix from vec1 to vec2.
    """
    # vec1 -> vec2 の回転ベクトルを導出
    b = normalize(vec1[:3])
    a = normalize(vec2[:3])
    cross = np.cross(a, b)
    dot = np.dot(a, b)
    angle = np.arccos(dot)
    rotvec = normalize(cross) * angle

    # 回転行列を導出
    rotation_matrix = Rotation.from_rotvec(rotvec).as_matrix()

    return rotation_matrix

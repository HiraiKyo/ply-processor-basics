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
    a = normalize(vec1[:3])
    b = normalize(vec2[:3])
    if np.allclose(a, b):
        return np.eye(3)
    if np.allclose(a, -b):
        perpendicular = normalize(np.array([1, 0, 0]) if np.allclose(a, [0, 0, 1]) else np.cross([0, 0, 1], a))
        return Rotation.from_rotvec(np.pi * perpendicular).as_matrix()
    axis = normalize(np.cross(a, b))
    angle = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))

    return Rotation.from_rotvec(axis * angle).as_matrix()

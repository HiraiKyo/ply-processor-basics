import numpy as np
from pytest import approx

from ply_processor_basics.points import get_distances_to_plane


def test_get_distances_to_plane_success() -> None:
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    plane_model = np.array([0, 0, 1, 0])
    distances = get_distances_to_plane(points, plane_model)
    assert distances == approx([0, 0, 0, 1])

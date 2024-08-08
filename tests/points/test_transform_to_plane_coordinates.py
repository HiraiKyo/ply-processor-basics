import numpy as np
from pytest import approx

from ply_processor_basics.points import transform_to_plane_coordinates


def test_simple_no_translate() -> None:
    points = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    origin = np.array([0, 0, 0])
    normal = np.array([0, 1, 0])
    points, _ = transform_to_plane_coordinates(points, origin, normal)
    assert points[0] == approx([1, 0, 1])
    assert points[1] == approx([1, -1, 0])
    assert points[2] == approx([0, -1, 1])


def test_simple_with_translate() -> None:
    points = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    origin = np.array([1, 1, 1])
    normal = np.array([0, 1, 0])
    points, _ = transform_to_plane_coordinates(points, origin, normal)
    assert points[0] == approx([0, 1, 0])
    assert points[1] == approx([0, 0, -1])
    assert points[2] == approx([-1, 0, 0])

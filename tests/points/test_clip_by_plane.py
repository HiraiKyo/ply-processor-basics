import numpy as np

from ply_processor_basics.points.clip_by_plane import clip_by_plane


def test_simple():
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    plane_eq = np.array([0, 0, 1, 1])
    clipped = clip_by_plane(points, plane_eq)
    assert clipped.shape[0] == 3

    clipped = clip_by_plane(points, plane_eq, invert=True)
    assert clipped.shape[0] == 1

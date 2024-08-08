import math

import numpy as np
from pytest import approx

from ply_processor_basics.points import rotate_euler


def test_rotate_euler() -> None:
    points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    radians = np.array([math.pi / 2, math.pi / 2, math.pi / 2])
    rotated = rotate_euler(points, radians)
    assert rotated[0] == approx([0, 0, -1])
    assert rotated[1] == approx([0, 1, 0])
    assert rotated[2] == approx([1, 0, 0])

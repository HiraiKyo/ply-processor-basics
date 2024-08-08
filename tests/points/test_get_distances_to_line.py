import math

import numpy as np
from pytest import approx

from ply_processor_basics.points import get_distances_to_line


def test_simple():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64)
    line_point = np.array([0, 0, 0], dtype=np.float64)
    line_vector = np.array([1, 1, 1], dtype=np.float64)
    distances = get_distances_to_line(points, line_point, line_vector)
    assert distances == approx([0, math.sqrt(6) / 3, math.sqrt(6) / 3, math.sqrt(6) / 3, 0])

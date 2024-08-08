import numpy as np

from ply_processor_basics.matrix import get_rotation_from_vectors


def test_simple_rotation():
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])

    rotation_matrix = get_rotation_from_vectors(vec1, vec2)
    assert np.allclose(np.dot(vec1, rotation_matrix), vec2)

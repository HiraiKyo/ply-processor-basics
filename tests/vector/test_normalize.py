import numpy as np
from pytest import approx

from ply_processor_basics.vector import normalize


def test_normalize_success() -> None:
    v = np.array([1, 1, 0])
    normed = normalize(v)
    assert normed == approx([1 / np.sqrt(2), 1 / np.sqrt(2), 0])

    v = np.array([1, 1, 1])
    normed = normalize(v)
    assert normed == approx([1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)])


def test_normalize_failed() -> None:
    v = np.array([0, 0, 0])
    # expecting raise ZeroDivisionError
    try:
        normalize(v)
        assert False
    except ZeroDivisionError:
        assert True

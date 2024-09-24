import numpy as np
import pytest

from ply_processor_basics.vector import ensure_consistent_direction, estimate_vector


@pytest.mark.parametrize(
    "vector_samples",
    [
        np.asarray(
            [
                [-1.01, -1.01, -1],
                [1, 1, 1.01],
                [1.01, 1, 1],
                [1, 1.01, 1],
                [1, 1, 1.01],
                [1, 1.001, 1],
                [1.0001, 1, 1.01],
                [1, 1, -1],
                [-1, -1, -1],
                [-1, -1, -1.01],
            ]
        )
    ],
)
def test_estimate_vector(vector_samples):
    vector_samples = ensure_consistent_direction(vector_samples)
    estimated = estimate_vector(vector_samples)
    assert np.allclose(estimated, [0.577, 0.577, 0.577], 1e-2) or np.allclose(estimated, [-0.577, -0.577, -0.577], 1e-2)

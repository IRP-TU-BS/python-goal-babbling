import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pygb import LinearPathGenerator


def test_linear_path_generator() -> None:
    generator = LinearPathGenerator()
    local_goals = generator.generate(start=np.array([1.0, 1.0]), stop=np.array([10.0, 6.0]), len_sequence=5)

    assert_array_almost_equal(
        local_goals,
        [
            np.array([2.8, 2.0]),
            np.array([4.6, 3.0]),
            np.array([6.4, 4.0]),
            np.array([8.2, 5.0]),
            np.array([10.0, 6.0]),
        ],
    )


def test_linear_path_generator_raises() -> None:
    generator = LinearPathGenerator()
    with pytest.raises(RuntimeError):
        generator.generate(np.array([1.0, 0.0]), np.array([1.0, 0.0]), 42)

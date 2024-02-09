import numpy as np
import pytest

from pygb._impl._utils._maths import rmse


@pytest.mark.parametrize(
    ("truth", "predictions", "outcome"),
    [
        (np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.5, 4.0]), 0.707),
        (np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), np.array([[0.5, 1.5, 4.0], [4.0, 3.0, 5.5]]), 0.790),
    ],
)
def test_rmse(truth: np.ndarray, predictions: np.ndarray, outcome: float) -> None:
    assert pytest.approx(rmse(truth, predictions), abs=1e-3) == outcome

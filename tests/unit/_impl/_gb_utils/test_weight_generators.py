from unittest.mock import MagicMock

import numpy as np
import pytest

from pygb import GBWeightGenerator, GoalBabblingContext


@pytest.mark.parametrize(
    ("local_goal", "prev_local", "local_goal_pred", "prev_local_pred", "expected_w_dir"),
    [
        (np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 0.0]), 1.0),  # same direction
        (np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 0.0]), 0.0),  # opposite
        (np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.5),  # 90°
        (np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, -1.0]), np.array([0.0, 0.0]), 0.5),  # 90°
    ],
)
def test_gb_weight_generator_directional_weight(
    local_goal: np.ndarray,
    prev_local: np.ndarray,
    local_goal_pred: np.ndarray,
    prev_local_pred: np.ndarray,
    expected_w_dir: np.ndarray,
) -> None:
    generator = GBWeightGenerator()

    w_dir, _ = generator._calc_weights(
        local_goal=local_goal,
        prev_local=prev_local,
        local_goal_pred=local_goal_pred,
        prev_local_pred=prev_local_pred,
        action=np.array([1.0, 1.0]),
        prev_action=np.array([0.0, 1.0]),
    )

    assert w_dir == expected_w_dir


def test_gb_weight_generator_efficiency_weight() -> None:
    generator = GBWeightGenerator()

    # no movement
    _, w_eff = generator._calc_weights(
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
    )

    assert w_eff == 0

    # high efficiency vs low efficiency
    _, w_eff_high = generator._calc_weights(
        local_goal=np.array([1.0, 1.0]),
        prev_local=np.array([1.0, 1.0]),
        local_goal_pred=np.array([10.0, 5.0]),
        prev_local_pred=np.array([0.0, 0.0]),
        action=np.array([0.5, 0.0]),
        prev_action=np.array([0.0, 0.0]),
    )

    _, w_eff_low = generator._calc_weights(
        local_goal=np.array([1.0, 1.0]),
        prev_local=np.array([1.0, 1.0]),
        local_goal_pred=np.array([5.0, 1.0]),
        prev_local_pred=np.array([0.0, 0.0]),
        action=np.array([0.5, 0.0]),
        prev_action=np.array([0.0, 0.0]),
    )

    assert w_eff_low < w_eff_high


def test_gb_weight_generator_raises() -> None:
    generator = GBWeightGenerator()

    context_mock = MagicMock(spec=GoalBabblingContext)
    context_mock.runtime_data = MagicMock()
    context_mock.runtime_data.observation_index = 0

    with pytest.raises(RuntimeError):
        generator.generate(context_mock)

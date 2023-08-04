from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from pygb import (
    GBParameters,
    GBParameterStore,
    GBWeightGenerator,
    GoalBabblingContext,
    ObservationSequence,
    RuntimeData,
)

param_store = GBParameterStore(
    GBParameters(
        None, None, None, None, None, None, None, None, home_action=np.array([42.0]), home_observation=np.array([24.0])
    )
)


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


@pytest.mark.parametrize(
    (
        "observation_index",
        "previous_sequence",
        "expected_previous_local_goal",
        "expected_local_goal_pred",
        "expected_action",
    ),
    [
        (
            0,
            None,
            np.array([4.0]),  # home observation
            np.array([4.0]),  # home observation
            np.array([5.0]),  # home action
        ),
        (
            0,
            ObservationSequence(
                None,
                None,
                None,
                local_goals=[np.array([6.0])],
                predicted_actions=[np.array([7.0])],
                observations=[np.array([8.0])],
            ),
            np.array([6.0]),  # previous sequence local goal
            np.array([8.0]),  # previous sequence observation
            np.array([7.0]),  # previous sequence action
        ),
        (
            1,
            None,
            np.array([1.0]),  # current sequence's previous local goal
            np.array([3.0]),  # current sequence's previous observation
            np.array([2.0]),  # current sequence's previous predicted action
        ),
    ],
)
def test_choose_previous_data(
    observation_index: int,
    previous_sequence: ObservationSequence | None,
    expected_previous_local_goal: np.ndarray,
    expected_local_goal_pred: np.ndarray,
    expected_action: np.ndarray,
) -> None:
    current_sequence = ObservationSequence(
        None,
        None,
        None,
        local_goals=[np.array([1.0]), np.array([1.5])],
        predicted_actions=[np.array([2.0]), np.array([2.5])],
        observations=[np.array([3.0]), np.array([3.5])],
    )
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=MagicMock(
            spec=RuntimeData,
            current_sequence=current_sequence,
            previous_sequence=previous_sequence,
            observation_index=observation_index,
        ),
        current_parameters=PropertyMock(
            spec=GBParameters,
            home_observation=np.array([4.0]),
            home_action=np.array([5.0]),
        ),
    )

    generator = GBWeightGenerator()

    prev_local_boal, prev_local_pred, prev_action = generator._choose_previous_data(context_mock)

    assert prev_local_boal == expected_previous_local_goal
    assert prev_local_pred == expected_local_goal_pred
    assert prev_action == expected_action

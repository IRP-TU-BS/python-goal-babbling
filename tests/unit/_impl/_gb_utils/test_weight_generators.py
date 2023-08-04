from unittest.mock import MagicMock

import numpy as np
import pytest

from pygb import (
    GBParameters,
    GBParameterStore,
    GBWeightGenerator,
    GoalBabblingContext,
    RuntimeData,
    SequenceData,
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
    ("sequence", "previous_sequence", "observation_index", "calls"),
    [
        (  # start of sequence with no previous sequence
            SequenceData(
                0, 1, local_goals=[np.array([1.0])], observations=[np.array([2.0])], predicted_actions=[np.array([3.0])]
            ),
            None,
            0,
            {
                "local_goal": np.array([1.0]),  # from sequence
                "prev_local": np.array([24.0]),  # home observation
                "local_goal_pred": np.array([2.0]),  # from sequence
                "prev_local_pred": np.array([24.0]),  # home observation
                "action": np.array([3.0]),  # from sequence
                "prev_action": np.array([42.0]),  # home action
            },
        ),
        (  # start of sequence with previous sequence:
            SequenceData(
                0,
                1,
                local_goals=[np.array([1.0])],
                observations=[np.array([2.0])],
                predicted_actions=[
                    np.array([3.0]),
                ],
            ),
            SequenceData(
                0,
                1,
                local_goals=[np.array([10.0])],
                observations=[np.array([20.0])],
                predicted_actions=[np.array([30.0])],
            ),
            0,
            {
                "local_goal": np.array([1.0]),  # from sequence
                "prev_local": np.array([10.0]),  # previous sequence's goal
                "local_goal_pred": np.array([2.0]),  # from sequence
                "prev_local_pred": np.array([20.0]),  # previous sequence's observation
                "action": np.array([3.0]),  # from sequence
                "prev_action": np.array([30.0]),  # previous sequence's action
            },
        ),
        (  # middle of sequence with previous sequence:
            SequenceData(
                0,
                1,
                local_goals=[np.array([1.0]), np.array([1.5])],
                observations=[np.array([2.0]), np.array([2.5])],
                predicted_actions=[np.array([3.0]), np.array([3.5])],
            ),
            SequenceData(
                0,
                1,
                local_goals=[np.array([10.0])],
                observations=[np.array([20.0])],
                predicted_actions=[np.array([30.0])],
            ),
            1,
            {
                "local_goal": np.array([1.5]),  # from sequence
                "prev_local": np.array([1.0]),  # current sequence, but previous observation index
                "local_goal_pred": np.array([2.5]),  # from sequence
                "prev_local_pred": np.array([2.0]),  # current sequence, but previous observation index
                "action": np.array([3.5]),  # from sequence
                "prev_action": np.array([3.0]),  # current sequence, but previous observation index
            },
        ),
        (  # middle of sequence, but no previous completed sequence:
            SequenceData(
                0,
                1,
                local_goals=[np.array([1.0]), np.array([1.5])],
                observations=[np.array([2.0]), np.array([2.5])],
                predicted_actions=[np.array([3.0]), np.array([3.5])],
            ),
            None,
            1,
            {
                "local_goal": np.array([1.5]),  # from sequence
                "prev_local": np.array([1.0]),  # current sequence, but previous observation index
                "local_goal_pred": np.array([2.5]),  # from sequence
                "prev_local_pred": np.array([2.0]),  # current sequence, but previous observation index
                "action": np.array([3.5]),  # from sequence
                "prev_action": np.array([3.0]),  # current sequence, but previous observation index
            },
        ),
    ],
)
def test_gb_weight_generator_no_previous_sequence(
    sequence: SequenceData, previous_sequence: SequenceData | None, observation_index: int, calls: dict[str, np.ndarray]
) -> None:
    runtime_data = RuntimeData(
        previous_sequence=previous_sequence, current_sequence=sequence, observation_index=observation_index
    )

    context = GoalBabblingContext(
        param_store=param_store, goal_store=None, forward_model=None, inverse_estimate=None, runtime_data=runtime_data
    )

    generator = GBWeightGenerator()
    generator._calc_weights = MagicMock(return_value=(0.5, 0.5))

    weight = generator.generate(context)
    assert weight == 0.5 * 0.5

    generator._calc_weights.assert_called_once_with(**calls)

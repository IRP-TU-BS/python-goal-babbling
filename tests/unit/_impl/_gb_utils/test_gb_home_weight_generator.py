from unittest.mock import MagicMock

import numpy as np
import pytest

from pygb import (
    ActionSequence,
    GBHomeWeightGenerator,
    GoalBabblingContext,
    ObservationSequence,
    RuntimeData,
)


@pytest.mark.parametrize(
    ("observation_index", "current_sequence", "previous_sequence", "expected_observation", "expected_action"),
    [
        (
            0,
            ActionSequence(None, None),
            ObservationSequence(
                None,
                None,
                None,
                observations=[np.array([0.0]), np.array([1.0])],
                predicted_actions=[np.array([-1.0]), np.array([-2.0])],
            ),
            np.array([1.0]),
            np.array([-2.0]),
        ),
        (
            0,
            ActionSequence(None, None),
            ActionSequence(
                None,
                None,
                observations=[np.array([2.0]), np.array([3.0])],
                actions=[np.array([-3.0]), np.array([-4.0])],
            ),
            np.array([3.0]),
            np.array([-4.0]),
        ),
        (
            1,
            ActionSequence(
                None,
                None,
                actions=[np.array([-5.5]), np.array([-6.0])],
                observations=[np.array([5.5]), np.array([6.0])],
            ),
            None,
            np.array([5.5]),
            np.array([-5.5]),
        ),
    ],
)
def test_choose_previous_data(
    observation_index: int,
    current_sequence: ObservationSequence | ActionSequence,
    previous_sequence: ObservationSequence | ActionSequence,
    expected_observation: np.ndarray,
    expected_action: np.ndarray,
) -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=MagicMock(
            spec=RuntimeData,
            observation_index=observation_index,
            current_sequence=current_sequence,
            previous_sequence=previous_sequence,
        ),
    )

    generator = GBHomeWeightGenerator()

    observation, action = generator._choose_previous_data(context_mock)

    assert observation == expected_observation
    assert action == expected_action

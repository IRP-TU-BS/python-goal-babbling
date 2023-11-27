from unittest.mock import MagicMock, call

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pygb import (
    ActionSequence,
    GoalBabblingContext,
    ObservationSequence,
    RandomGoalSelector,
    RuntimeData,
)


@pytest.mark.parametrize(
    ("previous_sequence", "expected_index", "expected_goal"),
    [
        (
            ObservationSequence(start_goal=np.array([3.0, 4.0]), stop_goal=np.array([5.0, 6.0]), stop_goal_index=2),
            1,
            np.array([3.0, 4.0]),
        ),
        (ActionSequence(start_action=None, stop_action=None), 1, np.array([3.0, 4.0])),
        (None, 1, np.array([3.0, 4.0])),
    ],
)
def test_random_goal_selector(
    previous_sequence: ActionSequence | ObservationSequence, expected_index: int, expected_goal: np.ndarray
) -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        current_parameters=MagicMock(
            home_observation=np.array([-1.0, -1.0]),
        ),
        runtime_data=MagicMock(spec=RuntimeData, previous_sequence=previous_sequence),
        current_goal_set=MagicMock(train=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])),
    )
    selector = RandomGoalSelector()
    selector._rng = MagicMock(integers=MagicMock(return_value=1))

    index, goal = selector.select(context_mock)

    selector._rng.integers.assert_called_once_with(0, 3, size=None)
    assert index == expected_index
    assert_array_equal(goal, expected_goal)


def test_random_goal_selector_does_not_choose_previous_stop_goal() -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        current_parameters=MagicMock(
            home_observation=np.array([-1.0, -1.0]),
        ),
        runtime_data=MagicMock(
            spec=RuntimeData,
            previous_sequence=ObservationSequence(
                start_goal=np.array([3.0, 4.0]), stop_goal=np.array([5.0, 6.0]), stop_goal_index=2
            ),
        ),
        current_goal_set=MagicMock(train=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])),
    )

    selector = RandomGoalSelector()

    # create RNG which returns previously visited goal index (2), then one which has not been previously visited (0)
    selector._rng = MagicMock(integers=MagicMock(side_effect=[2, 0]))

    index, goal = selector.select(context_mock)

    selector._rng.integers.assert_has_calls([call(0, 3, size=None), call(0, 3, size=None)])
    assert index == 0
    assert_array_equal(goal, np.array([1.0, 2.0]))

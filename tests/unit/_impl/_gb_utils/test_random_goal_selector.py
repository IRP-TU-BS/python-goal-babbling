from unittest.mock import MagicMock, call

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pygb import (
    ActionSequence,
    GoalBabblingContext,
    GoalSet,
    GoalStore,
    ObservationSequence,
    RandomGoalSelector,
    RuntimeData,
)


def create_context_mock(previous_sequence: ActionSequence | ObservationSequence | None = None) -> GoalBabblingContext:
    goal_store = GoalStore(
        [GoalSet(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), np.array([[10.0], [20.0], [30.0]]))]
    )
    context = GoalBabblingContext(
        param_store=None,
        goal_store=goal_store,
        forward_model=None,
        inverse_estimate=None,
        runtime_data=RuntimeData(
            sequences=[
                ObservationSequence(
                    start_goal=np.array([1.0, 2.0]), stop_goal=np.array([3.0, 4.0]), stop_goal_index=None
                )
            ],
            previous_sequence=previous_sequence,
        ),
    )

    return context


@pytest.mark.parametrize(
    ("previous_sequence", "expected_index", "expected_goal"),
    [
        (
            ObservationSequence(start_goal=np.array([3.0, 4.0]), stop_goal=np.array([5.0, 6.0]), stop_goal_index=None),
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
    selector = RandomGoalSelector()

    rng_mock = MagicMock()
    rng_mock.integers = MagicMock(return_value=1)
    selector._rng = rng_mock
    context = create_context_mock(previous_sequence)

    index, goal = selector.select(context)

    selector._rng.integers.assert_called_once_with(0, 3, size=None)
    assert index == expected_index
    assert_array_equal(goal, expected_goal)


def test_random_goal_selector_does_not_choose_previous_stop_goal() -> None:
    selector = RandomGoalSelector()

    # create RNG which returns previously visited goal index (2), then one which has not been previously visited (0)
    rng_mock = MagicMock()
    rng_mock.integers = MagicMock()
    rng_mock.integers.side_effect = [2, 0]
    selector._rng = rng_mock
    context = create_context_mock(
        ObservationSequence(start_goal=np.array([3.0, 4.0]), stop_goal=np.array([5.0, 6.0]), stop_goal_index=2)
    )

    index, goal = selector.select(context)

    selector._rng.integers.assert_has_calls([call(0, 3, size=None), call(0, 3, size=None)])
    assert index == 0
    assert_array_equal(goal, np.array([1.0, 2.0]))

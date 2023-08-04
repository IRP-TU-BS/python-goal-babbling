from unittest.mock import MagicMock, call, patch

import numpy as np
from numpy.testing import assert_array_equal

from pygb import (
    GoalBabblingContext,
    GoalSet,
    GoalStore,
    RandomGoalSelector,
    RuntimeData,
    SequenceData,
)


def create_context_mock() -> GoalBabblingContext:
    goal_store = GoalStore(
        [GoalSet(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), np.array([[10.0], [20.0], [30.0]]))]
    )
    context = GoalBabblingContext(
        param_store=None,
        goal_store=goal_store,
        forward_model=None,
        inverse_estimate=None,
        runtime_data=RuntimeData(sequences=[SequenceData(start_glob_goal_idx=0, stop_glob_goal_idx=2)]),
    )

    return context


def test_random_goal_selector() -> None:
    selector = RandomGoalSelector()

    rng_mock = MagicMock()
    rng_mock.integers = MagicMock(return_value=1)
    selector._rng = rng_mock
    context = create_context_mock()

    index, goal = selector.select(context)

    selector._rng.integers.assert_called_once_with(0, 3, size=None)
    assert index == 1
    assert_array_equal(goal, np.array([3.0, 4.0]))


def test_random_goal_selector_does_not_choose_previous_stop_goal() -> None:
    selector = RandomGoalSelector()

    # create RNG which returns previously visited goal index (2), then one which has not been previously visited (0)
    rng_mock = MagicMock()
    rng_mock.integers = MagicMock()
    rng_mock.integers.side_effect = [2, 0]
    selector._rng = rng_mock
    context = create_context_mock()

    index, goal = selector.select(context)

    selector._rng.integers.assert_has_calls([call(0, 3, size=None), call(0, 3, size=None)])
    assert index == 0
    assert_array_equal(goal, np.array([1.0, 2.0]))

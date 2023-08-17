from unittest.mock import MagicMock, PropertyMock, call

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pygb import (
    ActionSequence,
    ErrorBasedGoalSelector,
    GoalBabblingContext,
    GoalSet,
    GoalStore,
    ObservationSequence,
    RandomGoalSelector,
    RuntimeData,
    SequenceType,
)


def create_context_mock(previous_sequence: SequenceType | None = None) -> GoalBabblingContext:
    goal_store = GoalStore(
        [GoalSet(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), np.array([[10.0], [20.0], [30.0]]))]
    )
    context = GoalBabblingContext(
        param_store=None,
        goal_store=goal_store,
        forward_model=None,
        inverse_estimate=None,
        runtime_data=RuntimeData(
            sequences=[ObservationSequence(start_goal=np.array([1.0, 2.0]), stop_goal=np.array([3.0, 4.0]))],
            previous_sequence=previous_sequence,
        ),
    )

    return context


@pytest.mark.parametrize(
    ("previous_sequence", "expected_index", "expected_goal"),
    [
        (ObservationSequence(start_goal=np.array([3.0, 4.0]), stop_goal=np.array([5.0, 6.0])), 1, np.array([3.0, 4.0])),
        (ActionSequence(start_action=None, stop_action=None), 1, np.array([3.0, 4.0])),
        (None, 1, np.array([3.0, 4.0])),
    ],
)
def test_random_goal_selector(previous_sequence: SequenceType, expected_index: int, expected_goal: np.ndarray) -> None:
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
    context = create_context_mock(ObservationSequence(start_goal=np.array([3.0, 4.0]), stop_goal=np.array([5.0, 6.0])))

    index, goal = selector.select(context)

    selector._rng.integers.assert_has_calls([call(0, 3, size=None), call(0, 3, size=None)])
    assert index == 0
    assert_array_equal(goal, np.array([1.0, 2.0]))


def test_error_based_goal_selector_select_unvisited_goal() -> None:
    selector = ErrorBasedGoalSelector(select_from_top=0.2)

    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=MagicMock(spec=RuntimeData, train_goal_visit_count=[0, 1]),
        current_goal_set=PropertyMock(train=np.array([[1.5], [2.0]])),
    )
    goal_index, selected_goal = selector._select_unvisited_goal(previous_goal=np.array([2.0]), context=context_mock)

    assert goal_index == 0
    assert selected_goal == np.array([1.5])


def test_error_based_goal_selector_select_goal_by_error() -> None:
    selector = ErrorBasedGoalSelector(select_from_top=0.5)

    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=MagicMock(
            spec=RuntimeData,
            train_goal_error=[1.0, 0.5, 2.5, 4.0],
        ),
        current_goal_set=PropertyMock(train=np.array([[10.0], [20.0], [30.0], [40.0]])),
    )

    goal_index, selected_goal = selector._select_goal_by_error(previous_goal=np.array([40.0]), context=context_mock)

    assert goal_index == 2
    assert selected_goal == np.array([30.0])


def test_error_based_goal_selector_select_with_no_previous_sequence() -> None:
    train_goals = np.array([[1.0], [2.0], [3.0]])

    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=MagicMock(spec=RuntimeData, previous_sequence=None, train_goal_visit_count=[0, 0, 0]),
        current_goal_set=PropertyMock(train=train_goals),
    )

    rng_mock = MagicMock(integers=MagicMock(return_value=2))

    selector = ErrorBasedGoalSelector(select_from_top=0.25)
    selector._rng = rng_mock

    goal_index, selected_goal = selector.select(context_mock)

    assert goal_index == 2
    assert selected_goal == np.array([3.0])


def test_error_based_goal_selector_select_with_goal_unvisited() -> None:
    train_goals = np.array([[1.0], [2.0], [3.0]])

    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=MagicMock(
            spec=RuntimeData, previous_sequence=ObservationSequence(None, None), train_goal_visit_count=[0, 1, 1]
        ),
        current_goal_set=PropertyMock(train=train_goals),
    )

    select_unvisited_goal_mock = MagicMock(return_value=(0, np.array([1.0])))

    selector = ErrorBasedGoalSelector(select_from_top=0.25)
    selector._select_unvisited_goal = select_unvisited_goal_mock

    goal_index, selected_goal = selector.select(context_mock)

    assert goal_index == 0
    assert selected_goal == np.array([1.0])


def test_error_based_goal_selector_select_with_all_goals_visited() -> None:
    train_goals = np.array([[1.0], [2.0], [3.0]])

    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=MagicMock(
            spec=RuntimeData, previous_sequence=ObservationSequence(None, None), train_goal_visit_count=[0, 1, 1]
        ),
        current_goal_set=PropertyMock(train=train_goals),
    )

    select_goal_by_error_mock = MagicMock(return_value=(0, np.array([1.0])))

    selector = ErrorBasedGoalSelector(select_from_top=0.25)
    selector._select_goal_by_error = select_goal_by_error_mock

    goal_index, selected_goal = selector.select(context_mock)

    assert goal_index == 0
    assert selected_goal == np.array([1.0])

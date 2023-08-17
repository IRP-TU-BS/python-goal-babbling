from unittest.mock import MagicMock, PropertyMock

import numpy as np

from pygb import (
    ErrorBasedGoalSelector,
    GoalBabblingContext,
    ObservationSequence,
    RuntimeData,
)


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

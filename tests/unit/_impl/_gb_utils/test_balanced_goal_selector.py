from unittest.mock import MagicMock

import numpy as np
import pytest

from pygb import (
    BalancedGoalSelector,
    GoalBabblingContext,
    GoalSet,
    ObservationSequence,
    RuntimeData,
)


def test_selector_selects_unvisited_goal() -> None:
    selector = BalancedGoalSelector()
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=MagicMock(
            spec=RuntimeData,
            train_goal_visit_count=[1, 0, 1],
            previous_sequence=ObservationSequence(np.array([1.0, 2.0]), np.array([5.0, 6.0]), 2),
        ),
        current_goal_set=MagicMock(spec=GoalSet, train=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])),
    )

    goal_index, goal = selector.select(context_mock)

    assert goal_index == 1
    assert np.all(goal == np.array([3.0, 4.0]))


def test_choose_goal_by_error() -> None:
    selector = BalancedGoalSelector(error_percentile=0.5)
    errors = [0.1, 0.1, 0.5, 0.5, 0.1, 0.5, 0.1, 0.5]
    goals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.8]).reshape((-1, 1))

    previous_home = selector._choose_goal_by_error(errors, goals, prev_observation=np.array([-1.0]))

    assert previous_home[0] in (2, 3, 5, 7)

    # highest error goal is previous goal
    assert selector._choose_goal_by_error(errors, goals, np.array([6.0]))[0] in (2, 3, 7)


def test_choose_goal_by_visit_count() -> None:
    selector = BalancedGoalSelector(count_percentile=0.5)
    counts = [1, 5, 5, 1, 5, 1, 1, 5]
    goals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape((-1, 1))

    # previous observation is home
    assert selector._choose_goal_by_visit_count(counts, goals, np.array([-1.0]))[0] in (0, 3, 5, 6)

    # previous goal has lowest visit count
    assert selector._choose_goal_by_visit_count(counts, goals, np.array([7.0]))[0] in (0, 3, 5)


def test_select_below_and_above_ratio() -> None:
    rng_mock = MagicMock(random=MagicMock(side_effect=[0.6, 0.75]))
    selector = BalancedGoalSelector(ratio=0.7, rng=rng_mock)

    selector._choose_goal_by_error = MagicMock(return_value=(1, np.array([1.0])))
    selector._choose_goal_by_visit_count = MagicMock(return_value=(2, np.array([2.0])))

    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=MagicMock(
            spec=RuntimeData,
            train_goal_visit_count=[1, 2, 1, 2],
            train_goal_error=[0.1, 0.2, 0.15],
            previous_sequence=ObservationSequence(None, None, 2),
        ),
        current_goal_set=MagicMock(train=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])),
    )

    selector.select(context_mock)

    selector._choose_goal_by_error.assert_called_once()
    selector._choose_goal_by_visit_count.assert_not_called()

    selector.select(context_mock)

    selector._choose_goal_by_visit_count.assert_called_once()


def test_select_from_unvisited_goals_visit_count_alternative() -> None:
    selector = BalancedGoalSelector(ratio=0.7, count_percentile=0.9, rng=np.random.default_rng(seed=42))

    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=MagicMock(
            spec=RuntimeData,
            train_goal_visit_count=[0, 2, 1, 2],
            train_goal_error=[np.inf, 0.2, 0.15, 0.1],
            previous_sequence=ObservationSequence(
                [7.0, 8.0], np.array([1.0, 2.0]), 2
            ),  # last visited goal index 2 is equal to least visited goal index 0
        ),
        current_goal_set=MagicMock(
            train=np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [7.0, 8.0]])
        ),  # goals 0 and 2 are equal
    )

    index, _ = selector.select(context_mock)

    assert index not in (0, 2)


def test_choose_goal_by_error_does_not_pick_previous_goal() -> None:
    rng = np.random.default_rng(seed=42)
    selector = BalancedGoalSelector(error_percentile=0.50, count_percentile=0.50, rng=rng)

    goal_errors = [1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0]
    goals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape((-1, 1))
    assert selector._choose_goal_by_error(goal_errors, goals, np.array([8.0]))[0] != 7


def test_choose_goal_by_visit_count_does_not_pick_previous_goal() -> None:
    rng = np.random.default_rng(seed=42)
    selector = BalancedGoalSelector(error_percentile=0.5, count_percentile=0.5, rng=rng)

    visit_counts = [1, 2, 1, 1, 2, 2, 2, 1]
    goals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape((-1, 1))
    assert selector._choose_goal_by_visit_count(visit_counts, goals, np.array([1.0]))[0] != 0


def test_choose_goal_by_error_raises_if_selection_too_small() -> None:
    selector = BalancedGoalSelector(error_percentile=0.5, count_percentile=0.25)
    with pytest.raises(RuntimeError):
        selector._choose_goal_by_error(goal_errors=[0.1, 0.2], goals=np.array([[1.0], [2.0]]), prev_observation=None)


def test_choose_goal_by_visit_count_raises_if_selection_too_small() -> None:
    selector = BalancedGoalSelector(error_percentile=0.5, count_percentile=0.25)
    with pytest.raises(RuntimeError):
        selector._choose_goal_by_visit_count(
            goal_visit_counts=[0, 1], goals=np.array([[1.0], [2.0]]), prev_observation=None
        )

from unittest.mock import MagicMock

import numpy as np

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
        runtime_data=MagicMock(spec=RuntimeData, train_goal_visit_count=[1, 0, 1]),
        current_goal_set=MagicMock(spec=GoalSet, train=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])),
    )

    goal_index, goal = selector.select(context_mock)

    assert goal_index == 1
    assert np.all(goal == np.array([3.0, 4.0]))


def test_choose_goal_by_error() -> None:
    selector = BalancedGoalSelector(error_percentile=0.25)
    errors = [0.2, 0.1, 0.5, 0.4, 0.05, 0.6, 0.25, 0.35]
    goals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).reshape((-1, 1))

    previous_home = selector._choose_goal_by_error(errors, goals, prev_observation=np.array([-1.0]))

    assert previous_home[0] in (2, 5)

    # highest error goal is previous goal
    assert selector._choose_goal_by_error(errors, goals, np.array([6.0]))[0] == 2
    assert np.all(selector._choose_goal_by_error(errors, goals, np.array([6.0]))[1] == np.array([3.0]))


def test_choose_goal_by_visit_count() -> None:
    selector = BalancedGoalSelector(count_percentile=0.25)
    counts = [1, 4, 5, 3, 7, 2, 1, 5]
    goals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape((-1, 1))

    # previous observation is home
    assert selector._choose_goal_by_visit_count(counts, goals, np.array([-1.0]))[0] in (0, 6)

    # previous goal has lowest visit count
    assert selector._choose_goal_by_visit_count(counts, goals, np.array([7.0]))[0] == 0
    assert np.all(selector._choose_goal_by_visit_count(counts, goals, np.array([7.0]))[1] == np.array([1.0]))


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


def test_choose_goal_by_error_does_not_pick_previous_goal() -> None:
    rng = np.random.default_rng(seed=42)
    selector = BalancedGoalSelector(error_percentile=0.25, count_percentile=0.25, rng=rng)

    goal_errors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    goals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape((-1, 1))
    assert selector._choose_goal_by_error(goal_errors, goals, np.array([8.0]))[0] != 7


def test_choose_goal_by_visit_count_does_not_pick_previous_goal() -> None:
    rng = np.random.default_rng(seed=42)
    selector = BalancedGoalSelector(error_percentile=0.25, count_percentile=0.25, rng=rng)

    visit_counts = [1, 2, 3, 4, 5, 6, 7, 8]
    goals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape((-1, 1))
    assert selector._choose_goal_by_visit_count(visit_counts, goals, np.array([1.0]))[0] != 0

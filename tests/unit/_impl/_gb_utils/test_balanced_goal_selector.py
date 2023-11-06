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

    assert selector._choose_goal_by_error(errors, prev_goal_idx=-1) in (2, 5)
    assert selector._choose_goal_by_error(errors, prev_goal_idx=5) == 2


def test_choose_goal_by_visit_count() -> None:
    selector = BalancedGoalSelector(count_percentile=0.25)
    counts = [1, 4, 5, 3, 7, 2, 1, 5]

    assert selector._choose_goal_by_visit_count(counts, prev_goal_idx=-1) in (0, 6)
    assert selector._choose_goal_by_visit_count(counts, prev_goal_idx=6) == 0


def test_select_below_and_above_ratio() -> None:
    rng_mock = MagicMock(random=MagicMock(side_effect=[0.6, 0.75]))
    selector = BalancedGoalSelector(ratio=0.7, rng=rng_mock)

    selector._choose_goal_by_error = MagicMock(return_value=1)
    selector._choose_goal_by_visit_count = MagicMock(return_value=2)

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


def test_select_goal_randomly() -> None:
    rng = np.random.default_rng(seed=42)
    selector = BalancedGoalSelector(rng=rng)

    context_mock = MagicMock(
        spec=GoalBabblingContext, runtime_data=MagicMock(spec=RuntimeData, train_goal_visit_count=[0, 0, 0, 0, 0])
    )

    target_goals = []
    for _ in range(5):
        index = selector.select(context_mock)[0]
        target_goals.append(index)
        context_mock.runtime_data.train_goal_visit_count[index] += 1

    assert target_goals != [0, 1, 2, 3, 4]


def test_choose_goal_by_error_does_not_pick_previous_goal() -> None:
    rng = np.random.default_rng(seed=42)
    selector = BalancedGoalSelector(error_percentile=0.25, count_percentile=0.25, rng=rng)

    goal_errors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert selector._choose_goal_by_error(goal_errors, 7) != 7


def test_choose_goal_by_visit_count_does_not_pick_previous_goal() -> None:
    rng = np.random.default_rng(seed=42)
    selector = BalancedGoalSelector(error_percentile=0.25, count_percentile=0.25, rng=rng)

    visit_counts = [1, 2, 3, 4, 5, 6, 7, 8]
    assert selector._choose_goal_by_visit_count(visit_counts, 0) != 0

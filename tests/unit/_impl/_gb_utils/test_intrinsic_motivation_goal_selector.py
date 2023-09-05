from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pygb import (
    ActionSequence,
    EventSystem,
    GoalBabblingContext,
    GoalSet,
    IntrinsicMotivationGoalSelector,
    ObservationSequence,
    RuntimeData,
    SequenceType,
)


def test_init_registers_event_listener() -> None:
    events = EventSystem.instance()
    selector = IntrinsicMotivationGoalSelector(2, 0.5, 0.5, event_system=events)

    assert events.event_observers["sequence-finished"] == [selector._update_data_callback]


def test_init_raises_if_window_size_is_not_even() -> None:
    with pytest.raises(ValueError):
        IntrinsicMotivationGoalSelector(window_size=3, gamma=0.5, lambda_=0.5, event_system=EventSystem.instance())


def test_update_goal_error() -> None:
    selector = IntrinsicMotivationGoalSelector(2, 0.5, 0.5, EventSystem.instance())
    selector._goal_error_matrix = np.zeros(shape=(4, 2))  # 4 goals, window_size=2
    selector._goals_e_max = np.zeros(shape=(4,))
    selector._goals_e_min = np.inf * np.ones(shape=(4,))

    selector._update_goal_error(goal_index=2, goal_error=3.0)
    assert np.all(selector._goal_error_matrix == np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 0.0]]))
    assert np.all(selector._goals_e_max == np.array([0.0, 0.0, 3.0, 0.0]))
    assert np.all(selector._goals_e_min == np.array([np.inf, np.inf, 3.0, np.inf]))

    selector._update_goal_error(goal_index=2, goal_error=3.5)
    assert np.all(selector._goal_error_matrix == np.array([[0.0, 0.0], [0.0, 0.0], [3.5, 3.0], [0.0, 0.0]]))
    assert np.all(selector._goals_e_max == np.array([0.0, 0.0, 3.5, 0.0]))
    assert np.all(selector._goals_e_min == np.array([np.inf, np.inf, 3.0, np.inf]))

    selector._update_goal_error(goal_index=2, goal_error=2.5)
    assert np.all(selector._goal_error_matrix == np.array([[0.0, 0.0], [0.0, 0.0], [2.5, 3.5], [0.0, 0.0]]))
    assert np.all(selector._goals_e_max == np.array([0.0, 0.0, 3.5, 0.0]))
    assert np.all(selector._goals_e_min == np.array([np.inf, np.inf, 2.5, np.inf]))


@pytest.mark.parametrize(
    ("current_sequence", "expect_update_call"),
    [
        (ObservationSequence(start_goal=None, stop_goal=None, stop_goal_index=2), True),
        (ActionSequence(None, None), False),
    ],
)
def test_update_data_callback(
    current_sequence: SequenceType,
    expect_update_call: bool,
) -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=MagicMock(
            spec=RuntimeData,
            epoch_set_index=0,
            current_sequence=current_sequence,
            train_goal_error=[1.0, 2.0, 3.0, 4.0],
        ),
        current_goal_set=MagicMock(train=np.ones(shape=(4, 3))),
    )

    selector = IntrinsicMotivationGoalSelector(
        window_size=2, gamma=0.5, lambda_=0.5, event_system=EventSystem.instance()
    )
    assert selector._goal_error_matrix is None
    assert selector._goals_e_max is None
    assert selector._goals_e_min is None

    selector._update_goal_error = MagicMock()

    selector._update_data_callback(context_mock)

    assert np.all(selector._goal_error_matrix == np.zeros(shape=(4, 2)))  # 4 training goals, window size=2
    assert np.all(selector._goals_e_max == np.zeros(shape=(4,)))
    assert np.all(selector._goals_e_min == np.inf * np.ones(shape=(4,)))

    if expect_update_call:
        selector._update_goal_error.assert_called_once_with(2, 3.0)
    else:
        selector._update_goal_error.assert_not_called()


def test_update_data_callback_resets_internal_data_for_new_epoch_set_index() -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        epoch_set_index=1,
        current_goal_set=MagicMock(spec=GoalSet, train=np.ones(shape=(2, 1))),
        runtime_data=MagicMock(spec=RuntimeData, current_sequence=ActionSequence(None, None), epoch_set_index=1),
    )

    selector = IntrinsicMotivationGoalSelector(
        window_size=2, gamma=0.5, lambda_=0.5, event_system=EventSystem.instance()
    )
    selector._update_goal_error = MagicMock()
    selector._valid_for_epoch_set = 0

    selector._update_data_callback(context_mock)

    assert np.all(selector._goal_error_matrix == np.zeros(shape=(2, 2)))
    assert np.all(selector._goals_e_max == np.zeros(shape=(2,)))
    assert np.all(selector._goals_e_min == np.inf * np.ones(shape=(2,)))
    assert selector._valid_for_epoch_set == 1


def test_relative_errors() -> None:
    selector = IntrinsicMotivationGoalSelector(
        window_size=2, gamma=0.5, lambda_=0.5, event_system=EventSystem.instance()
    )

    goal_error_matrix = np.array([[2.0, 10.0], [3.0, 20.0], [4.0, 30.0]])  # 3 goals, window_size 2
    expected_relative_errors = (goal_error_matrix[:, 0] - 2.0) / (4.0 - 2.0)

    assert_array_almost_equal(selector._relative_errors(goal_error_matrix), expected_relative_errors)


def test_relative_errors_for_equal_error_values() -> None:
    selector = IntrinsicMotivationGoalSelector(
        window_size=2, gamma=0.5, lambda_=0.5, event_system=EventSystem.instance()
    )

    goal_error_matrix = np.array([[2.0, 10.0], [2.0, 20.0], [2.0, 30.0]])  # 3 goals, window_size 2

    equally_distr_errors = np.array([0.333333, 0.333333, 0.333333])
    assert_array_almost_equal(selector._relative_errors(goal_error_matrix, delta=1e-9), equally_distr_errors)


def test_current_progresses() -> None:
    selector = IntrinsicMotivationGoalSelector(
        window_size=2, gamma=0.5, lambda_=0.5, event_system=EventSystem.instance()
    )

    goal_error_matrix = np.array([[2.0, 2.5], [3.5, 3.0], [1.5, 4.0]])  # 3 goals, window size 2
    expected_current_progress = (np.array([(2.0 - 2.5) / 2, (3.5 - 3.0) / 2, (1.5 - 4.0) / 2]) + 2.5 / 2) / (
        0.5 / 2 + 2.5 / 2
    )

    assert_array_almost_equal(selector._current_progresses(goal_error_matrix), expected_current_progress)


def test_current_progresses_for_equal_progress_values() -> None:
    selector = IntrinsicMotivationGoalSelector(
        window_size=2, gamma=0.5, lambda_=0.5, event_system=EventSystem.instance()
    )

    goal_error_matrix = np.array([[2.5, 2.0], [2.5, 2.0], [2.5, 2.0]])  # 3 goals, window size 2
    expected_current_progress = np.ones(shape=(3,)) / 3

    assert_array_almost_equal(selector._current_progresses(goal_error_matrix), expected_current_progress)


def test_general_progress_overviews() -> None:
    selector = IntrinsicMotivationGoalSelector(
        window_size=2, gamma=0.5, lambda_=0.5, event_system=EventSystem.instance()
    )

    goal_error_matrix = np.array([[2.5, 2.0], [3.1, 2.7], [1.0, 1.5]])  # 3 goals, window size 2
    goals_e_max = np.array([4.0, 5.0, 6.0])
    goals_e_min = np.array([2.0, 2.3, 1.0])

    expected_general_progress_overview = np.array(
        [(2.5 - 2.0) / (4.0 - 2.0), (3.1 - 2.3) / (5.0 - 2.3), (1.0 - 1.0) / (6.0 - 1.0)]
    )

    assert_array_almost_equal(
        selector._general_progress_overviews(goal_error_matrix, goals_e_min, goals_e_max),
        expected_general_progress_overview,
    )


@pytest.mark.parametrize(("previous_index", "expected_return"), [(1, 0), (0, 1)])
def test_select_goal_by_interest_isolated_relative_errors(
    previous_index: int,
    expected_return: int,
) -> None:
    selector = IntrinsicMotivationGoalSelector(
        window_size=2, gamma=0.5, lambda_=1.0, event_system=EventSystem.instance()
    )
    goal_error_matrix = np.array([[10.0, 12.0], [3.0, 5.0], [2.5, 3.0]])  # 3 train goals, window size 2
    goals_e_min = np.array([10.0, 3.0, 1.5])
    goals_e_max = np.array([20.0, 12.0, 9.0])

    assert (
        selector._select_goal_by_interest(goal_error_matrix, goals_e_min, goals_e_max, previous_index)
        == expected_return
    )


@pytest.mark.parametrize(("previous_index", "expected_return"), [(1, 0), (0, 1)])
def test_select_goal_by_interest_isolated_current_progress(
    previous_index: int,
    expected_return: int,
) -> None:
    selector = IntrinsicMotivationGoalSelector(
        window_size=2, gamma=1.0, lambda_=0.0, event_system=EventSystem.instance()
    )
    goal_error_matrix = np.array([[10.0, 5.0], [5.0, 3.0], [2.5, 3.0]])  # 3 train goals, window size 2
    goals_e_min = np.array([10.0, 3.0, 1.5])
    goals_e_max = np.array([20.0, 12.0, 9.0])

    selected_index = selector._select_goal_by_interest(goal_error_matrix, goals_e_min, goals_e_max, previous_index)
    assert selected_index == expected_return


@pytest.mark.parametrize(("previous_index", "expected_return"), [(1, 0), (0, 2)])
def test_select_goal_by_interest_isolated_general_progress_overview(
    previous_index: int,
    expected_return: int,
) -> None:
    selector = IntrinsicMotivationGoalSelector(
        window_size=2, gamma=0.0, lambda_=0.0, event_system=EventSystem.instance()
    )
    goal_error_matrix = np.array([[19.0, 3.0], [3.0, 5.0], [6.0, 3.0]])  # 3 train goals, window size 2
    goals_e_min = np.array([3.0, 3.0, 1.5])
    goals_e_max = np.array([20.0, 12.0, 9.0])

    selected_index = selector._select_goal_by_interest(goal_error_matrix, goals_e_min, goals_e_max, previous_index)

    assert selected_index == expected_return

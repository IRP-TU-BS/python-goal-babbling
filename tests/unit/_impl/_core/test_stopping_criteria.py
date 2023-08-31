from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from pygb import (
    GoalBabblingContext,
    PerformanceSlopeStop,
    RuntimeData,
    TargetPerformanceStop,
    TimeBudgetStop,
)


@pytest.mark.parametrize(
    ("performance_error", "epoch", "output"), [(42.1, 6, False), (42.0, 6, True), (41.9, 6, True), (41.0, 4, False)]
)
def test_target_performance_stop(performance_error: float, epoch: int, output: bool) -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=MagicMock(spec=RuntimeData, performance_error=performance_error, epoch_index=epoch),
    )

    criterion = TargetPerformanceStop(performance=42.0, start_epoch=5)

    assert criterion.fulfilled(context_mock) == output


@pytest.mark.parametrize(
    ("now_side_effect", "budget", "epoch", "output"),
    [
        ([datetime(2023, 8, 1, 10, 0, 0), datetime(2023, 8, 1, 10, 9, 59)], timedelta(minutes=10), 5, False),
        ([datetime(2023, 8, 1, 10, 0, 0), datetime(2023, 8, 1, 10, 10, 0)], timedelta(minutes=10), 5, True),
        ([datetime(2023, 8, 1, 10, 0, 0), datetime(2023, 8, 1, 10, 10, 0)], timedelta(minutes=10), 4, False),
    ],
)
def test_time_budget_stop(now_side_effect: list[datetime], budget: timedelta, epoch: int, output: bool) -> None:
    context_mock = MagicMock(spec=GoalBabblingContext, runtime_data=MagicMock(spec=RuntimeData, epoch_index=epoch))

    with patch("pygb._impl._core._stopping_criteria.datetime", wraps=datetime) as now_mock:
        now_mock.now.side_effect = now_side_effect

        criterion = TimeBudgetStop(budget=budget, start_epoch=5)

        # first call set _start attribute:
        criterion.fulfilled(context_mock) == False

        assert criterion.fulfilled(context_mock) == output


def test_performance_slope_stop_sets_new_best_performance() -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext, runtime_data=MagicMock(spec=RuntimeData, performance_error=24.0, epoch_index=4)
    )

    criterion = PerformanceSlopeStop(grace_period=10)
    criterion._best_performance = 42.0
    criterion._period = 8

    assert not criterion.fulfilled(context_mock)

    assert criterion._period == 0
    assert criterion._best_performance == 24.0


def test_performance_slope_returns_false_if_start_epoch_not_reached() -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext, runtime_data=MagicMock(spec=RuntimeData, performance_error=24.0, epoch_index=4)
    )

    criterion = PerformanceSlopeStop(grace_period=10, start_epoch=5)
    criterion._best_performance = 43.0
    criterion._period = 11

    assert not criterion.fulfilled(context_mock)


def test_performance_slope_stop_updates_period_and_stops_training() -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext, runtime_data=MagicMock(spec=RuntimeData, performance_error=44.0, epoch_index=4)
    )

    criterion = PerformanceSlopeStop(grace_period=10)
    criterion._best_performance = 42.0
    criterion._period = 7

    assert not criterion.fulfilled(context_mock)
    assert criterion._period == 8

    assert criterion.fulfilled(context_mock)
    assert criterion._period == 9

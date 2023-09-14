from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from pygb import (
    ActionSequence,
    EpochSetRecord,
    EpochSetStats,
    EventSystem,
    GBParameters,
    GoalBabblingContext,
    GoalSet,
    ObservationSequence,
    RuntimeData,
)
from pygb.interfaces import (
    AbstractEstimateCache,
    AbstractForwardModel,
    AbstractInverseEstimate,
    AbstractStoppingCriteria,
)
from pygb.states import EpochFinishedState


class DummyStoppingCriteria(AbstractStoppingCriteria):
    def __init__(self, return_fulfilled: bool) -> None:
        super().__init__()
        self.return_fulfilled = return_fulfilled

    def fulfilled(self, context: GoalBabblingContext) -> bool:
        return self.return_fulfilled

    def __str__(self) -> str:
        return self.__class__.__qualname__

    def __eq__(self, __o: object) -> bool:
        return super().__eq__(__o)


def get_context_mock() -> MagicMock:
    current_goal_set_mock = PropertyMock(spec=GoalSet, test=np.array([1.0]), optional_test={"foobar": np.array([2.0])})
    runtime_data_mock = MagicMock(
        spec=RuntimeData,
        performance_error=0.0,
        opt_performance_errors={"foobar": 0.0},
        epoch_index=0,
        epoch_set_index=0,
    )
    current_parameters_mock = PropertyMock(spec=GBParameters, stopping_criteria=[], len_epoch_set=2)
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=runtime_data_mock,
        current_goal_set=current_goal_set_mock,
        current_parameters=current_parameters_mock,
        estimate_cache=None,
        forward_model=None,
        inverse_estimate=None,
        epoch_set_records=[EpochSetRecord()],
    )

    return context_mock


def test_evaluate() -> None:
    state = EpochFinishedState(None)

    forward_model_mock = MagicMock(spec=AbstractForwardModel)
    forward_model_mock.forward_batch = lambda *_, **__: np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    forward_model_mock.clip_batch = MagicMock(side_effect=lambda x: x)

    inverse_estimate_mock = MagicMock(spec=AbstractInverseEstimate)
    inverse_estimate_mock.predict_batch = lambda x: x

    rmse = state._evaluate(forward_model_mock, inverse_estimate_mock, np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))
    assert rmse == 1.0

    forward_model_mock.clip_batch.assert_called_once()


@patch("pygb._impl._gb_states._epoch_finished_state.EpochFinishedState._evaluate")
@patch("pygb._impl._gb_states._epoch_finished_state.EpochFinishedState._update_record")
def test_execute_state_calculates_performance_errors(update_record_mock: MagicMock, evaluate_mock: MagicMock) -> None:
    context_mock = get_context_mock()
    evaluate_mock.side_effect = [42.0, 420.0]
    update_record_mock.side_effect = None

    state = EpochFinishedState(context_mock, EventSystem.instance())
    state()

    assert context_mock.runtime_data.performance_error == 42.0
    assert context_mock.runtime_data.opt_performance_errors == {"foobar": 420.0}


@patch("pygb._impl._gb_states._epoch_finished_state.EpochFinishedState._evaluate")
@patch("pygb._impl._gb_states._epoch_finished_state.EpochFinishedState._update_record")
def test_execute_state_returns_early_if_stopping_criteria_fulfilled(
    update_record_mock: MagicMock, evaluate_mock: MagicMock
) -> None:
    evaluate_mock.side_effect = [42.0, 420.0]
    update_record_mock.side_effect = None

    stopping_criteria_mock = MagicMock(spec=AbstractStoppingCriteria)
    stopping_criteria_mock.fulfilled = lambda *_, **__: True

    context_mock = get_context_mock()
    context_mock.current_parameters.stopping_criteria = [stopping_criteria_mock]

    state = EpochFinishedState(context_mock, EventSystem.instance())

    assert state() == EpochFinishedState.epoch_set_complete
    assert context_mock.runtime_data.epoch_index == 0  # unchanged


@patch("pygb._impl._gb_states._epoch_finished_state.EpochFinishedState._evaluate")
@patch("pygb._impl._gb_states._epoch_finished_state.EpochFinishedState._update_record")
def test_execute_state_proceeds_and_stops_epoch_set(update_record_mock: MagicMock, evaluate_mock: MagicMock) -> None:
    evaluate_mock.side_effect = [4.2, 42.0, 420.0, 4200.0]
    update_record_mock.side_effect = None
    context_mock = get_context_mock()

    state = EpochFinishedState(context_mock, EventSystem.instance())
    assert context_mock.runtime_data.epoch_index == 0

    assert state() == EpochFinishedState.epoch_set_not_complete
    assert context_mock.runtime_data.epoch_index == 1

    assert state() == EpochFinishedState.epoch_set_complete
    assert context_mock.runtime_data.epoch_index == 1


@patch("pygb._impl._gb_states._epoch_finished_state.EpochFinishedState._evaluate")
@patch("pygb._impl._gb_states._epoch_finished_state.EpochFinishedState._update_record")
def test_execute_state_resets_epoch_runtime_data(update_record_mock: MagicMock, evaluate_mock: MagicMock) -> None:
    evaluate_mock.return_value = 42.0
    update_record_mock.side_effect = None
    context_mock = get_context_mock()

    context_mock.runtime_data.current_sequence = ObservationSequence(None, None, None)
    context_mock.runtime_data.sequences = [ObservationSequence(None, None, None)]
    context_mock.runtime_data.sequence_index = 20

    state = EpochFinishedState(context_mock, event_system=EventSystem.instance())

    state()

    assert context_mock.runtime_data.current_sequence == None
    assert context_mock.runtime_data.sequences == []
    assert context_mock.runtime_data.sequence_index == 0


@patch("pygb._impl._gb_states._epoch_finished_state.EpochFinishedState._evaluate")
@patch("pygb._impl._gb_states._epoch_finished_state.EpochFinishedState._update_record")
def test_execute_state_stores_model_if_model_store_is_specified(
    update_record_mock: MagicMock, evaluate_mock: MagicMock
) -> None:
    context_mock = get_context_mock()
    context_mock.model_store = MagicMock(spec=AbstractEstimateCache, conditional_save=MagicMock())
    update_record_mock.side_effect = None
    state = EpochFinishedState(context_mock, event_system=EventSystem.instance())

    state()

    context_mock.model_store.conditional_save.assert_called_once()


@pytest.mark.parametrize(
    ("epoch_index", "len_epoch_set", "stopping_criteria", "expected_return"),
    [
        (5, 6, [], "epoch_count_reached"),
        (4, 6, [DummyStoppingCriteria(True)], "DummyStoppingCriteria"),
        (4, 6, [DummyStoppingCriteria(False)], None),
        (4, 6, [], None),
    ],
)
def test_evaluate_stop_epoch_set_complete(
    epoch_index: int, len_epoch_set: int, stopping_criteria: list[AbstractStoppingCriteria], expected_return: str | None
) -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=MagicMock(spec=RuntimeData, epoch_index=epoch_index),
        current_parameters=MagicMock(
            spec=GBParameters, len_epoch_set=len_epoch_set, stopping_criteria=stopping_criteria
        ),
    )

    state = EpochFinishedState(context_mock)

    assert state._evaluate_stop(context_mock) == expected_return


def test_update_record_without_estimate_cache() -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        epoch_set_records=[],
        runtime_data=MagicMock(
            spec=RuntimeData,
            epoch_set_index=0,
            performance_error=42.0,
            opt_performance_errors={"foo": 24.0},
            sequences=[
                ObservationSequence(None, None, None),
                ActionSequence(None, None, None),
                ObservationSequence(None, None, None),
            ],
        ),
        estimate_cache=None,
        current_parameters=MagicMock(
            spec=GBParameters,
            len_sequence=10,
        ),
    )

    state = EpochFinishedState(context_mock)

    # no estimate cache, no previous best
    state._update_record(context_mock, stop_reason=None)

    assert len(context_mock.epoch_set_records) == 1
    assert context_mock.epoch_set_records[0] == EpochSetRecord(
        total=EpochSetStats(
            performance={"test": 42.0, "foo": 24.0},
            epoch_count=1,
            observation_sequence_count=2,
            action_sequence_count=1,
            observation_sample_count=20,
            action_sample_count=10,
        ),
        best=EpochSetStats(
            performance={"test": 42.0, "foo": 24.0},
            epoch_count=1,
            observation_sequence_count=2,
            action_sequence_count=1,
            observation_sample_count=20,
            action_sample_count=10,
        ),
        stop_reason=None,
    )

    # current performance worse than previous best
    context_mock.runtime_data.performance_error = 142.0
    state._update_record(context_mock, stop_reason=None)
    assert len(context_mock.epoch_set_records) == 1
    assert context_mock.epoch_set_records[0] == EpochSetRecord(
        total=EpochSetStats(
            performance={"test": 142.0, "foo": 24.0},
            epoch_count=2,
            observation_sequence_count=4,
            action_sequence_count=2,
            observation_sample_count=40,
            action_sample_count=20,
        ),
        best=EpochSetStats(
            performance={"test": 42.0, "foo": 24.0},
            epoch_count=1,
            observation_sequence_count=2,
            action_sequence_count=1,
            observation_sample_count=20,
            action_sample_count=10,
        ),
        stop_reason=None,
    )

    # current performance better than previous best
    context_mock.runtime_data.performance_error = 4.2
    state._update_record(context_mock, stop_reason="foobar")
    assert len(context_mock.epoch_set_records) == 1
    assert context_mock.epoch_set_records[0] == EpochSetRecord(
        total=EpochSetStats(
            performance={"test": 4.2, "foo": 24.0},
            epoch_count=3,
            observation_sequence_count=6,
            action_sequence_count=3,
            observation_sample_count=60,
            action_sample_count=30,
        ),
        best=EpochSetStats(
            performance={"test": 4.2, "foo": 24.0},
            epoch_count=3,
            observation_sequence_count=6,
            action_sequence_count=3,
            observation_sample_count=60,
            action_sample_count=30,
        ),
        stop_reason="foobar",
    )


@pytest.mark.parametrize(
    ("estimate_cache", "expected_best"),
    [(MagicMock(spec=AbstractEstimateCache, conditional_save=lambda *_, **__: False), EpochSetStats())],
)
def test_update_record_with_estimate_cache(estimate_cache: MagicMock, expected_best: EpochSetStats) -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        epoch_set_records=[],
        inverse_estimate=None,
        runtime_data=MagicMock(
            spec=RuntimeData,
            epoch_set_index=0,
            performance_error=42.0,
            opt_performance_errors={"foo": 24.0},
            sequences=[
                ObservationSequence(None, None, None),
                ActionSequence(None, None, None),
                ObservationSequence(None, None, None),
            ],
        ),
        estimate_cache=estimate_cache,
        current_parameters=MagicMock(
            spec=GBParameters,
            len_sequence=10,
        ),
    )

    state = EpochFinishedState(context_mock)

    # no estimate cache, no previous best
    state._update_record(context_mock, stop_reason=None)

    assert context_mock.epoch_set_records[0].best == expected_best

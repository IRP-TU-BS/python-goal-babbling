from typing import Generator
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np

from pygb import (
    EventSystem,
    GBParameters,
    GoalBabblingContext,
    GoalSet,
    ObservationSequence,
    RuntimeData,
)
from pygb.interfaces import (
    AbstractForwardModel,
    AbstractInverseEstimator,
    AbstractStoppingCriteria,
)
from pygb.states import EpochFinishedState


def get_context_mock() -> MagicMock:
    current_goal_set_mock = PropertyMock(spec=GoalSet, test=np.array([1.0]), optional_test={"foobar": np.array([2.0])})
    runtime_data_mock = MagicMock(
        spec=RuntimeData, performance_error=0.0, opt_performance_errors={"foobar": 0.0}, epoch_index=1
    )
    current_parameters_mock = PropertyMock(spec=GBParameters, stopping_criteria=[], len_epoch_set=2)
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=runtime_data_mock,
        current_goal_set=current_goal_set_mock,
        current_parameters=current_parameters_mock,
        forward_model=None,
        inverse_estimate=None,
    )

    return context_mock


def test_evaluate() -> None:
    state = EpochFinishedState(None, None)

    forward_model_mock = MagicMock(spec=AbstractForwardModel)
    forward_model_mock.forward_batch = lambda *_, **__: np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    forward_model_mock.clip_batch = MagicMock(side_effect=lambda x: x)

    inverse_estimate_mock = MagicMock(spec=AbstractInverseEstimator)
    inverse_estimate_mock.predict_batch = lambda x: x

    rmse = state._evaluate(forward_model_mock, inverse_estimate_mock, np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))
    assert rmse == 1.0

    forward_model_mock.clip_batch.assert_called_once()


@patch("pygb._impl._gb_states._epoch_finished_state.EpochFinishedState._evaluate")
def test_execute_state_calculates_performance_errors(
    evaluate_mock: MagicMock, mock_event_system: Generator[None, None, None]
) -> None:
    context_mock = get_context_mock()
    evaluate_mock.side_effect = [42.0, 420.0]

    state = EpochFinishedState(context_mock, EventSystem.instance())
    state()

    assert context_mock.runtime_data.performance_error == 42.0
    assert context_mock.runtime_data.opt_performance_errors == {"foobar": 420.0}


@patch("pygb._impl._gb_states._epoch_finished_state.EpochFinishedState._evaluate")
def test_execute_state_returns_early_if_stopping_criteria_fulfilled(
    evaluate_mock: MagicMock, mock_event_system: Generator[None, None, None]
) -> None:
    evaluate_mock.side_effect = [42.0, 420.0]

    stopping_criteria_mock = MagicMock(spec=AbstractStoppingCriteria)
    stopping_criteria_mock.fulfilled = lambda *_, **__: True

    context_mock = get_context_mock()
    context_mock.current_parameters.stopping_criteria = [stopping_criteria_mock]

    state = EpochFinishedState(context_mock, EventSystem.instance())

    assert state() == EpochFinishedState.epoch_set_complete
    assert context_mock.runtime_data.epoch_index == 1  # unchanged


@patch("pygb._impl._gb_states._epoch_finished_state.EpochFinishedState._evaluate")
def test_execute_state_proceeds_and_stops_epoch_set(
    evaluate_mock: MagicMock, mock_event_system: Generator[None, None, None]
) -> None:
    evaluate_mock.side_effect = [4.2, 42.0, 420.0, 4200.0]
    context_mock = get_context_mock()

    state = EpochFinishedState(context_mock, EventSystem.instance())
    assert context_mock.runtime_data.epoch_index == 1

    assert state() == EpochFinishedState.epoch_set_not_complete
    assert context_mock.runtime_data.epoch_index == 2

    assert state() == EpochFinishedState.epoch_set_complete
    assert context_mock.runtime_data.epoch_index == 2


@patch("pygb._impl._gb_states._epoch_finished_state.EpochFinishedState._evaluate")
def test_execute_state_resets_epoch_runtime_data(
    evaluate_mock: MagicMock, mock_event_system: Generator[None, None, None]
) -> None:
    evaluate_mock.return_value = 42.0
    context_mock = get_context_mock()

    context_mock.runtime_data.current_sequence = ObservationSequence(None, None)
    context_mock.runtime_data.sequences = [ObservationSequence(None, None)]
    context_mock.runtime_data.sequence_index = 20

    state = EpochFinishedState(context_mock, event_system=EventSystem.instance())

    state()

    assert context_mock.runtime_data.current_sequence == None
    assert context_mock.runtime_data.sequences == []
    assert context_mock.runtime_data.sequence_index == 0

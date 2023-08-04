from typing import Generator
from unittest.mock import MagicMock, PropertyMock

from pygb import EventSystem, GBParameters, GoalBabblingContext, RuntimeData
from pygb.states import SequenceFinishedState


def test_execute_state_emits_sequence_finished_event(mock_event_system: Generator[None, None, None]) -> None:
    runtime_data_mock = PropertyMock(spec=RuntimeData, sequence_index=0)
    current_parameters_mock = PropertyMock(spec=GBParameters, len_epoch=2)
    context_mock = MagicMock(runtime_data=runtime_data_mock, current_parameters=current_parameters_mock)

    event_system = EventSystem.instance()

    callback_called = False
    attached_context = None

    def sequence_finished_callback(context: GoalBabblingContext) -> None:
        nonlocal callback_called
        nonlocal attached_context

        callback_called = True
        attached_context = context

    event_system.register_observer("sequence-finished", sequence_finished_callback)

    state = SequenceFinishedState(
        context=context_mock,
        event_system=event_system,
    )
    state()

    assert callback_called
    assert attached_context == context_mock


def test_execute_state(mock_event_system: Generator[None, None, None]) -> None:
    runtime_data_mock = PropertyMock(spec=RuntimeData, sequence_index=0)
    current_parameters_mock = PropertyMock(spec=GBParameters, len_epoch=2)
    context_mock = MagicMock(
        spec=GoalBabblingContext, runtime_data=runtime_data_mock, current_parameters=current_parameters_mock
    )

    state = SequenceFinishedState(context_mock, EventSystem.instance())

    assert state() == SequenceFinishedState.epoch_not_finished
    assert runtime_data_mock.sequence_index == 1

    assert state() == SequenceFinishedState.epoch_finished
    assert runtime_data_mock.sequence_index == 1

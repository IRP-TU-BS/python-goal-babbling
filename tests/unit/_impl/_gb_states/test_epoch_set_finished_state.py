from typing import Generator
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from pygb import EventSystem, GoalBabblingContext, GoalSet, RuntimeData, observes
from pygb.interfaces import AbstractModelStore
from pygb.states import EpochSetFinishedState


def test_state_emits_epoch_set_complete_event(mock_event_system: Generator[None, None, None]) -> None:
    event_system = EventSystem.instance()
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=MagicMock(spec=RuntimeData, epoch_set_index=1),
        num_epoch_sets=1,
    )

    called = False
    called_with_context = None

    @observes("epoch-set-complete")
    def callback(context) -> None:
        nonlocal called
        nonlocal called_with_context

        called = True
        called_with_context = context

    state = EpochSetFinishedState(context=context_mock, event_system=event_system, load_previous_best=False)
    state()

    assert called
    assert called_with_context == context_mock


def test_execute_state_continue_training(mock_event_system: Generator[None, None, None]) -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext, runtime_data=MagicMock(spec=RuntimeData, epoch_set_index=0), num_epoch_sets=2
    )

    state = EpochSetFinishedState(context_mock, event_system=EventSystem.instance(), load_previous_best=False)

    assert state() == EpochSetFinishedState.continue_training
    assert context_mock.runtime_data.epoch_set_index == 1


def test_execute_state_stop_training(mock_event_system: Generator[None, None, None]) -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext, runtime_data=MagicMock(spec=RuntimeData, epoch_set_index=1), num_epoch_sets=2
    )

    state = EpochSetFinishedState(context_mock, event_system=EventSystem.instance(), load_previous_best=False)

    assert state() == EpochSetFinishedState.stop_training
    assert context_mock.runtime_data.epoch_set_index == 1


def test_execute_state_resets_epoch_data(mock_event_system: Generator[None, None, None]) -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=MagicMock(
            spec=RuntimeData,
            epoch_set_index=1,
            epoch_index=10,
            train_goal_error=[0.5, 0.4],
            train_goal_visit_count=[2, 4],
        ),
        current_goal_set=PropertyMock(spec=GoalSet, train=np.ones((2, 3)), test=np.ones((1, 3))),
        num_epoch_sets=2,
    )

    state = EpochSetFinishedState(context_mock, event_system=EventSystem.instance(), load_previous_best=False)
    state()

    assert context_mock.runtime_data.train_goal_error == [0.0, 0.0]
    assert context_mock.runtime_data.train_goal_visit_count == [0, 0]
    assert context_mock.runtime_data.epoch_index == 0


def test_execute_state_loads_previous_best_estimate(mock_event_system: Generator[None, None, None]) -> None:
    model_store_mock = MagicMock(spec=AbstractModelStore)
    model_store_mock.load.return_value = MagicMock(id_=1)

    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=MagicMock(
            spec=RuntimeData,
            epoch_set_index=0,
        ),
        num_epoch_sets=2,
        inverse_estimate=MagicMock(id_=0),
        model_store=model_store_mock,
    )

    state = EpochSetFinishedState(context_mock, EventSystem.instance(), load_previous_best=True)

    state()

    model_store_mock.load.assert_called_once_with(0)
    assert context_mock.inverse_estimate.id_ == 1


def test_init_raises_if_load_previous_is_set_and_no_model_storage_is_specified(
    mock_event_system: Generator[None, None, None]
) -> None:
    with pytest.raises(RuntimeError):
        EpochSetFinishedState(
            context=MagicMock(spec=GoalBabblingContext, model_store=None), event_system=EventSystem.instance()
        )

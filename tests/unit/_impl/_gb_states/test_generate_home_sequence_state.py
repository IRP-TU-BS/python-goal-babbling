from typing import Generator
from unittest.mock import MagicMock, PropertyMock, call, patch

import numpy as np
import pytest

from pygb import (
    ActionSequence,
    EventSystem,
    GoalBabblingContext,
    ObservationSequence,
    RuntimeData,
    SequenceType,
)
from pygb.interfaces import (
    AbstractForwardModel,
    AbstractInverseEstimator,
    AbstractSequenceGenerator,
    AbstractWeightGenerator,
)
from pygb.states import GenerateHomeSequenceState


def test_generate_sequence_raises_if_no_previous_sequence() -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext, runtime_data=PropertyMock(spec=RuntimeData, previous_sequence=None)
    )
    state = GenerateHomeSequenceState(context_mock, None, None, None)

    with pytest.raises(NotImplementedError):
        state._generate_sequence(np.array([1.0]))


@pytest.mark.parametrize(
    ("expected_start_action", "previous_sequence"),
    [
        (
            np.array([-2.0]),
            ObservationSequence(None, None, None, None, None, predicted_actions=[np.array([-1.0]), np.array([-2.0])]),
        ),
        (
            np.array([-1.0]),
            ActionSequence(None, stop_action=np.array([-1.0])),
        ),
    ],
)
def test_generate_sequence(expected_start_action: np.ndarray, previous_sequence: SequenceType) -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext, runtime_data=MagicMock(spec=RuntimeData, previous_sequence=previous_sequence)
    )
    sequence_generator_mock = MagicMock(spec=AbstractSequenceGenerator)
    sequence_generator_mock.generate = MagicMock(return_value=[np.array([1.0]), np.array([2.0]), np.array([3.0])])

    state = GenerateHomeSequenceState(
        context_mock, home_sequence_generator=sequence_generator_mock, weight_generator=None
    )

    generated_sequence = state._generate_sequence(target_action=np.array([4.0]))

    assert generated_sequence.start_action == expected_start_action
    assert generated_sequence.stop_action == np.array([4.0])
    assert generated_sequence.actions == [np.array([1.0]), np.array([2.0]), np.array([3.0])]


@patch("pygb._impl._gb_states._generate_home_sequence_state.GenerateHomeSequenceState._generate_sequence")
def test_execute_state(generate_sequence_mock: MagicMock, mock_event_system: Generator[None, None, None]) -> None:
    dummy_sequence = ActionSequence(
        start_action=np.array([0.0]),
        stop_action=np.array([3.0]),
        actions=[np.array([1.0]), np.array([2.0]), np.array([3.0])],
    )
    generate_sequence_mock.return_value = dummy_sequence

    forward_model_mock = MagicMock(spec=AbstractForwardModel)
    forward_model_mock.forward = MagicMock(side_effect=[np.array([-1.0]), np.array([-2.0]), np.array([-3.0])])
    forward_model_mock.clip = MagicMock()

    inverse_estimator_mock = MagicMock(spec=AbstractInverseEstimator)
    inverse_estimator_mock.fit = MagicMock()

    weight_generator_mock = MagicMock(spec=AbstractWeightGenerator)
    weight_generator_mock.generate = MagicMock(side_effect=[0.1, 0.2, 0.3])

    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=RuntimeData(),
        forward_model=forward_model_mock,
        inverse_estimate=inverse_estimator_mock,
        observation_index=0,
    )

    state = GenerateHomeSequenceState(
        context_mock,
        home_sequence_generator=generate_sequence_mock,
        weight_generator=weight_generator_mock,
        event_system=EventSystem.instance(),
    )

    assert state() == GenerateHomeSequenceState.sequence_finished

    assert dummy_sequence.weights == [0.1, 0.2, 0.3]
    assert dummy_sequence.observations == [np.array([-1.0]), np.array([-2.0]), np.array([-3.0])]

    forward_model_mock.forward.assert_has_calls(
        [
            call(np.array([1.0])),
            call(np.array([2.0])),
            call(
                np.array([3.0]),
            ),
        ]
    )

    inverse_estimator_mock.fit.assert_has_calls(
        [
            call(
                np.array([-1.0]),
                np.array([1.0]),
                0.1,
            ),
            call(
                np.array([-2.0]),
                np.array([2.0]),
                0.2,
            ),
            call(
                np.array([-3.0]),
                np.array([3.0]),
                0.3,
            ),
        ]
    )

    forward_model_mock.clip.assert_called_once()

from typing import Generator
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from pygb import (
    ActionSequence,
    EventSystem,
    GBParameters,
    GBParameterStore,
    GoalBabblingContext,
    GoalSet,
    GoalStore,
    ObservationSequence,
    RuntimeData,
    SequenceType,
)
from pygb.interfaces import (
    AbstractForwardModel,
    AbstractGoalSelector,
    AbstractInverseEstimator,
    AbstractNoiseGenerator,
    AbstractSequenceGenerator,
    AbstractWeightGenerator,
)
from pygb.states import GenerateSequenceState


def generate_dummy_context(previous_sequence: SequenceType | None = None) -> GoalBabblingContext:
    gb_params = GBParameters(
        sigma=0.1,
        sigma_delta=0.01,
        dim_act=2,
        dim_obs=1,
        len_sequence=3,
        len_epoch=None,
        len_epoch_set=10,
        go_home_chance=0.1,
        home_action=np.array([0.0, 0.0]),
        home_observation=np.array([0.0]),
    )
    goal_store = GoalStore(GoalSet(train=np.array([[12.0], [13.0]]), test=None))

    if previous_sequence is not None:
        runtime_data = RuntimeData(previous_sequence=previous_sequence)
    else:
        runtime_data = RuntimeData()

    context = GoalBabblingContext(
        param_store=GBParameterStore(gb_params),
        goal_store=goal_store,
        forward_model=None,
        inverse_estimate=None,
        runtime_data=runtime_data,
    )

    return context


def test_generate_new_sequence_with_no_previous_sequence() -> None:
    goal_selector_mock = MagicMock(spec=AbstractGoalSelector)
    goal_selector_mock.select = MagicMock(return_value=(42, np.array([1.0])))

    dummy_sequence = [np.array([0.35]), np.array([0.75]), np.array([1.0])]

    local_goal_generator_mock = MagicMock(spec=AbstractSequenceGenerator)
    local_goal_generator_mock.generate = MagicMock(return_value=dummy_sequence)

    context = generate_dummy_context(previous_sequence=None)
    state = GenerateSequenceState(
        context=context,
        goal_selector=goal_selector_mock,
        local_goal_generator=local_goal_generator_mock,
        weight_generator=None,
        noise_generator=None,
    )

    sequence = state._generate_new_sequence(np.array([1.0]), context)

    assert sequence.start_goal == np.array([0.0])  # home observation
    assert sequence.stop_goal == np.array([1.0])  # target goal
    assert sequence.observations == []
    assert sequence.predicted_actions == []
    assert sequence.local_goals == dummy_sequence


@pytest.mark.parametrize(
    ("sequence", "start_goal"),
    [
        (
            ObservationSequence(start_goal=np.array([100.0]), stop_goal=np.array([200.0])),
            np.array([200.0]),
        ),
        (
            ActionSequence(None, None, actions=[np.array([1.0, 2.0])], observations=[np.array([300.0])]),
            np.array([300.0]),
        ),
    ],
)
def test_generate_new_sequence_with_previous_sequence(sequence: SequenceType, start_goal: np.ndarray) -> None:
    goal_selector_mock = MagicMock(spec=AbstractGoalSelector)
    goal_selector_mock.select = MagicMock(return_value=(42, np.array([1.0])))

    dummy_sequence = [np.array([0.35]), np.array([0.75]), np.array([1.0])]

    local_goal_generator_mock = MagicMock(spec=AbstractSequenceGenerator)
    local_goal_generator_mock.generate = MagicMock(return_value=dummy_sequence)

    context = generate_dummy_context(previous_sequence=sequence)

    state = GenerateSequenceState(
        context=context,
        goal_selector=goal_selector_mock,
        local_goal_generator=local_goal_generator_mock,
        weight_generator=None,
        noise_generator=None,
    )

    sequence = state._generate_new_sequence(np.array([1.0]), context)

    assert sequence.start_goal == start_goal  # previous sequence stop goal
    assert sequence.stop_goal == np.array([1.0])  # mocked goal selector output
    assert sequence.observations == []
    assert sequence.predicted_actions == []
    assert sequence.local_goals == dummy_sequence

    # start goal from goal store (goal index 1)
    # stop goal from mocked goal selector
    local_goal_generator_mock.generate.assert_called_once_with(start=start_goal, stop=np.array([1.0]), len_sequence=3)


@patch("pygb._impl._gb_states._generate_sequence_state.GenerateSequenceState._generate_new_sequence")
def test_execute_state(generate_sequence_mock: MagicMock, mock_event_system: Generator[None, None, None]) -> None:
    sequence = ObservationSequence(
        np.array([0.1]), np.array([1.0]), weights=[], local_goals=[np.array([0.0]), np.array([0.5]), np.array([1.0])]
    )
    generate_sequence_mock.return_value = sequence

    inverse_estimator_mock = MagicMock(spec=AbstractInverseEstimator)
    inverse_estimator_mock.predict = MagicMock(side_effect=[np.array([-1.0]), np.array([-1.5]), np.array([-2.0])])
    inverse_estimator_mock.fit = MagicMock()

    forward_model_mock = MagicMock(spec=AbstractForwardModel)
    forward_model_mock.clip = MagicMock(side_effect=lambda x: x)
    forward_model_mock.forward = MagicMock(side_effect=[np.array([100.0]), np.array([150.0]), np.array([200.0])])

    weight_generator_mock = MagicMock(spec=AbstractWeightGenerator)
    weight_generator_mock.generate = MagicMock(side_effect=[42.0, 41.0, 40.0])

    noise_generator_mock = MagicMock(spec=AbstractNoiseGenerator)
    noise_generator_mock.generate = MagicMock(side_effect=[0.1, 0.2, 0.3])

    goal_selector_mock = MagicMock(spec=AbstractGoalSelector)
    # unimportant, as _generate_new_sequence is mocked anyways
    goal_selector_mock.select = MagicMock(return_value=(1, np.array([33.0])))

    runtime_data = RuntimeData(train_goal_visit_count=[0, 0])

    context = GoalBabblingContext(
        param_store=None,
        goal_store=None,
        forward_model=forward_model_mock,
        inverse_estimate=inverse_estimator_mock,
        runtime_data=runtime_data,
    )

    state = GenerateSequenceState(
        context,
        goal_selector=goal_selector_mock,
        local_goal_generator=None,
        weight_generator=weight_generator_mock,
        event_system=EventSystem.instance(),
        noise_generator=noise_generator_mock,
    )

    transition_name = state()

    assert transition_name == GenerateSequenceState.sequence_finished
    assert context.runtime_data.current_sequence == sequence
    assert context.runtime_data.sequences == [sequence]

    assert sequence.local_goals == [np.array([0.0]), np.array([0.5]), np.array([1.0])]
    assert sequence.observations == [
        np.array([100.0]),
        np.array([150.0]),
        np.array([200.0]),
    ]  # forward model mock output
    assert sequence.predicted_actions == [
        np.array([-0.9]),
        np.array([-1.3]),
        np.array([-1.7]),
    ]  # inverse estimator mock output with added noise
    assert sequence.weights == [42.0, 41.0, 40.0]  # weight generator mock output
    assert context.runtime_data.train_goal_visit_count == [0, 1]  # sequence's stop goal count increases

    forward_model_mock.clip.assert_has_calls([call(np.array([-0.9])), call(np.array([-1.3])), call(np.array([-1.7]))])
    forward_model_mock.forward.assert_has_calls(
        [call(np.array([-0.9])), call(np.array([-1.3])), call(np.array([-1.7]))]
    )

    inverse_estimator_mock.fit.assert_has_calls(
        [
            call(np.array([100.0]), np.array([-0.9]), 42.0),
            call(np.array([150.0]), np.array([-1.3]), 41.0),
            call(np.array([200.0]), np.array([-1.7]), 40.0),
        ]
    )

    noise_generator_mock.generate.assert_has_calls(
        [call(np.array([0.0])), call(np.array([0.5])), call(np.array([1.0]))]  # local goals from sequence
    )

    goal_selector_mock.select.assert_called_once_with(context)

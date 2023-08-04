from unittest.mock import MagicMock, call, patch

import numpy as np

from pygb import (
    AbstractForwardModel,
    AbstractGoalSelector,
    AbstractInverseEstimator,
    AbstractLocalGoalGenerator,
    AbstractWeightGenerator,
    GBParameters,
    GBParameterStore,
    GenerateSequenceState,
    GoalBabblingContext,
    GoalSet,
    GoalStore,
    RuntimeData,
    SequenceData,
)


def generate_dummy_context(previous_sequence: bool = True) -> GoalBabblingContext:
    gb_params = GBParameters(
        sigma=0.1,
        sigma_delta=0.01,
        dim_act=2,
        dim_obs=1,
        len_sequence=3,
        len_epoch=None,
        epoch_sets=1,
        home_action=np.array([0.0, 0.0]),
        home_observation=np.array([0.0]),
    )
    goal_store = GoalStore(GoalSet(train=np.array([[12.0], [13.0]]), test=None))

    if previous_sequence:
        runtime_data = RuntimeData(previous_sequence=SequenceData(start_glob_goal_idx=0, stop_glob_goal_idx=1))
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

    local_goal_generator_mock = MagicMock(spec=AbstractLocalGoalGenerator)
    local_goal_generator_mock.generate = MagicMock(return_value=dummy_sequence)

    context = generate_dummy_context(previous_sequence=False)
    state = GenerateSequenceState(
        context=context,
        goal_selector=goal_selector_mock,
        local_goal_generator=local_goal_generator_mock,
        weight_generator=None,
    )

    sequence = state._generate_new_sequence(context)

    assert sequence.start_glob_goal_idx == -1  # home observation
    assert sequence.stop_glob_goal_idx == 42  # mocked goal selector output
    assert sequence.observations == []
    assert sequence.predicted_actions == []
    assert sequence.local_goals == dummy_sequence


def test_generate_new_sequence_with_previous_sequence() -> None:
    goal_selector_mock = MagicMock(spec=AbstractGoalSelector)
    goal_selector_mock.select = MagicMock(return_value=(42, np.array([1.0])))

    dummy_sequence = [np.array([0.35]), np.array([0.75]), np.array([1.0])]

    local_goal_generator_mock = MagicMock(spec=AbstractLocalGoalGenerator)
    local_goal_generator_mock.generate = MagicMock(return_value=dummy_sequence)

    context = generate_dummy_context(previous_sequence=True)
    state = GenerateSequenceState(
        context=context,
        goal_selector=goal_selector_mock,
        local_goal_generator=local_goal_generator_mock,
        weight_generator=None,
    )

    sequence = state._generate_new_sequence(context)

    assert sequence.start_glob_goal_idx == 1  # previous sequence
    assert sequence.stop_glob_goal_idx == 42  # mocked goal selector output
    assert sequence.observations == []
    assert sequence.predicted_actions == []
    assert sequence.local_goals == dummy_sequence

    # start goal from goal store (goal index 1)
    # stop goal from mocked goal selector
    local_goal_generator_mock.generate.assert_called_once_with(
        start_goal=np.array([13.0]), stop_goal=np.array([1.0]), len_sequence=3
    )


@patch("pygb._impl._gb_states._generate_sequence_state.GenerateSequenceState._generate_new_sequence")
def test_execute_state(generate_sequence_mock: MagicMock) -> None:
    sequence = SequenceData(0, 1, weights=[], local_goals=[np.array([0.0]), np.array([0.5]), np.array([1.0])])
    generate_sequence_mock.return_value = sequence

    inverse_estimator_mock = MagicMock(spec=AbstractInverseEstimator)
    inverse_estimator_mock.predict = MagicMock(side_effect=[np.array([-1.0]), np.array([-1.5]), np.array([-2.0])])
    inverse_estimator_mock.fit = MagicMock()

    forward_model_mock = MagicMock(spec=AbstractForwardModel)
    forward_model_mock.clip = MagicMock(side_effect=lambda x: x)
    forward_model_mock.forward = MagicMock(side_effect=[np.array([100.0]), np.array([150.0]), np.array([200.0])])

    weight_generator_mock = MagicMock(spec=AbstractWeightGenerator)
    weight_generator_mock.generate = MagicMock(side_effect=[42.0, 41.0, 40.0])

    runtime_data = RuntimeData()

    context = GoalBabblingContext(
        param_store=None,
        goal_store=None,
        forward_model=forward_model_mock,
        inverse_estimate=inverse_estimator_mock,
        runtime_data=runtime_data,
    )

    state = GenerateSequenceState(
        context,
        goal_selector=None,
        local_goal_generator=None,
        weight_generator=weight_generator_mock,
        event_system=None,
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
        np.array([-1.0]),
        np.array([-1.5]),
        np.array([-2.0]),
    ]  # inverse estimator mock output
    assert sequence.weights == [42.0, 41.0, 40.0]  # weight generator mock output

    forward_model_mock.clip.assert_has_calls([call(np.array([-1.0])), call(np.array([-1.5])), call(np.array([-2.0]))])
    forward_model_mock.forward.assert_has_calls(
        [call(np.array([-1.0])), call(np.array([-1.5])), call(np.array([-2.0]))]
    )

    inverse_estimator_mock.fit.assert_has_calls(
        [
            call(np.array([100.0]), np.array([-1.0]), 42.0),
            call(np.array([150.0]), np.array([-1.5]), 41.0),
            call(np.array([200.0]), np.array([-2.0]), 40.0),
        ]
    )

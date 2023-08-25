from pygb._impl._core._abstract_model_store import AbstractModelStore
from pygb._impl._core._abstract_utils import AbstractGoalSelector
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._events import EventSystem
from pygb._impl._core._goals import GoalSet, GoalStore
from pygb._impl._core._model import AbstractForwardModel, AbstractInverseEstimator
from pygb._impl._core._parameters import (
    GBParameterIncrement,
    GBParameters,
    GBParameterStore,
)
from pygb._impl._core._state_machine import StateMachine
from pygb._impl._gb_states._epoch_finished_state import EpochFinishedState
from pygb._impl._gb_states._epoch_set_finished_state import EpochSetFinishedState
from pygb._impl._gb_states._generate_home_sequence_state import (
    GenerateHomeSequenceState,
)
from pygb._impl._gb_states._generate_sequence_state import GenerateSequenceState
from pygb._impl._gb_states._go_home_decision_state import GoHomeDecisionState
from pygb._impl._gb_states._sequence_finished_state import SequenceFinishedState
from pygb._impl._gb_states._setup_state import SetupState
from pygb._impl._gb_states._stopped_state import StoppedState
from pygb._impl._gb_utils._gb_home_weight_generator import GBHomeWeightGenerator
from pygb._impl._gb_utils._gb_weight_generator import GBWeightGenerator
from pygb._impl._gb_utils._local_goal_generators import GBPathGenerator
from pygb._impl._gb_utils._noise_generators import GBNoiseGenerator


def vanilla_goal_babbling(
    parameters: GBParameters | list[GBParameters | GBParameterIncrement],
    goal_sets: GoalSet | list[GoalSet],
    forward_model: AbstractForwardModel,
    inverse_estimate: AbstractInverseEstimator,
    model_store: AbstractModelStore,
    goal_selector: AbstractGoalSelector,
    load_previous_best: bool = True,
) -> StateMachine:
    """Factory for creating a pre-configured Goal Babbling state machine.

    Args:
        parameters: Goal Babbling parameters. Either a single instance or a list of parameter sets or parameter
            increments. Length must match the amount of goal sets.
        goal_sets: Training and test goal sets. Either a single instance or a list of goal sets. Length determines the
            amount of epoch sets and must match the amount of Goal Babbling parameters.
        forward_model: Forward model instance. Implements f(a) = o (a -> actions, o -> observations).
        inverse_estimate: Inverse estimate. Implements g*(o) = a*.
        model_store: Model store. Used for internally storing the 'best' (best is defined by your concrete
            implementation) inverse estimate. Also used to load previous estimates in case of multiple-epoch-set
                training.
        goal_selector: Goal selector instance.
        load_previous_best: Whether or not to load previous best inverse estimates in case of multiple-epoch-set
            training. Defaults to True.

    Returns:
        Pre-configured state machine.
    """
    context = GoalBabblingContext(
        param_store=GBParameterStore(parameters),
        goal_store=GoalStore(goal_sets),
        forward_model=forward_model,
        inverse_estimate=inverse_estimate,
        model_store=model_store,
    )

    setup_state = SetupState(context)
    generate_sequence_state = GenerateSequenceState(
        context,
        goal_selector=goal_selector,
        goal_sequence_generator=GBPathGenerator(),
        noise_generator=GBNoiseGenerator(context),
        weight_generator=GBWeightGenerator(norm=2),
        event_system=EventSystem.instance(),
    )
    go_home_decision_state = GoHomeDecisionState(context, event_system=EventSystem.instance())
    generate_home_sequence_state = GenerateHomeSequenceState(
        context,
        home_sequence_generator=GBPathGenerator(),
        weight_generator=GBHomeWeightGenerator(norm=2),
        event_system=EventSystem.instance(),
    )
    sequence_finished_state = SequenceFinishedState(context, event_system=EventSystem.instance())
    epoch_finished_state = EpochFinishedState(context, event_system=EventSystem.instance())
    epoch_set_finished_state = EpochSetFinishedState(
        context, EventSystem.instance(), load_previous_best=load_previous_best
    )
    stopped_state = StoppedState(context, event_system=None)

    state_machine = StateMachine(context=context, initial_state=setup_state)

    state_machine.add(SetupState.setup_complete, generate_sequence_state)

    state_machine.add(GenerateSequenceState.sequence_finished, sequence_finished_state)

    state_machine.add(SequenceFinishedState.epoch_finished, epoch_finished_state)
    state_machine.add(SequenceFinishedState.epoch_not_finished, go_home_decision_state)

    state_machine.add(GoHomeDecisionState.go_home, generate_home_sequence_state)
    state_machine.add(GoHomeDecisionState.generate_sequence, generate_sequence_state)

    state_machine.add(EpochFinishedState.epoch_set_complete, epoch_set_finished_state)
    state_machine.add(EpochFinishedState.epoch_set_not_complete, go_home_decision_state)

    state_machine.add(EpochSetFinishedState.continue_training, go_home_decision_state)
    state_machine.add(EpochSetFinishedState.stop_training, stopped_state)

    return state_machine

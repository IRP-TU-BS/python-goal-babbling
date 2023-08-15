import logging

import numpy as np
from goals import X_TEST_MM, X_TRAIN_MM
from utils import ForwardModel, InverseEstimator

from pygb import (
    EventSystem,
    GBHomeWeightGenerator,
    GBNoiseGenerator,
    GBParameters,
    GBParameterStore,
    GBPathGenerator,
    GBWeightGenerator,
    GoalBabblingContext,
    GoalSet,
    GoalStore,
    RandomGoalSelector,
    StateMachine,
    observes,
    setup_logging,
)
from pygb.states import (
    EpochFinishedState,
    GenerateHomeSequenceState,
    GenerateSequenceState,
    GoHomeDecisionState,
    SequenceFinishedState,
    SetupState,
    StoppedState,
)

setup_logging(log_level=logging.INFO)

_logger = logging.getLogger(__name__)
_logger.debug("foobar")

X_TRAIN_M = X_TRAIN_MM / 1000
X_TEST_M = X_TEST_MM / 1000

X_TEST_M = X_TEST_M[:20]

JOINT_LIMITS = np.array(
    [
        [np.deg2rad(-170), np.deg2rad(-170), np.deg2rad(-170), 5.0 / 1000, 10.0 / 1000.0, 15.0 / 1000.0],
        [np.deg2rad(170), np.deg2rad(170), np.deg2rad(170), 110.0 / 1000.0, 165.0 / 1000.0, 210.0 / 1000.0],
    ]
)
HOME_JOINTS = np.array([0.0, 0.0, 0.0, 55.0 / 1000.0, 82.5 / 1000.0, 105.0 / 1000.0])

forward_model = ForwardModel(joint_limits=JOINT_LIMITS)
inverse_estimator = InverseEstimator(
    observation0=forward_model.forward(HOME_JOINTS), action0=HOME_JOINTS, radius=0.00963843, learning_rate=0.8055518
)


gb_parameters = GBParameters(
    sigma=0.0096249,
    sigma_delta=0.1,
    dim_act=6,
    dim_obs=3,
    len_sequence=15,
    len_epoch=15,
    len_epoch_set=40,
    go_home_chance=0.1,
    home_action=HOME_JOINTS,
    home_observation=forward_model.forward(HOME_JOINTS),
)

goal_set = GoalSet(train=X_TRAIN_M, test=X_TEST_M)

gb_context = GoalBabblingContext(
    param_store=GBParameterStore(gb_parameters),  # or list of parameters per epoch set
    goal_store=GoalStore(goal_set),  # or list of goal sets per epoch set
    forward_model=forward_model,
    inverse_estimate=inverse_estimator,
)

setup_state = SetupState(gb_context)
generate_sequence_state = GenerateSequenceState(
    gb_context,
    goal_selector=RandomGoalSelector(),
    goal_sequence_generator=GBPathGenerator(),
    noise_generator=GBNoiseGenerator(gb_context),
    weight_generator=GBWeightGenerator(norm=1),
    event_system=EventSystem.instance(),
)
go_home_decision_state = GoHomeDecisionState(gb_context, event_system=EventSystem.instance())
generate_home_sequence_state = GenerateHomeSequenceState(
    gb_context,
    home_sequence_generator=GBPathGenerator(),
    weight_generator=GBHomeWeightGenerator(norm=1),
    event_system=EventSystem.instance(),
)
sequence_finished_state = SequenceFinishedState(gb_context, event_system=EventSystem.instance())
epoch_finished_state = EpochFinishedState(gb_context, event_system=EventSystem.instance())
stopped_state = StoppedState(gb_context, event_system=None)

state_machine = StateMachine(context=gb_context, initial_state=setup_state)

state_machine.add(SetupState.setup_complete, generate_sequence_state)

state_machine.add(GenerateSequenceState.sequence_finished, sequence_finished_state)

state_machine.add(SequenceFinishedState.epoch_finished, epoch_finished_state)
state_machine.add(SequenceFinishedState.epoch_not_finished, go_home_decision_state)

state_machine.add(GoHomeDecisionState.go_home, generate_home_sequence_state)
state_machine.add(GoHomeDecisionState.generate_sequence, generate_sequence_state)

state_machine.add(EpochFinishedState.epoch_set_complete, stopped_state)
state_machine.add(EpochFinishedState.epoch_set_not_complete, go_home_decision_state)


@observes("epoch-complete")
def log_progress(context: GoalBabblingContext) -> None:
    msg = f"Epoch {context.runtime_data.epoch_index}: "
    msg += f"RMSE: {context.runtime_data.performance_error * 1000}mm"
    print(msg)


state_machine.run()

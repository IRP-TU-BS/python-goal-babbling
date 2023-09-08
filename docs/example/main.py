import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from goals import X_TEST_MM, X_TRAIN_MM
from utils import FileLLMCache, ForwardModel, InverseEstimator

from pygb import (
    EventSystem,
    GBHomeWeightGenerator,
    GBNoiseGenerator,
    GBParameterIncrement,
    GBParameters,
    GBParameterStore,
    LinearPathGenerator,
    GBWeightGenerator,
    GoalBabblingContext,
    GoalSet,
    GoalStore,
    IntrinsicMotivationGoalSelector,
    RandomGoalSelector,
    StateMachine,
    observes,
    setup_logging,
    vanilla_goal_babbling,
)
from pygb.states import (
    EpochFinishedState,
    EpochSetFinishedState,
    GenerateHomeSequenceState,
    GenerateSequenceState,
    GoHomeDecisionState,
    SequenceFinishedState,
    SetupState,
    StoppedState,
)
from pygb.tracking import MLFlowWrapper

setup_logging(log_level=logging.INFO)

X_TRAIN_M = X_TRAIN_MM / 1000
X_TEST_M = X_TEST_MM / 1000

X_TRAIN_M = X_TRAIN_M[:200]
X_TEST_M = X_TEST_M[:100]

JOINT_LIMITS = np.array(
    [
        [np.deg2rad(-170), np.deg2rad(-170), np.deg2rad(-170), 5.0 / 1000, 10.0 / 1000.0, 15.0 / 1000.0],
        [np.deg2rad(170), np.deg2rad(170), np.deg2rad(170), 110.0 / 1000.0, 165.0 / 1000.0, 210.0 / 1000.0],
    ]
)
HOME_JOINTS = np.array([0.0, 0.0, 0.0, 55.0 / 1000.0, 82.5 / 1000.0, 105.0 / 1000.0])

forward_model = ForwardModel(joint_limits=JOINT_LIMITS)
inverse_estimator = InverseEstimator(
    observation0=forward_model.forward(HOME_JOINTS), action0=HOME_JOINTS, radius=0.00963843, learning_rate=0.6055518
)


gb_parameters = GBParameters(
    sigma=0.0096249,
    sigma_delta=0.1,
    dim_act=6,
    dim_obs=3,
    len_sequence=10,
    len_epoch=10,
    len_epoch_set=2,
    go_home_chance=0.1,
    home_action=HOME_JOINTS,
    home_observation=forward_model.forward(HOME_JOINTS),
)

goal_set = GoalSet(train=X_TRAIN_M, test=X_TEST_M)

model_path = Path(__file__).parent.joinpath(".models")
model_path.mkdir(exist_ok=True)

gb_context = GoalBabblingContext(
    param_store=GBParameterStore(gb_parameters),  # or list of parameters per epoch set
    goal_store=GoalStore(goal_set),  # or list of goal sets per epoch set
    forward_model=forward_model,
    inverse_estimate=inverse_estimator,
    model_store=FileLLMCache(model_path),
)

setup_state = SetupState(gb_context)
generate_sequence_state = GenerateSequenceState(
    gb_context,
    goal_selector=IntrinsicMotivationGoalSelector(
        window_size=12, gamma=0.5, lambda_=0.5, event_system=EventSystem.instance()
    ),
    goal_sequence_generator=LinearPathGenerator(),
    noise_generator=GBNoiseGenerator(gb_context),
    weight_generator=GBWeightGenerator(),
    event_system=EventSystem.instance(),
)
go_home_decision_state = GoHomeDecisionState(gb_context, event_system=EventSystem.instance())
generate_home_sequence_state = GenerateHomeSequenceState(
    gb_context,
    home_sequence_generator=LinearPathGenerator(),
    weight_generator=GBHomeWeightGenerator(),
    event_system=EventSystem.instance(),
)
sequence_finished_state = SequenceFinishedState(gb_context, event_system=EventSystem.instance())
epoch_finished_state = EpochFinishedState(gb_context, event_system=EventSystem.instance())
epoch_set_finished_state = EpochSetFinishedState(gb_context, EventSystem.instance(), load_previous_best=True)
stopped_state = StoppedState(gb_context, event_system=None)

state_machine = StateMachine(context=gb_context, initial_state=setup_state)

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

# alternatively use a pre-configured state machine (example shows using multiple epoch sets):

# state_machine = vanilla_goal_babbling(
#     [gb_parameters, GBParameterIncrement(len_epoch_set=5)],
#     [goal_set, goal_set],
#     forward_model,
#     inverse_estimator,
#     FileLLMStore(model_path),
#     IntrinsicMotivationGoalSelector(window_size=10, gamma=0.5, lambda_=0.5, event_system=EventSystem.instance()),
# )


@observes("epoch-complete")
def log_progress(context: GoalBabblingContext) -> None:
    msg = f"Epoch {context.runtime_data.epoch_index}: "
    msg += f"RMSE: {context.runtime_data.performance_error * 1000}mm"
    print(msg)


mlflow_wrapper = MLFlowWrapper(experiment_name="dummy_experiment", parent_run=datetime.now().strftime("%y%m%d-%H%M%S"))
EventSystem.instance().register_observer("epoch-complete", mlflow_wrapper.epoch_complete_callback)
EventSystem.instance().register_observer("epoch-set-complete", mlflow_wrapper.epoch_set_complete_callback)

with mlflow_wrapper:
    state_machine.run()

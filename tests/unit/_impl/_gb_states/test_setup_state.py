from unittest.mock import MagicMock, PropertyMock

import numpy as np

from pygb import GoalBabblingContext, GoalSet, RuntimeData
from pygb.states import SetupState


def test_execute_state_sets_up_runtime_data() -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        current_goal_set=PropertyMock(spec=GoalSet, train=np.array([[1.0], [2.0]])),
        runtime_data=MagicMock(spec=RuntimeData),
    )
    state = SetupState(context_mock)

    state()

    assert context_mock.runtime_data.train_goal_error == [0.0, 0.0]
    assert context_mock.runtime_data.train_goal_visit_count == [0, 0]


def test_execute_state_returns_transition_name() -> None:
    context_mock = MagicMock(spec=GoalBabblingContext, runtime_data=MagicMock(spec=RuntimeData))
    state = SetupState(context_mock)

    assert state() == SetupState.setup_complete

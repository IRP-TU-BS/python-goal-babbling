from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from pygb import (
    ActionSequence,
    GBParameters,
    GoalBabblingContext,
    ObservationSequence,
    RuntimeData,
)
from pygb.states import GoHomeDecisionState


@pytest.mark.parametrize(
    ("rng_return", "expected_transition"),
    [
        (0.05, GoHomeDecisionState.go_home),
        (0.1, GoHomeDecisionState.go_home),
        (0.5, GoHomeDecisionState.generate_sequence),
    ],
)
def test_go_home(rng_return: float, expected_transition: str) -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        current_parameters=PropertyMock(
            spec=GBParameters,
            go_home_chance=0.1,
        ),
        runtime_data=MagicMock(spec=RuntimeData, previous_sequence=ObservationSequence(None, None)),
    )

    state = GoHomeDecisionState(context_mock)
    state._rng = MagicMock(random=lambda: rng_return)

    assert state() == expected_transition


def test_execute_state_returns_generate_sequence_if_previous_sequence_is_action_sequence() -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        current_parameters=PropertyMock(
            spec=GBParameters,
            go_home_chance=0.1,
        ),
        runtime_data=MagicMock(spec=RuntimeData, previous_sequence=ActionSequence(None, None)),
    )

    state = GoHomeDecisionState(context_mock)
    assert state() == GoHomeDecisionState.generate_sequence

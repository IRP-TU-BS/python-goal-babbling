from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from pygb import GBParameters, GoalBabblingContext
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
        spec=GoalBabblingContext, current_parameters=PropertyMock(spec=GBParameters, go_home_chance=0.1)
    )

    state = GoHomeDecisionState(context_mock)
    state._rng = MagicMock(random=lambda: rng_return)

    assert state() == expected_transition

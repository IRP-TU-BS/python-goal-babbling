from typing import Any

import numpy as np
import pytest

from pygb import GoalSet, GoalStore


def test_goal_set_equality() -> None:
    assert GoalSet(np.array([1.0, 2.0]), np.array([10.0, 20.0])) == GoalSet(
        np.array([1.0, 2.0]), np.array([10.0, 20.0])
    )
    assert not GoalSet(np.array([1.0, 2.0]), np.array([10.0, 20.0])) == GoalSet(
        np.array([1.0, 2.0]), np.array([10.0, 35.0])
    )
    assert not GoalSet(np.array([1.0, 2.0]), np.array([10.0, 20.0])) == GoalSet(
        np.array([1.0, 2.0]), np.array([10.0, 20.0]), {"foobar": np.array([-1.0])}
    )
    assert GoalSet(np.array([1.0, 2.0]), np.array([10.0, 20.0]), {"foobar": np.array([-1.0])}) == GoalSet(
        np.array([1.0, 2.0]), np.array([10.0, 20.0]), {"foobar": np.array([-1.0])}
    )
    assert not GoalSet(np.array([1.0, 2.0]), np.array([10.0, 20.0]), {"foobar": np.array([-1.0])}) == GoalSet(
        np.array([1.0, 2.0]), np.array([10.0, 20.0]), {"foobar": np.array([-42.0])}
    )


@pytest.mark.parametrize(
    ("goal_set", "expected_parameters"),
    [
        (
            GoalSet(np.ones((2, 1)), np.ones((1, 1)), optional_test={"foo": np.ones((4, 1))}, name="foobar"),
            {"dataset_name": "foobar", "train_size": 2, "test_size": 1, "foo_size": 4},
        ),
        (
            GoalSet(np.ones((2, 1)), np.ones((1, 1))),
            {"dataset_name": "-", "train_size": 2, "test_size": 1},
        ),
    ],
)
def test_goal_set_parameters(goal_set: GoalSet, expected_parameters: dict[str, Any]) -> None:
    assert goal_set.parameters() == expected_parameters


def test_goal_store() -> None:
    store = GoalStore(GoalSet(np.array([1.0, 2.0]), np.array([10.0, 20.0])))
    assert store[0] == GoalSet(np.array([1.0, 2.0]), np.array([10.0, 20.0]))

    store = GoalStore([GoalSet(np.array([1.0, 2.0]), np.array([10.0, 20.0]))])
    assert store[0] == GoalSet(np.array([1.0, 2.0]), np.array([10.0, 20.0]))

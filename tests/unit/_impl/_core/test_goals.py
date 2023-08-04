import numpy as np

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


def test_goal_store() -> None:
    store = GoalStore(GoalSet(np.array([1.0, 2.0]), np.array([10.0, 20.0])))
    assert store[0] == GoalSet(np.array([1.0, 2.0]), np.array([10.0, 20.0]))

    store = GoalStore([GoalSet(np.array([1.0, 2.0]), np.array([10.0, 20.0]))])
    assert store[0] == GoalSet(np.array([1.0, 2.0]), np.array([10.0, 20.0]))

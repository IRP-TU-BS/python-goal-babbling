import numpy as np
from numpy.random import Generator

from pygb._impl._core._abstract_utils import AbstractGoalSelector
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._runtime_data import ActionSequence, ObservationSequence


class RandomGoalSelector(AbstractGoalSelector[GoalBabblingContext]):
    """Random goal selector class."""

    def __init__(self, rng: Generator = np.random.default_rng()) -> None:
        """Constructor.

        Args:
            rng: Numpy random number generator. Defaults to a randomly initialized RNG.
        """
        self._rng = rng

    def select(self, context: GoalBabblingContext) -> tuple[int, np.ndarray]:
        """Selects one goal randomly that is different from the previous sequence's stop goal.

        Args:
            context: Goal Babbling context.

        Returns:
            Randomly selected goal index and the goal itself.
        """
        if context.runtime_data.previous_sequence is None or isinstance(
            context.runtime_data.previous_sequence, ActionSequence
        ):
            prev_observation = context.current_parameters.home_observation
        else:
            prev_observation = context.runtime_data.previous_sequence.stop_goal

        selected_index = None
        while selected_index is None or np.all(context.current_goal_set.train[selected_index] == prev_observation):
            selected_index = self._rng.integers(0, context.current_goal_set.train.shape[0], size=None)

        return selected_index, context.current_goal_set.train[selected_index]

import numpy as np

from pygb._impl._core._abstract_utils import AbstractGoalSelector
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._runtime_data import ActionSequence, ObservationSequence


class RandomGoalSelector(AbstractGoalSelector[GoalBabblingContext]):
    """Random goal selector class."""

    def __init__(self, random_seed: int | None = None) -> None:
        """Constructor.

        Args:
            random_seed: Random seed which is used to initilaize a numpy random number generator. Defaults to None.
        """
        self._rng = np.random.default_rng(seed=random_seed)

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
            # start of epoch set or we've just been at home -> choose any goal randomly
            random_idx = self._rng.integers(0, context.current_goal_set.train.shape[0], size=None)
            selected_goal = context.current_goal_set.train[random_idx]

        elif isinstance(context.runtime_data.previous_sequence, ObservationSequence):
            previous_goal = context.runtime_data.previous_sequence.stop_goal
            selected_goal = None

            while selected_goal is None or np.all(previous_goal == selected_goal):
                random_idx = self._rng.integers(0, context.current_goal_set.train.shape[0], size=None)
                selected_goal = context.current_goal_set.train[random_idx]

        return random_idx, selected_goal

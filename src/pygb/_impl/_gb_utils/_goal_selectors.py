import numpy as np

from pygb._impl._core._abstract_utils import AbstractGoalSelector
from pygb._impl._core._context import GoalBabblingContext


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
        previous_goal_idx = context.runtime_data.sequences[-1].stop_glob_goal_idx
        while (
            random_idx := self._rng.integers(0, context.current_goal_set.train.shape[0], size=None)
        ) == previous_goal_idx:
            continue

        return random_idx, context.current_goal_set.train[random_idx]

import numpy as np

from pygb._impl._core._abstract_utils import AbstractLocalGoalGenerator
from pygb._impl._core._context import GoalBabblingContext


class GBPathGenerator(AbstractLocalGoalGenerator[GoalBabblingContext]):
    """Linear path generator."""

    def __init__(self) -> None:
        super().__init__()

    def generate(self, start_goal: np.ndarray, stop_goal: np.ndarray, len_sequence: int) -> list[np.ndarray]:
        """Generates a linear sequence of local observations between two global goals.

        The generated sequence excludes the start goal (assumption: start goal has just been visited in the previous
        sequence) but does include the stop goal.

        Args:
            context: Goal Babbling context.

        Returns:
            A seqence of observations on a linear path between a start and a stop goal.

        Raises:
            RuntimeError: If start and end goal are equal.
        """
        if np.all(start_goal == stop_goal):
            raise RuntimeError(
                f"""Failed to generate linear path between goals: Start goal ({start_goal}) """
                f"""and stop goal ({stop_goal}) are equal."""
            )
        local_goals = []

        # exclude start and include stop goal:
        for observation_index in range(1, len_sequence + 1):
            rel_progress = observation_index / len_sequence
            local_goals.append(rel_progress * stop_goal + (1.0 - rel_progress) * start_goal)

        return local_goals

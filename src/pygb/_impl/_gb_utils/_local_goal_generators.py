import numpy as np

from pygb._impl._core._abstract_context import ContextType
from pygb._impl._core._abstract_utils import AbstractSequenceGenerator
from pygb._impl._core._context import GoalBabblingContext


class LinearPathGenerator(AbstractSequenceGenerator[GoalBabblingContext]):
    """Linear path generator."""

    def __init__(self) -> None:
        super().__init__()

    def generate(
        self, start: np.ndarray, stop: np.ndarray, len_sequence: int, context: ContextType | None = None
    ) -> list[np.ndarray]:
        """Generates a linear sequence of local observations between two global goals.

        The generated sequence excludes the start goal (assumption: start goal has just been visited in the previous
        sequence) but does include the stop goal.

        Args:
            start: Start observation, e.g. a global goal.
            stop: Stop (/target) observation, e.g. a global goal.
            len_sequence: Length of the sequence, i.e. how many steps make one sequence (excluding start, but including
                stop).
            context: Goal Babbling context (unused).

        Returns:
            A seqence of observations on a linear path between a start and a stop goal.

        Raises:
            RuntimeError: If start and end goal are equal.
        """
        if np.all(start == stop):
            raise RuntimeError(
                f"""Failed to generate linear path between goals: Start goal ({start}) """
                f"""and stop goal ({stop}) are equal."""
            )
        local_goals = []

        # exclude start and include stop goal:
        for observation_index in range(1, len_sequence + 1):
            rel_progress = observation_index / len_sequence
            local_goals.append(rel_progress * stop + (1.0 - rel_progress) * start)

        return local_goals

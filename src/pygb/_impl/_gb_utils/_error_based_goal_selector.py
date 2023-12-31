import numpy as np
from numpy.random import Generator

from pygb._impl._core._abstract_utils import AbstractGoalSelector
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._runtime_data import ActionSequence, ObservationSequence


class ErrorBasedGoalSelector(AbstractGoalSelector[GoalBabblingContext]):
    """Goal selector class which selects a goal randomly from the top X% training goals with highest prediction
    error."""

    def __init__(self, select_from_top: float = 0.25, rng: Generator = np.random.default_rng()) -> None:
        """Constructor.

        Args:
            select_from_top: Defines the interval size of high-error goals to choose from. Defaults to 0.25.
            rng: Numpy random number generator. Defaults to a randomly initialized RNG.
        """
        self._rng = rng

        self.select_from_top = select_from_top

    def select(self, context: GoalBabblingContext) -> tuple[int, np.ndarray]:
        """Selects one goal randomly of the top X% of high-error goals.

        Selection strategy:
            - selects an unvisited goal randomly if at least one exists or training has just started
            - selects one goal of the top X% of high-error goals, where X is defined by 'select_from_top'

        Args:
            context: Goal Babbling context. Used to read the training goal error stats.

        Returns:
            Selected goal index and goal.
        """
        if context.runtime_data.previous_sequence is None:
            # start of epoch set or we've just been at home -> choose any goal randomly
            goal_index: int = self._rng.integers(0, context.current_goal_set.train.shape[0], size=None)
            selected_goal = context.current_goal_set.train[goal_index]
        else:
            previous_goal = (
                context.runtime_data.previous_sequence.stop_goal
                if isinstance(context.runtime_data.previous_sequence, ObservationSequence)
                else context.forward_model.forward(context.runtime_data.previous_sequence.stop_action)
            )
            if np.any(np.array(context.runtime_data.train_goal_visit_count) == 0):
                goal_index, selected_goal = self._select_unvisited_goal(previous_goal, context)
            else:
                goal_index, selected_goal = self._select_goal_by_error(previous_goal, context)

        return goal_index, selected_goal

    def _select_unvisited_goal(self, previous_goal: np.ndarray, context: GoalBabblingContext) -> tuple[int, np.ndarray]:
        selected_goal = None
        while selected_goal is None or np.all(selected_goal == previous_goal):
            not_visited_indices = np.argwhere(np.array(context.runtime_data.train_goal_visit_count) == 0).reshape(-1)
            goal_index = self._rng.choice(not_visited_indices, size=1, replace=False)[0]
            selected_goal = context.current_goal_set.train[goal_index]

        return goal_index, selected_goal

    def _select_goal_by_error(self, previous_goal: np.ndarray, context: GoalBabblingContext) -> tuple[int, np.ndarray]:
        sorted_indices = list(reversed(np.argsort(context.runtime_data.train_goal_error)))

        # select top self.select_from_top % goal indices :
        sorted_indices = sorted_indices[: int(self.select_from_top * len(sorted_indices))]

        selected_goal = None
        while selected_goal is None or np.all(selected_goal == previous_goal):
            goal_index = self._rng.choice(sorted_indices, size=1, replace=False)[0]
            selected_goal = context.current_goal_set.train[goal_index]

        return goal_index, selected_goal

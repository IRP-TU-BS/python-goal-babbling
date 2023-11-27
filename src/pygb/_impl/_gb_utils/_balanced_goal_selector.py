import numpy as np
from numpy.random import Generator

from pygb._impl._core._abstract_utils import AbstractGoalSelector
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._runtime_data import ActionSequence


class BalancedGoalSelector(AbstractGoalSelector[GoalBabblingContext]):
    """Goal selector which selects goals based on highest prediction errors and lowest visit count.

    A goal is randomly selected among unvisited training goals as long as there exist training goals which have never
    been visited. As soon as every training goal has been visited at least once, target goals are selected either based
    on their prediction errors (on the last visit) or based on their visit counts."""

    def __init__(
        self,
        error_percentile: float = 0.25,
        count_percentile: float = 0.25,
        ratio: float = 0.5,
        rng: Generator = np.random.default_rng(),
    ) -> None:
        """Constructor.

        Args:
            error_percentile: Defines the percentile of the training goals which is used to pick a high-error goal. E.g.
                if set to 0.25/25%, a random selection is done from the top 25% high prediction error training goals.
                Defaults to 0.25.
            count_percentile: Defines the precentile of the training goals which is used to pick a training goal wiht
                low visit count. E.g. if set to 0.25/25%, a random selection is done from the bottom 25% of goals with
                low visit counts. Defaults to 0.25.
            ratio: Defines the ratio of goal selections by error over goal selections by visit count. E.g. if set to
                0.7/70%, 70% of selections are done based on goals' prediction errors, and 30% of selections are done
                based on the goals' visit counts. Defaults to 0.5.
            rng: Numpy random number generator. Defaults to np.random.default_rng().
        """
        self.error_percentile = error_percentile
        self.count_percentile = count_percentile
        self.ratio = ratio
        self.rng = rng

        self.stats: dict[str, int] = {"n_random": 0, "n_by_error": 0, "n_by_count": 0}

    def select(self, context: GoalBabblingContext) -> tuple[int, np.ndarray]:
        """Selects a new target goal.

        Args:
            context: GoalBabblingContext instance.

        Returns:
            A tuple containing the goal index and the goal as a numpy array.
        """
        visit_count = np.asanyarray(context.runtime_data.train_goal_visit_count)
        if np.any(visit_count == 0):
            unvisited_idx = np.argwhere(visit_count == 0).reshape(-1)
            goal_idx = self.rng.choice(unvisited_idx)
            goal = context.current_goal_set.train[goal_idx]

            self.stats["n_random"] += 1
        else:
            if context.runtime_data.previous_sequence is None or isinstance(
                context.runtime_data.previous_sequence, ActionSequence
            ):
                prev_observation = context.current_parameters.home_observation
            else:
                prev_observation = context.current_goal_set.train[
                    context.runtime_data.previous_sequence.stop_goal_index
                ]

            if self.rng.random() <= self.ratio:
                goal_idx, goal = self._choose_goal_by_error(
                    context.runtime_data.train_goal_error, context.current_goal_set.train, prev_observation
                )
                self.stats["n_by_error"] += 1
            else:
                goal_idx, goal = self._choose_goal_by_visit_count(
                    context.runtime_data.train_goal_visit_count, context.current_goal_set.train, prev_observation
                )
                self.stats["n_by_count"] += 1

        return goal_idx, goal

    def _choose_goal_by_error(
        self, goal_errors: list[float], goals: np.ndarray, prev_observation: int
    ) -> tuple[int, np.ndarray]:
        sorted_idx = np.argsort(goal_errors)[::-1]

        selected_idx = None
        while selected_idx is None or np.all(goals[selected_idx] == prev_observation):
            selected_idx = self.rng.choice(sorted_idx[: int(len(sorted_idx) * self.error_percentile)], replace=False)

        return selected_idx, goals[selected_idx]

    def _choose_goal_by_visit_count(
        self, goal_visit_counts: list[int], goals: np.ndarray, prev_observation: np.ndarray
    ) -> tuple[int, np.ndarray]:
        sorted_idx = np.argsort(goal_visit_counts)

        selected_idx = None
        while selected_idx is None or np.all(goals[selected_idx] == prev_observation):
            selected_idx = self.rng.choice(sorted_idx[: int(len(sorted_idx) * self.count_percentile)], replace=False)

        return selected_idx, goals[selected_idx]

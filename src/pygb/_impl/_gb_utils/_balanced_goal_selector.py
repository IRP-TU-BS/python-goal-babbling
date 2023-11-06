import numpy as np
from numpy.random import Generator

from pygb._impl._core._abstract_utils import AbstractGoalSelector
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._runtime_data import ActionSequence


class BalancedGoalSelector(AbstractGoalSelector[GoalBabblingContext]):
    def __init__(
        self,
        error_percentile: float = 0.25,
        count_percentile: float = 0.25,
        ratio: float = 0.5,
        rng: Generator = np.random.default_rng(),
    ) -> None:
        self.error_percentile = error_percentile
        self.count_percentile = count_percentile
        self.ratio = ratio
        self.rng = rng

        self.stats: dict[str, int] = {"n_random": 0, "n_by_error": 0, "n_by_count": 0}

    def select(self, context: GoalBabblingContext) -> tuple[int, np.ndarray]:
        visit_count = np.asanyarray(context.runtime_data.train_goal_visit_count)
        if np.any(visit_count == 0):
            unvisited_idx = np.argwhere(visit_count == 0).reshape(-1)
            goal_idx = self.rng.choice(unvisited_idx)

            self.stats["n_random"] += 1
        else:
            if context.runtime_data.previous_sequence is None or isinstance(
                context.runtime_data.previous_sequence, ActionSequence
            ):
                prev_goal_idx = -1
            else:
                prev_goal_idx = context.runtime_data.previous_sequence.stop_goal_index

            if self.rng.random() <= self.ratio:
                goal_idx = self._choose_goal_by_error(context.runtime_data.train_goal_error, prev_goal_idx)
                self.stats["n_by_error"] += 1
            else:
                goal_idx = self._choose_goal_by_visit_count(context.runtime_data.train_goal_visit_count, prev_goal_idx)
                self.stats["n_by_count"] += 1

        return goal_idx, context.current_goal_set.train[goal_idx]

    def _choose_goal_randomly(self, goal_count: int, prev_goal_idx: int) -> int:
        selected_idx = None

        while selected_idx is None or selected_idx == prev_goal_idx:
            selected_idx = self.rng.integers(low=0, high=goal_count)

        return selected_idx

    def _choose_goal_by_error(self, goal_errors: list[float], prev_goal_idx: int) -> int:
        sorted_idx = np.argsort(goal_errors)[::-1]

        selected_idx = None
        while selected_idx is None or selected_idx == prev_goal_idx:
            selected_idx = self.rng.choice(sorted_idx[: int(len(sorted_idx) * self.error_percentile)], replace=False)

        return selected_idx

    def _choose_goal_by_visit_count(self, goal_visit_counts: list[int], prev_goal_idx: int) -> int:
        sorted_idx = np.argsort(goal_visit_counts)

        selected_idx = None
        while selected_idx is None or selected_idx == prev_goal_idx:
            selected_idx = self.rng.choice(sorted_idx[: int(len(sorted_idx) * self.count_percentile)], replace=False)

        return selected_idx

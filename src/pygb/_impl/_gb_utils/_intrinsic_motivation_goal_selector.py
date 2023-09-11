import numpy as np
from numpy.random import Generator

from pygb._impl._core._abstract_utils import AbstractGoalSelector
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._event_system import EventSystem
from pygb._impl._core._events import Events
from pygb._impl._core._runtime_data import ObservationSequence


class IntrinsicMotivationGoalSelector(AbstractGoalSelector[GoalBabblingContext]):
    """Goal selector which implements intrinsically motivated goal selection (Rayyes et al., 2020).

    Goals are selected using an interest score composed of a relative error RE(g_i) and a forgetting factor FF(g_i):

        interest(g_i) = l * RE(g_i) + (1 - l) * FF(g_i)

    where g_i is the i-th training goal and l (lambda_) is between 0 and 1 and weights the influence of RE(g_i) over
    FF(g_i). The relative error is defined as

        RE(g_i) = (E(g_i) - E_min) / (E_max - E_min)

    where E(g_i) is the most recent prediction error on g_i, and E_min/E_max are the minimum and maximum prediction
    errors across all training goals.

    FF(g_i) is composed of the current progress PROG(g_i) and the general progress overview REG(g_i):

        FF(g_i) = g * PROG(g_i) + (1 - g) * REG(g_i)

    where g (gamma) is between 0 and 1 and weights the influence of PROG(g_i) over REG(g_i). PROG(g_i) measures the
    performance on g_i over a sliding window with width n and is normalized relative to maximum and minimum progress
    values across all training goals:

        PROG(g_i) = ( sum_{j=n/2 -> j=n}(E_j(g_i)) - sum_{j=1 -> j=n/2}(E_j(g_i)) ) / n
        PROG(g_i) = (PROG(g_i) - Prog_min) / (Prog_max - Prog_min)

    The general progress overview REG(g_i) is defined as

        REG(g_i) = ( E(g_i) - E_min(g_i) ) / ( E_max(g_i) - E_min(g_i) )

    Goals are chosen by finding the highest interest score. Exceptions to this rule are:

        1) Not every goal has been visited n times, i.e. the defined sliding window is not fully filled. In this case a
            random goal is selected from the subset of goals with 'open' sliding windows.
        2) The most interesting goal has been visited in the previous sequence. In this case the goal with the second
            highest goal is selected.

    Instances of this class register for 'sequence-finished' events by default. All necessary data is collected and
    stored for the duration of one epoch set. Training goals may change between epoch sets, which is why the collected
    data is reset after each epoch set.
    """

    def __init__(
        self,
        window_size: int,
        gamma: float,
        lambda_: float,
        event_system: EventSystem,
        rng: Generator = np.random.default_rng(),
    ) -> None:
        """Constructor.

        Args:
            window_size: Widht of the sliding window. Must be an even number.
            gamma: Weighting factor between 0 and 1. Weights PROG(g_i) over REG(g_i).
            lambda_: Weighting factor between 0 and 1. Weights RE(g_i) over FF(g_i).
            event_system: Event system singleton instance.
            rng: Numpy random number generator. Defaults to a randomly initialized RNG.

        Raises:
            ValueError: If the window size is uneven.
        """
        if window_size % 2 != 0:
            raise ValueError(
                f"Failed to initialize {self.__class__.__qualname__}: Window size '{window_size}' is no even number."
            )

        self.gamma = gamma
        self.lambda_ = lambda_
        self.window_size = window_size

        # matrix keeping track of errors per train goal (column === goal index, row === past prediction errors)
        # newest error at index 0, oldest array at index window_size-1
        self._goal_error_matrix: np.ndarray | None = None
        self._goals_e_max: np.ndarray | None = None
        self._goals_e_min: np.ndarray | None = None
        self._valid_for_epoch_set: int | None = None

        self._rng = rng

        event_system.register_observer(Events.SEQUENCE_FINISHED, self._update_data_callback)

    def select(self, context: GoalBabblingContext) -> tuple[int, np.ndarray]:
        """Selects a new target goal.

        See the class docstring for details on the selection process.

        Args:
            context: Goal Babbling context.

        Returns:
            The selected goal index and goal.
        """
        if self._goal_error_matrix is None or context.runtime_data.epoch_set_index != self._valid_for_epoch_set:
            self._init(context)

        previous_index = -1
        if context.runtime_data.previous_sequence is not None and isinstance(
            context.runtime_data.previous_sequence, ObservationSequence
        ):
            previous_index = context.runtime_data.previous_sequence.stop_goal_index

        # self._goal_error_matrix is asserted to not be None before
        if np.any(self._goal_error_matrix == 0):
            # windows not filled
            goal_index = self._select_random_index(self._goal_error_matrix, previous_index)  # type: ignore
        else:
            goal_index = self._select_goal_by_interest(
                self._goal_error_matrix, self._goals_e_min, self._goals_e_max, previous_index  # type: ignore
            )

        return goal_index, context.current_goal_set.train[goal_index]

    def _select_random_index(self, goal_error_matrix: np.ndarray, previous_goal_index: int) -> int:
        open_window_indices = [idx for idx, sub in enumerate(goal_error_matrix) if np.any(sub == 0)]
        index = None

        while index is None or index == previous_goal_index:
            index = self._rng.choice(open_window_indices, size=None, replace=True)

        return index

    def _init(self, context: GoalBabblingContext) -> None:
        self._goal_error_matrix = np.zeros(shape=(context.current_goal_set.train.shape[0], self.window_size))
        self._goals_e_max = np.zeros(shape=(context.current_goal_set.train.shape[0],))
        self._goals_e_min = np.inf * np.ones(shape=(context.current_goal_set.train.shape[0]))
        self._valid_for_epoch_set = context.runtime_data.epoch_set_index

    def _update_data_callback(self, context: GoalBabblingContext) -> None:
        if self._goal_error_matrix is None or context.runtime_data.epoch_set_index != self._valid_for_epoch_set:
            self._init(context)

        if isinstance(context.runtime_data.current_sequence, ObservationSequence):
            goal_index = context.runtime_data.current_sequence.stop_goal_index
            self._update_goal_error(goal_index, context.runtime_data.train_goal_error[goal_index])

    def _update_goal_error(self, goal_index, goal_error: float) -> None:
        if self._goal_error_matrix is None or self._goals_e_max is None or self._goals_e_min is None:
            raise RuntimeError("Failed to update goal errors: Instance is not initialized. Call init() first.")

        # make room for newest goal error at index 0:
        self._goal_error_matrix[goal_index] = np.roll(self._goal_error_matrix[goal_index], 1)
        # insert newest error:
        self._goal_error_matrix[goal_index, 0] = goal_error

        self._goals_e_min[goal_index] = min(goal_error, self._goals_e_min[goal_index])
        self._goals_e_max[goal_index] = max(goal_error, self._goals_e_max[goal_index])

    def _relative_errors(self, goal_error_matrix: np.ndarray, delta: float = 1e-9) -> np.ndarray:
        e_min = np.min(goal_error_matrix[:, 0])
        e_max = np.max(goal_error_matrix[:, 0])

        denominator = e_max - e_min

        if denominator <= delta:
            # all errors are almost equal
            return np.ones(shape=(goal_error_matrix.shape[0],)) / goal_error_matrix.shape[0]

        return (goal_error_matrix[:, 0] - e_min) / denominator

    def _forgetting_factors(
        self, goal_error_matrix: np.ndarray, goals_e_min: np.ndarray, goals_e_max: np.ndarray
    ) -> np.ndarray:
        return self.gamma * self._current_progresses(goal_error_matrix) + (
            1 - self.gamma
        ) * self._general_progress_overviews(goal_error_matrix, goals_e_min, goals_e_max)

    def _current_progresses(self, goal_error_matrix: np.ndarray, delta: float = 1e-9) -> np.ndarray:
        half = self.window_size // 2
        e_second_half = np.sum(goal_error_matrix[:, :half], axis=1)
        e_first_half = np.sum(goal_error_matrix[:, half:], axis=1)

        current_progress = (e_second_half - e_first_half) / self.window_size
        prog_max = np.max(current_progress)
        prog_min = np.min(current_progress)

        denominator = prog_max - prog_min

        if denominator <= delta:
            return np.ones(shape=(goal_error_matrix.shape[0])) / goal_error_matrix.shape[0]

        # normalize:
        return (current_progress - prog_min) / denominator

    def _general_progress_overviews(
        self, goal_error_matrix: np.ndarray, goals_e_min: np.ndarray, goals_e_max: np.ndarray
    ) -> np.ndarray:
        reg = []  # general progress overview
        for goal_index in range(goal_error_matrix.shape[0]):
            e_min_goal = goals_e_min[goal_index]
            e_max_goal = goals_e_max[goal_index]
            reg.append((goal_error_matrix[goal_index, 0] - e_min_goal) / (e_max_goal - e_min_goal))

        return np.array(reg)

    def _select_goal_by_interest(
        self, goal_error_matrix: np.ndarray, goals_e_min: np.ndarray, goals_e_max: np.ndarray, previous_index: int
    ) -> int:
        interest_values = self.lambda_ * self._relative_errors(goal_error_matrix) + (
            1 - self.lambda_
        ) * self._forgetting_factors(goal_error_matrix, goals_e_min, goals_e_max)

        sorted_indices = np.argsort(interest_values)
        index = sorted_indices[-1]
        if index == previous_index:
            return sorted_indices[-2]

        return index

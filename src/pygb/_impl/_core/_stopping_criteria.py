from dataclasses import dataclass
from datetime import datetime, timedelta

from pygb._impl._core._abstract_context import ContextType
from pygb._impl._core._abstract_stopping_criteria import AbstractStoppingCriteria
from pygb._impl._core._context import GoalBabblingContext


@dataclass
class TargetPerformanceStop(AbstractStoppingCriteria[GoalBabblingContext]):
    """Stopping cirterion which stops the training as soon as a target performance error is reached."""

    def __init__(self, performance: float, start_epoch: int = 0) -> None:
        """Constructor.

        Args:
            performance: Target performance (e.g. RMSE).
            start_epoch: Epoch at which this stopping criterion becomes active (starting at 0). Defaults to 0.
        """
        self.performance = performance
        self.start_epoch = start_epoch

    def fulfilled(self, context: GoalBabblingContext) -> bool:
        """Checks if the target performance error is reached.

        Args:
            context: Goal Babbling context. Used to read the current performance error.

        Returns:
            True if the target performance error is reached, False otherwise.
        """
        if (
            context.runtime_data.performance_error > self.performance
            or context.runtime_data.epoch_index < self.start_epoch
        ):
            return False

        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(performance = {self.performance}, start_epoch = {self.start_epoch})"


@dataclass
class TimeBudgetStop(AbstractStoppingCriteria[GoalBabblingContext]):
    """Stopping criterion which stops training after a specified time budget is reached."""

    def __init__(self, budget: timedelta, start_epoch: int = 0) -> None:
        """Constructor.

        Args:
            budget: Time budget.
            start_epoch: Epoch at which this stopping criterion becomes active (starting at 0). Defaults to 0.
        """
        self.budget = budget
        self.start_epoch = start_epoch

        self._start: datetime | None = None

    def fulfilled(self, context: GoalBabblingContext) -> bool:
        """Checks if the time budget is reached.

        Note that the start is defined as the first call to this function, not at the initialization of this class.

        Args:
            context: Goal Babbling context (unused).

        Returns:
            True if the time budget is reached, False otherwise.
        """
        if self._start is None:
            self._start = datetime.now()
            return False

        if context.runtime_data.epoch_index < self.start_epoch:
            return False

        return datetime.now() - self._start >= self.budget

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(budget = {self.budget}, start_epoch = {self.start_epoch})"


@dataclass
class PerformanceSlopeStop(AbstractStoppingCriteria[GoalBabblingContext]):
    """Stopping criterion which stops training if the performance error increases over a specified amount of epochs."""

    def __init__(self, grace_period: int, start_epoch: int = 0) -> None:
        """Constructor.

        Args:
            grace_period: Amount of epochs in which a performance error increase is allowed.
            start_epoch: Epoch at which this stopping criterion becomes active (starting at 0). Defaults to 0.
        """
        self.grace_period = grace_period
        self.start_epoch = start_epoch

        self._best_performance: float | None = None
        self._period = 0

    def fulfilled(self, context: GoalBabblingContext) -> bool:
        """Checks if the performance error has been increasing for longer than the specified grace period.

        The counter which counts the number of epochs in which the performance has not been increasing is reset as soon
        as the previous best performance is exceeded.

        Args:
            context: Goal Babbling context. Used to read the current epoch's performance error.

        Returns:
            True if the performance has not been increasing for longer than the specified grace period, False otherwise.
        """
        if context.runtime_data.epoch_index < self.start_epoch:
            return False

        if self._best_performance is None:
            self._best_performance = context.runtime_data.performance_error
            return False

        if context.runtime_data.performance_error <= self._best_performance:
            self._period = 0
            self._best_performance = context.runtime_data.performance_error
            return False

        self._period += 1
        if self._period >= self.grace_period - 1:
            return True

        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(grace_period = {self.grace_period}, start_epoch = {self.start_epoch})"

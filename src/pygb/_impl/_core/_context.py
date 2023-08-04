from abc import ABC, abstractmethod
from typing import TypeVar

from pygb._impl._core._goals import GoalSet, GoalStore
from pygb._impl._core._model import AbstractForwardModel, AbstractInverseEstimator
from pygb._impl._core._parameters import GBParameterStore
from pygb._impl._core._runtime_data import RuntimeData


class AbstractContext(ABC):
    """Abstract (state machine) context class."""

    @abstractmethod
    def is_running(self) -> bool:
        """Checks if the state machine (which uses the context) is running.

        Returns:
            Whether or not the state machine is running.
        """

    @abstractmethod
    def set_running(self) -> None:
        """Set the state machine context to running."""

    @abstractmethod
    def set_stopped(self) -> None:
        """Set the state machine context to stopped."""


ContextType = TypeVar("ContextType", bound=AbstractContext)


class GoalBabblingContext(AbstractContext):
    def __init__(
        self,
        param_store: GBParameterStore,
        goal_store: GoalStore,
        forward_model: AbstractForwardModel,
        inverse_estimate: AbstractInverseEstimator,
        runtime_data: RuntimeData = RuntimeData(),
    ) -> None:
        self.gb_param_store = param_store
        self.goal_store = goal_store
        self.runtime_data = runtime_data
        self.forward_model = forward_model
        self.inverse_estimate = inverse_estimate

    @property
    def current_goal_set(self) -> GoalSet:
        """Return the current epoch set's goal set.

        Returns:
            Current epoch set's goal set.
        """
        return self.goal_store[self.runtime_data.epoch_set_index]

    def is_running(self) -> bool:
        # TODO
        ...

    def set_running(self) -> None:
        # TODO
        ...

    def set_stopped(self) -> None:
        # TODO
        ...

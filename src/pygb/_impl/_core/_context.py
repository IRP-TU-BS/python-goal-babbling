from abc import ABC, abstractmethod
from typing import TypeVar

from pygb._impl._core._goals import GoalStore
from pygb._impl._core._parameters import GBParameterStore
from pygb._impl._core._runtime_data import RuntimeData


class AbstractContext(ABC):
    """Abstract (state machine) context class."""

    @abstractmethod
    def is_running(self) -> bool:
        """Checks if the state machine which uses the context is running.

        Returns:
            Whether or not the state machine is running.
        """

    @abstractmethod
    def set_running(self) -> None:
        """Setter for the state machine (which uses the context) is running."""


ContextType = TypeVar("ContextType", bound=AbstractContext)


class GoalBabblingContext(AbstractContext):
    def __init__(
        self, param_store: GBParameterStore, goal_store: GoalStore, runtime_data: RuntimeData = RuntimeData()
    ) -> None:
        self.gb_param_store = param_store
        self.goal_store = goal_store
        self.runtime_data = runtime_data

    def is_running(self) -> bool:
        # TODO
        ...

    def set_running(self) -> None:
        # TODO
        ...

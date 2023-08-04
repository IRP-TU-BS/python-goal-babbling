from abc import ABC, abstractmethod
from typing import TypeVar

from pygb._impl._core._goals import GoalStore
from pygb._impl._core._parameters import GBParameterStore


class AbstractContext(ABC):
    @abstractmethod
    def is_running(self) -> bool:
        ...

    @abstractmethod
    def set_running(self) -> None:
        ...


ContextType = TypeVar("ContextType", bound=AbstractContext)


class GoalBabblingContext(AbstractContext):
    def __init__(self, param_store: GBParameterStore, goal_store: GoalStore) -> None:
        self.gb_param_store = param_store
        self.goal_store = goal_store

    def is_running(self) -> bool:
        # TODO
        ...

    def set_running(self) -> None:
        # TODO
        ...

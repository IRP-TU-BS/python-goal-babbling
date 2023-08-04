from abc import ABC, abstractmethod
from typing import Generic

from pygb._impl._core._context import ContextType
from pygb._impl._core._events import EventSystem


class AbstractState(ABC, Generic[ContextType]):
    def __init__(self, context: ContextType, event_system: EventSystem | None = None, name: str | None = None) -> None:
        self.name = name or self.__class__.__qualname__
        self.context = context
        self.events = event_system

    @abstractmethod
    def __call__(self) -> str:
        ...

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, type(self)):
            return False

        return self.name == __o.name

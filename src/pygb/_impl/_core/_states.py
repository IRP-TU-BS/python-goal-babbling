from abc import ABC, abstractmethod

from pygb._impl._core._context import AbstractContext
from pygb._impl._core._events import EventSystem


class AbstractState(ABC):
    def __init__(self, context: AbstractContext, event_system: EventSystem, name: str | None = None) -> None:
        self.name = name or self.__class__.__qualname__
        self.context = context
        self.events = event_system

    @abstractmethod
    def __call__(self) -> None:
        ...

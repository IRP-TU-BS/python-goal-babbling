from collections import defaultdict
from typing import Callable, Optional

from pygb._impl._core._context import AbstractContext


class EventSystem:
    _instance: Optional["EventSystem"] = None

    def __init__(self) -> None:
        self.event_observers: dict[str, list[Callable]] = defaultdict(list)

    @staticmethod
    def instance() -> "EventSystem":
        if EventSystem._instance is None:
            EventSystem._instance = EventSystem()

        return EventSystem._instance

    def emit(self, event: str, context: AbstractContext) -> None:
        for func in self.event_observers[event]:
            func(context)

    def register_observer(
        self, event: str, observer: Callable[[AbstractContext], None], no_raise: bool = False
    ) -> None:
        if observer in self.event_observers[event]:
            if no_raise:
                raise RuntimeError(
                    f"""Failed to register observer: '{observer.__qualname__}' is already registered for event """
                    f""" '{event}'. To suppress this exception, set 'no_raise' to True."""
                )
            return

        self.event_observers[event].append(observer)

    def remove_observer(
        self, event: str, observer: Callable[[AbstractContext], None], no_raise: bool = True
    ) -> Callable[[AbstractContext], None] | None:
        if observer not in self.event_observers[event]:
            if not no_raise:
                raise RuntimeError(
                    f"Failed to remove event observer: '{observer.__qualname__}' is not registered for event '{event}'"
                )
            return None

        self.event_observers[event].remove(observer)
        return observer


def observes(event: str) -> Callable[[AbstractContext], None]:
    def decorator(func: Callable[[AbstractContext], None]) -> None:
        EventSystem.instance().register_observer(event, func)

        return func

    return decorator

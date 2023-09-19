import logging
from collections import defaultdict
from typing import Callable, Optional

from pygb._impl._core._abstract_context import AbstractContext, ContextType

_logger = logging.getLogger(__name__)


class EventSystem:
    """Event system singleton class."""

    _instance: Optional["EventSystem"] = None

    def __init__(self) -> None:
        """Constructor."""
        self.event_observers: dict[str, list[Callable]] = defaultdict(list)

    @staticmethod
    def instance() -> "EventSystem":
        """Static getter for the global event system singleton instance.

        Returns:
            Event system instance.
        """
        if EventSystem._instance is None:
            EventSystem._instance = EventSystem()

        return EventSystem._instance

    def emit(self, event: str, context: AbstractContext) -> None:
        """Emits an event including the context instance.

        Args:
            event: Event name.
            context: Context instance.
        """
        _logger.debug("Emitting event '%s'" % event)
        for func in self.event_observers[event]:
            func(context)

    def register_observer(self, event: str, observer: Callable[[ContextType], None], no_raise: bool = False) -> None:
        """Registers a callable as an observer for the specified event.

        Args:
            event: Event name.
            observer: Callable observer.
            no_raise: If set, prevents a RuntimeError if the observer is already registered for the event. Defaults to False.

        Raises:
            RuntimeError: If the observer is already registered for the event and 'no_raise' is set to False.
        """
        if observer in self.event_observers[event]:
            if no_raise:
                raise RuntimeError(
                    f"""Failed to register observer: '{observer.__qualname__}' is already registered for event """
                    f""" '{event}'. To suppress this exception, set 'no_raise' to True."""
                )
            return

        self.event_observers[event].append(observer)
        _logger.debug("Registered observer '%s' for event '%s'" % (observer.__name__, event))

    def remove_observer(
        self, event: str, observer: Callable[[AbstractContext], None], no_raise: bool = True
    ) -> Callable[[AbstractContext], None] | None:
        """Unregisters an observer from an event.

        Args:
            event: Event name.
            observer: Callable observer.
            no_raise: If set, prevents a RuntimeError in case the observer is not registered for the specified event. Defaults to True.

        Raises:
            RuntimeError: If the observer is not registered for the specified event and 'no_raise' is set to False.

        Returns:
            The unregistered observer.
        """
        if observer not in self.event_observers[event]:
            if not no_raise:
                raise RuntimeError(
                    f"Failed to remove event observer: '{observer.__qualname__}' is not registered for event '{event}'"
                )
            return None

        self.event_observers[event].remove(observer)
        return observer

    def clear(self) -> None:
        """Resets all registered events and event observers."""
        len_obs = len(self.event_observers)
        self.event_observers: dict[str, list[Callable]] = defaultdict(list)
        _logger.debug("Reset observers (were: %d)" % len_obs)


def observes(event: str) -> Callable[[Callable[[AbstractContext], None]], Callable[[AbstractContext], None]]:
    """Decorator which registers a function as an observer for the specified event.

    The observer callable must accept an AbstractContext instance as its only argument.

    Args:
        event: Event name.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable[[AbstractContext], None]) -> Callable[[AbstractContext], None]:
        EventSystem.instance().register_observer(event, func)

        return func

    return decorator

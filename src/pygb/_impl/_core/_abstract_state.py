from abc import ABC, abstractmethod
from typing import Generic

from pygb._impl._core._abstract_context import ContextType
from pygb._impl._core._events import EventSystem


class AbstractState(ABC, Generic[ContextType]):
    """Abstract state class."""

    def __init__(self, context: ContextType, event_system: EventSystem | None = None, name: str | None = None) -> None:
        """Constructor.

        Args:
            context: The context which represents the central data store.
            event_system: Event system instance. Defaults to None.
            name: Name. If set to None, the class name is set as a default value. Defaults to None.
        """
        self.name = name or self.__class__.__qualname__
        self.context = context
        self.events = event_system

    @abstractmethod
    def __call__(self) -> str | None:
        """Execute the state. Must return a state transition string or None.

        Note: Returning None marks this state as the final state. The state machine stops after receiving None as a
        transition."""
        ...

    def transitions(self) -> list[str]:
        """Return all tranistion names that can be returned by this state's __call__() method.

        This is required for plotting the state machine graph.

        Returns:
            List of transition names. Empty list as a default value.
        """
        return []

    def __eq__(self, __o: object) -> bool:
        """Checks two objects for equality. Note that two states are considered equal if their names are equal.

        Args:
            __o: Other object.

        Returns:
            Whether or not the objects are equal.
        """
        if not isinstance(__o, type(self)):
            return False

        return self.name == __o.name

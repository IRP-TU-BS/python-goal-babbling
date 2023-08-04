from abc import ABC, abstractmethod
from typing import TypeVar


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

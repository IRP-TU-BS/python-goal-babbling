from abc import ABC, abstractmethod, abstractproperty
from typing import TypeVar


class AbstractContext(ABC):
    """Abstract (state machine) context class."""

    @abstractproperty
    def num_epoch_sets(self) -> int:
        """Returns the number of epoch sets per training

        Returns:
            Number of epoch sets.
        """

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

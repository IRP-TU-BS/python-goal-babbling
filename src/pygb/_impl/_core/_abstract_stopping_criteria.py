from abc import ABC, abstractmethod
from typing import Generic

from pygb._impl._core._abstract_context import ContextType


class AbstractStoppingCriteria(ABC, Generic[ContextType]):
    @abstractmethod
    def fulfilled(self, context: ContextType) -> bool:
        """Check if the stopping criteria is fulfilled.

        Args:
            context: Goal Babbling context.

        Returns:
            Whether the stopping criteria is fulfilled.
        """

    @abstractmethod
    def __eq__(self, __o: object) -> bool:
        """Equality check.

        Args:
            __o: Other object.

        Returns:
            True if both elements are equal.
        """

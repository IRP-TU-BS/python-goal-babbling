from abc import ABC, abstractmethod
from typing import Generic

import numpy as np

from pygb._impl._core._context import ContextType


class AbstractGoalSelector(ABC, Generic[ContextType]):
    @abstractmethod
    def select(self, context: ContextType) -> tuple[int, np.ndarray]:
        """Select the target global goal for the next training sequence.

        Args:
            context: Goal Babbling context.

        Returns:
            Target global goal as a tuple (index, goal).
        """


class AbstractLocalGoalGenerator(ABC, Generic[ContextType]):
    @abstractmethod
    def generate(self, context: ContextType) -> np.ndarray:
        """Generate a local goal between the global start and target goals.

        Args:
            context: Goal Babbling context.

        Returns:
            Local goal.
        """


class AbstractWeightGenerator(ABC, Generic[ContextType]):
    @abstractmethod
    def generate(self, context: ContextType) -> float:
        """Calculate the weight for the next training sample.

        Args:
            context: Goal Babbling context.

        Returns:
            Training sample weight.
        """

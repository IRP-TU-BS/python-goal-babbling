from abc import ABC, abstractmethod
from typing import Generic

import numpy as np

from pygb._impl._core._context import ContextType


class AbstractGoalSelector(ABC, Generic[ContextType]):
    def __init__(self, context: ContextType) -> None:
        self.context = context

    @abstractmethod
    def select(self) -> np.ndarray:
        """Select the target global goal for the next training sequence.

        Returns:
            Target global goal.
        """


class AbstractLocalGoalGenerator(ABC, Generic[ContextType]):
    def __init__(self, context: ContextType) -> None:
        self.context = context

    @abstractmethod
    def generate(self) -> np.ndarray:
        """Generate a local goal between the global start and target goals.

        Returns:
            Local goal.
        """


class AbstractWeightGenerator(ABC, Generic[ContextType]):
    def __init__(self, context: ContextType) -> None:
        self.context = context

    @abstractmethod
    def generate(self) -> float:
        """Calculate the weight for the next training sample.

        Returns:
            Training sample weight.
        """

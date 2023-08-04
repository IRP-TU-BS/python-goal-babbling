from abc import ABC, abstractmethod
from typing import Generic

import numpy as np

from pygb._impl._core._abstract_context import ContextType


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
    def generate(
        self, start_goal: np.ndarray, stop_goal: np.ndarray, len_sequence: int, context: ContextType | None = None
    ) -> list[np.ndarray]:
        """Generate local goals between the global start and target goals for a sequence.

        Args:
            start_goal: (Global) Start goal.
            stopp_goal: (Global) Stop goal.
            len_sequence: Number of observations per sequence.
            context: Goal Babbling context. Defaults to None.

        Returns:
            List of local goals between two global goals.
        """


class AbstractNoiseGenerator(ABC, Generic[ContextType]):
    @abstractmethod
    def generate(self, observation: np.ndarray, context: ContextType | None = None) -> np.ndarray:
        """Generate noise.

        Args:
            observation: Observation, e.g. a local goal.
            context: Goal Babbling context. Defaults to None.

        Returns:
            Noise vector, which must match the action shape.
        """

    @abstractmethod
    def update(self) -> None:
        """Update the noise generator."""


class AbstractWeightGenerator(ABC, Generic[ContextType]):
    @abstractmethod
    def generate(self, context: ContextType) -> float:
        """Calculate the weight for the next training sample.

        Args:
            context: Goal Babbling context.

        Returns:
            Training sample weight.
        """

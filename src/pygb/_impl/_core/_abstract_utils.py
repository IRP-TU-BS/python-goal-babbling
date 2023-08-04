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


class AbstractSequenceGenerator(ABC, Generic[ContextType]):
    @abstractmethod
    def generate(
        self, start: np.ndarray, stop: np.ndarray, len_sequence: int, context: ContextType | None = None
    ) -> list[np.ndarray]:
        """Generate a sequence of observations or actions between start and stop observations/actions.

        Note: The kind of sequence (e.g. ObservationSequenc or ActionSequence) depends on the concrete implementation.

        Args:
            start_goal: Sequence start.
            stopp_goal: SequenceStop.
            len_sequence: Number of observations/actions per sequence.
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

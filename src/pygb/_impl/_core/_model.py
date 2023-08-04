from abc import ABC, abstractmethod

import numpy as np


class AbstractForwardModel(ABC):
    """Forward model, which implements the forward calculation f(action) = observation."""

    @abstractmethod
    def forward(self, action: np.ndarray) -> np.ndarray:
        """Calculate the outcome of the specified action f(action) = observation.

        Args:
            action: 1D action vector.

        Returns:
            The 1D observation which can be observerd after executing the action.
        """

    @abstractmethod
    def forward_batch(self, action_batch: np.ndarray) -> np.ndarray:
        """Calculate the outcomes of a batch of actions.

        Args:
            action_batch: 2D batch of actions.

        Returns:
            2D batch of observations.
        """


class AbstractInverseEstimator(ABC):
    """Inverse model, which wraps an estimator (e.g. a neural network) for the reverse calculation
    g(observation) = action."""

    @abstractmethod
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Estimate the necessary action vector to produce the specified observation.

        Args:
            observation: 1D observation vector.

        Returns:
            1D action estimate.
        """

    @abstractmethod
    def predict_batch(self, observation_batch: np.ndarray) -> np.ndarray:
        """Estimate the required actions for a batch of observations.

        Args:
            observation_batch: 2D observation batch.

        Returns:
            2D batch of action estimates.
        """

    @abstractmethod
    def fit(self, observation: np.ndarray, action: np.ndarray) -> float:
        """Train the estimate incrementally using one observation/action pair.

        The observation o represents the input, the action a represents the output.

        Args:
            observation: 1D observation vector.
            action: 1D action vector.

        Returns:
            The estimator's prediction error on the sample (o, a) after fitting.
        """

    @abstractmethod
    def fit_batch(self, observation_batch: np.ndarray, action_batch: np.ndarray) -> float:
        """Train the estimate incrementally on a batch of N observation/action samples.

        Args:
            observation_batch: Observations of shape (N, dim(o)).
            action_batch: Actions of shape (N, dim(a)).

        Returns:
            The estimator's mean prediction error on the training data after fitting.
        """
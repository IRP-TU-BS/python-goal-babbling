import numpy as np
from numpy.random import Generator

from pygb._impl._core._abstract_utils import AbstractNoiseGenerator
from pygb._impl._core._context import GoalBabblingContext


class GBNoiseGenerator(AbstractNoiseGenerator[GoalBabblingContext]):
    """Variation generator from the original Goal Babbling paper (Rolf, 2010)."""

    def __init__(self, context: GoalBabblingContext, rng: Generator = np.random.default_rng()) -> None:
        """Constructor.

        Creates the A matrix of shape (dim_act, dim_obs) and b vector of shape (1, dim_act).

        Args:
            context: Goal Babbling context.
            rng: Numpy random number generator. Defaults to a randomly initialized RNG.
        """
        self._rng = rng

        dim_act = context.current_parameters.dim_act
        dim_obs = context.current_parameters.dim_obs

        self.sigma = np.ones(shape=(1, dim_act)) * context.current_parameters.sigma
        self.sigma_delta = self.sigma * context.current_parameters.sigma_delta

        self.A = self.sigma.T * self._rng.standard_normal(size=(dim_act, dim_obs))
        self.b = self.sigma.T * self._rng.standard_normal(size=(dim_act, 1))

    def generate(self, observation: np.ndarray, context: GoalBabblingContext | None = None) -> np.ndarray:
        """Generates noise of shape (dim_act,).

        Args:
            observation: Observation vector of shape (1, dim_obs).
            context: Goal Babbling context. Defaults to None.

        Returns:
            Noise vector of shape (dim_act,).
        """
        return (self.A @ observation + self.b.T).squeeze()

    def update(self) -> None:
        """Updates the A matrix and b vector."""
        n_v = np.sqrt(self.sigma**2 / (self.sigma**2 + self.sigma_delta**2))

        self.A = n_v.T * (self.A + self.sigma_delta.T * self._rng.standard_normal(size=self.A.shape))
        self.b = n_v.T * (self.b + self.sigma_delta.T * self._rng.standard_normal(size=self.b.shape))

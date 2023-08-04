import numpy as np

from pygb._impl._core._abstract_utils import AbstractNoiseGenerator
from pygb._impl._core._context import GoalBabblingContext


class GBNoiseGenerator(AbstractNoiseGenerator[GoalBabblingContext]):
    """Variation generator from the original Goal Babbling paper (Rolf, 2010)."""

    def __init__(self, context: GoalBabblingContext, random_seed: int | None = None) -> None:
        """Constructor.

        Creates the A matrix of shape (dim_act, dim_obs) and b vector of shape (1, dim_act).

        Args:
            context: Goal Babbling context.
            random_seed: Random seed used to initialize a numpy random number generator. Defaults to None.
        """
        self._rng = np.random.default_rng(seed=random_seed)

        dim_act = context.current_parameters.dim_act
        dim_obs = context.current_parameters.dim_obs

        self.sigma = np.ones(shape=(1, dim_act)) * context.current_parameters.sigma
        self.sigma_delta = self.sigma * context.current_parameters.sigma_delta

        self.A = self.sigma.T * self._rng.standard_normal(size=(dim_act, dim_obs))
        self.b = self.sigma.T * self._rng.standard_normal(size=(dim_act, 1))

    def generate(self, observation: np.ndarray, context: GoalBabblingContext | None = None) -> np.ndarray:
        """Generates noise of shape (1, dim_act).

        Args:
            observation: Observation vector of shape (1, dim_obs).
            context: Goal Babbling context. Defaults to None.

        Returns:
            Noise vector of shape (1, dim_act).
        """
        return self.A @ observation + self.b.T

    def update(self) -> None:
        """Updates the A matrix and b vector."""
        n_v = np.sqrt(self.sigma**2 / (self.sigma**2 + self.sigma_delta**2))

        self.A = n_v.T * (self.A + self.sigma_delta.T * self._rng.standard_normal(size=self.A.shape))
        self.b = n_v.T * (self.b + self.sigma_delta.T * self._rng.standard_normal(size=self.b.shape))

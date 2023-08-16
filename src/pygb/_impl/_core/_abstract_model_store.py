from abc import ABC, abstractmethod

from pygb._impl._core._abstract_context import ContextType
from pygb._impl._core._model import AbstractInverseEstimator


class AbstractModelStore(ABC):
    """Model store interface. Defines functionality for a model store which stores inverse estimate instances per epoch
    set indices.

    Note that the implementation of this abstract class must support identifying previously trained estimates via the
    epoch set index in which it was trained."""

    @abstractmethod
    def conditional_save(self, estimate: AbstractInverseEstimator, epoch_set_index: int, context: ContextType) -> bool:
        """Stores an estimate for the specifies epoch set index.

        Note that this function must decide on its own (e.g. based on the context attribute) whether or not to store the
        estimate.

        Args:
            estimate: (Trained) Inverse estimate instance.
            epoch_set_index: Index of the epoch set in which the estimate was trained.
            context: Goal Babbling context.

        Returns:
            True if the estimate was saved, False if it was not saved (e.g. if the recorded performance was worse than
            the previously saved estimate).
        """

    @abstractmethod
    def load(self, epoch_set_index: int) -> AbstractInverseEstimator:
        """Loads an inverse estimate instance. The epoch set index specifies the epoch set in which the estimate was
        trained.

        Args:
            epoch_set_index: Epoch set index in which the inverse estimate was trained.

        Returns:
            Loaded inverse estimate instance.
        """

from abc import ABC, abstractmethod

from pygb._impl._core._abstract_context import ContextType
from pygb._impl._core._model import AbstractInverseEstimate


class AbstractEstimateCache(ABC):
    """Model store interface. Defines functionality for a model store which stores inverse estimate instances per epoch
    set indices.

    Make sure that
        1) the implementation of this abstract class supports identifying previously trained estimates via the
            epoch set index in which it was trained
        2) the implementation of this class stores a model for each epoch set

    You can ignore both points in case you only train singular epoch sets. Simply store and load the estimate if it
    exceeds the previous metric.
    """

    @abstractmethod
    def conditional_save(self, estimate: AbstractInverseEstimate, epoch_set_index: int, context: ContextType) -> bool:
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
    def load(self, epoch_set_index: int) -> AbstractInverseEstimate:
        """Loads an inverse estimate instance. The epoch set index specifies the epoch set in which the estimate was
        trained.

        Args:
            epoch_set_index: Epoch set index in which the inverse estimate was trained.

        Returns:
            Loaded inverse estimate instance.
        """

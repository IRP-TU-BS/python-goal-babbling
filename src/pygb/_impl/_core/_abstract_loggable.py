from abc import ABC
from typing import Any


class AbstractLoggable(ABC):
    """Interface which defines methods for classes which need to log data during training."""

    def parameters(self) -> dict[str, Any]:
        """Method which is used to log this instance's fixed parameters at the start of an epoch set. Must return an
        empty dictionary if there aren't any.

        Returns:
            The instances parameters in form of a dictionary.
        """
        return {}

    def metrics(self) -> dict[str, Any]:
        """Method which is used to log the instance's time-dependent metrics, e.g. a loss over the course of an epoch
        set. Must return an empty dictionary if there are no loggable metrics.

        Returns:
            The instance's metrics in form of a dictionary.
        """
        return {}

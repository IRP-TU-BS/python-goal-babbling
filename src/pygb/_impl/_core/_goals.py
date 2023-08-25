from typing import Any

import numpy as np

from pygb._impl._core._abstract_loggable import AbstractLoggable


class GoalSet(AbstractLoggable):
    """Class which represents one goal set. A goal set represents all global goals which are necessary for one epoch
    set.
    """

    def __init__(
        self,
        train: np.ndarray,
        test: np.ndarray,
        optional_test: dict[str, np.ndarray] | None = None,
        name: str | None = None,
    ) -> None:
        """Constructor.

        Args:
            train: Training goals. These are used to update the estimate.
            test: Test goals. The estimate's performance is measured against these.
            optional_test: Dictionary of optional test sets. While the estimate's performance is measured on these
                goals, they are not used internally, e.g. for goal selection. Defaults to None.
            name: Optional goal set name. Defaults to None.
        """
        self.train = train
        self.test = test
        self.optional_test = optional_test
        self.name = name

    def __eq__(self, __o: object) -> bool:
        """Checks two objects for equality.

        Args:
            __o: Other object.

        Returns:
            Whether or not the objects are equal.
        """
        if not isinstance(__o, type(self)):
            return False

        for goals, other_goals in zip((self.train, self.test), (__o.train, __o.test)):
            if not np.all(goals == other_goals):
                return False

        if (self.optional_test is None and __o.optional_test is not None) or (
            self.optional_test is not None and __o.optional_test is None
        ):
            return False

        if self.optional_test is not None and __o.optional_test is not None:
            if sorted(self.optional_test.keys()) != sorted(__o.optional_test.keys()):
                return False

            for key, goals in self.optional_test.items():
                if not np.all(goals == __o.optional_test[key]):
                    return False

        return True

    def parameters(self) -> dict[str, Any]:
        """Returns the goal set sizes (train, test and optional) as a dictionary.

        Returns:
            Goal set sizes.
        """
        params = {"dataset_name": self.name or "-", "train_size": self.train.shape[0], "test_size": self.test.shape[0]}

        if self.optional_test is not None:
            params.update({f"{key}_size": goal_set.shape[0] for key, goal_set in self.optional_test.items()})

        return params


class GoalStore:
    """Goal store class, which contains multiple goal sets, indexed by epoch set indices."""

    def __init__(self, goals_sets: GoalSet | list[GoalSet]) -> None:
        """Constructor.

        Args:
            goals_sets: List of goal sets. The index corresponds with the epoch set index.
        """
        self.goal_sets = goals_sets if isinstance(goals_sets, list) else [goals_sets]

    def __getitem__(self, epoch_set_index: int) -> GoalSet:
        """Returns the specified epoch set's goal set.

        Args:
            epoch_set_index: Epoch set index.

        Returns:
            The epoch set's goal set.
        """
        return self.goal_sets[epoch_set_index]

    def __len__(self) -> int:
        """Implements the builtin 'len' functionality.

        Returns:
            Number of parameter sets.
        """
        return len(self.goal_sets)

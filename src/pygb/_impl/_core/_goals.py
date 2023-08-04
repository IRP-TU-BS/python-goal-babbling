import numpy as np


class GoalSet:
    def __init__(self, train: np.ndarray, test: np.ndarray, optional_test: dict[str, np.ndarray] | None = None) -> None:
        self.train = train
        self.test = test
        self.optional_test = optional_test

    def __eq__(self, __o: object) -> bool:
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


class GoalStore:
    def __init__(self, goals_sets: GoalSet | list[GoalSet]) -> None:
        self.goal_sets = goals_sets if isinstance(goals_sets, list) else [goals_sets]

    def __getitem__(self, epoch_set_index: int) -> GoalSet:
        return self.goal_sets[epoch_set_index]

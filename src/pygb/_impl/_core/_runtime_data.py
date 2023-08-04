from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np


@dataclass
class ObservationSequence:
    start_glob_goal_idx: int
    stop_glob_goal_idx: int
    weights: list[float] = field(default_factory=list)
    local_goals: list[np.ndarray] = field(default_factory=list)
    predicted_actions: list[np.ndarray] = field(default_factory=list)
    observations: list[np.ndarray] = field(default_factory=list)


@dataclass
class RuntimeData:
    current_sequence: ObservationSequence | None = None
    previous_sequence: ObservationSequence | None = None
    performance_error: float | None = None  # error on test goals after last completed epoch
    opt_performance_errors: dict[str, float] = field(
        default_factory=dict
    )  # error on optional test goals calculated after each epoch
    observation_index: int = 0  # how far are we within the sequence?
    sequence_index: int = 0  # current sequence index (i.e. how far into the epoch are we?)
    epoch_index: int = 0  # current epoch index (i.e. how far into the epoch set are we?)
    epoch_set_index: int = 0  # current epoch set (i.e. how far into the training are we?)
    sequences: list[ObservationSequence] = field(default_factory=list)  # list of COMPLETED (i.e. previous) sequences
    train_goal_error: list[float] = field(
        default_factory=list
    )  # training goal error, calculated pre goal after a completed sequence
    train_goal_visit_count: list[int] = field(default_factory=list)  # visit count per goal

    def update_current_sequence(self, sequence: ObservationSequence) -> None:
        if self.current_sequence is not None:
            self.previous_sequence = self.current_sequence

        self.current_sequence = sequence

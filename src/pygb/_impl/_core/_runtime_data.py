from dataclasses import dataclass, field

import numpy as np


@dataclass
class SequenceData:
    start_glob_goal_idx: int
    stop_glob_goal_idx: int
    weights: list[float] = field(default_factory=list)
    local_goals: list[np.ndarray] = field(default_factory=list)
    predicted_actions: list[np.ndarray] = field(default_factory=list)
    predicted_local_goals: list[np.ndarray] = field(default_factory=list)


@dataclass
class RuntimeData:
    current_sequence: SequenceData | None = None
    performance_error: float | None = None  # error on test goals after last completed epoch
    sequence_index: int = 0  # current sequence index (i.e. how far into the epoch are we?)
    epoch_index: int = 0  # current epoch index (i.e. how far into the epoch set are we?)
    epoch_set_index: int = 0  # current epoch set (i.e. how far into the training are we?)
    sequences: list[SequenceData] = field(default_factory=list)  # list of COMPLETED (i.e. previous) sequences
    train_goal_error: list[float] = field(default_factory=list)  # error on training goals after last completed epoch
    train_goal_visit_count: list[int] = field(default_factory=list)  # visit count per goal

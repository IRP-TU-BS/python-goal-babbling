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
    performance_error: float = -1.0
    sequence_index: int = 0
    epoch_index: int = 0
    epoch_set_index: int = 0
    sequences: list[SequenceData] = field(default_factory=list)
    train_goal_error: list[float] = field(default_factory=list)

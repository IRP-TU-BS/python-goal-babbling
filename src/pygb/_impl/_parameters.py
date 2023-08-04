from copy import deepcopy
from dataclasses import dataclass

import numpy as np


@dataclass
class GBParameters:
    sigma: float
    sigma_delta: float
    dim_act: int
    dim_obs: int
    len_sequence: int
    len_epoch: int
    epoch_sets: int
    home_action_set: np.ndarray

    def combine(self, increment: "GBParameterIncrement") -> "GBParameters":
        combined = deepcopy(self)
        for attribute_name in increment.__match_args__:
            if (increment_value := increment.__getattribute__(attribute_name)) is not None:
                combined.__setattr__(attribute_name, increment_value)

        return combined


@dataclass
class GBParameterIncrement:
    sigma: float | None = None
    sigma_delta: float | None = None
    dim_act: int | None = None
    dim_obs: int | None = None
    len_sequence: int | None = None
    len_epoch: int | None = None
    epoch_sets: int | None = None
    home_action_set: np.ndarray | None = None

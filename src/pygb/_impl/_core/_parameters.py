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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False

        for attribute_name in self.__match_args__:
            attribute = self.__getattribute__(attribute_name)
            other_attribute = other.__getattribute__(attribute_name)

            if isinstance(attribute, np.ndarray):
                if not np.all(attribute == other_attribute):
                    return False
            else:
                if attribute != other_attribute:
                    return False

        return True

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


class GBParameterStore:
    def __init__(self, gb_parameters: GBParameters | list[GBParameters | GBParameterIncrement]) -> None:
        if isinstance(gb_parameters, list):
            self.gb_parameters = self._combine_parameter_sets(gb_parameters)
        else:
            self.gb_parameters = [gb_parameters]

    def __getitem__(self, epoch_index: int) -> GBParameters:
        return self.gb_parameters[epoch_index]

    def _combine_parameter_sets(self, gb_parameters: list[GBParameters | GBParameterIncrement]) -> list[GBParameters]:
        previous_default = gb_parameters[0]
        if not isinstance(previous_default, GBParameters):
            raise ValueError(
                f"""Failed to extract GB parameter list: First entry must be of type {GBParameters.__name__} """
                f"""but is of type {type(previous_default)}"""
            )

        params = [previous_default]
        if len(gb_parameters) == 1:
            return params

        for index, p in enumerate(gb_parameters):
            if index == 0:
                continue

            if isinstance(p, GBParameterIncrement):
                params.append(previous_default.combine(p))
            else:
                params.append(p)
                previous_default = p

        return params

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from pygb._impl._core._abstract_stopping_criteria import AbstractStoppingCriteria


@dataclass
class GBParameters:
    """Goal Babbling parameters which are valid for one Goal Babbling epoch set."""

    sigma: float
    sigma_delta: float
    dim_act: int
    dim_obs: int
    len_sequence: int
    len_epoch: int
    len_epoch_set: int
    epoch_sets: int
    home_action: np.ndarray
    home_observation: np.ndarray
    stopping_criteria: list[AbstractStoppingCriteria] = field(default_factory=list)

    def __eq__(self, other: object) -> bool:
        """Checks two parameter sets for equality.

        Args:
            other: Other object.

        Returns:
            Whether or not the two ojects are equal.
        """
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
        """Combine this parameter set with the specified increment.

        This parameter set's parameters are updated with values from the increment. Increment values which are set to
        None are kept.

        Args:
            increment: Parameter increment.

        Returns:
            The combination of the current parameter set and the specified increment.
        """
        combined = deepcopy(self)
        for attribute_name in increment.__match_args__:
            if (increment_value := increment.__getattribute__(attribute_name)) is not None:
                combined.__setattr__(attribute_name, increment_value)

        return combined


@dataclass
class GBParameterIncrement:
    """Goal Babbling Parameter increment."""

    sigma: float | None = None
    sigma_delta: float | None = None
    dim_act: int | None = None
    dim_obs: int | None = None
    len_sequence: int | None = None
    len_epoch: int | None = None
    epoch_sets: int | None = None
    len_epoch_set: int | None = None
    home_action: np.ndarray | None = None
    home_observation: np.ndarray | None = None
    stopping_criteria: list[AbstractStoppingCriteria] | None = None


class GBParameterStore:
    """Goal Babbling parameter store. Contains parameter sets indexed by epoch set indices."""

    def __init__(self, gb_parameters: GBParameters | list[GBParameters | GBParameterIncrement]) -> None:
        """Constructor.

        If a list is provided as an argument and the list contains increment instances, they are combined with the
        parameter set from the previous epoch set.

        Args:
            gb_parameters: One ore multiple parameter sets or increments.
        """
        if isinstance(gb_parameters, list):
            self.gb_parameters = self._combine_parameter_sets(gb_parameters)
        else:
            self.gb_parameters = [gb_parameters]

    def __getitem__(self, epoch_index: int) -> GBParameters:
        """Returns the epoch sets's paramter set.

        Args:
            epoch_index: Epoch set index.

        Returns:
            Goal Babbling parameters.
        """
        return self.gb_parameters[epoch_index]

    def _combine_parameter_sets(self, gb_parameters: list[GBParameters | GBParameterIncrement]) -> list[GBParameters]:
        if not isinstance(gb_parameters[0], GBParameters):
            raise ValueError(
                f"""Failed to extract GB parameter list: First entry must be of type {GBParameters.__name__} """
                f"""but is of type {type(gb_parameters[0])}"""
            )

        params = [gb_parameters[0]]
        if len(gb_parameters) == 1:
            return params

        for index, p in enumerate(gb_parameters):
            if index == 0:
                continue

            if isinstance(p, GBParameterIncrement):
                params.append(params[-1].combine(p))
            else:
                params.append(p)

        return params

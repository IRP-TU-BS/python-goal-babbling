from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pygb import GBParameterIncrement, GBParameters


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("sigma", 42.0),
        ("sigma_delta", 42.0),
        ("dim_act", 42),
        ("dim_obs", 42),
        ("len_sequence", 42),
        ("len_epoch", 42),
        ("epoch_sets", 42),
    ],
)
def test_combine_parameters_with_increment_default_types(name: str, value: Any) -> None:
    parameters = GBParameters(0.1, 0.01, 3, 2, 10, 20, 1, np.array([1.0, 2.0, 3.0]))
    increment = GBParameterIncrement()
    increment.__setattr__(name, value)

    combined = parameters.combine(increment)

    assert combined.__getattribute__(name) == value

    for attribute_name in combined.__match_args__:
        attribute = combined.__getattribute__(attribute_name)

        if not isinstance(attribute, np.ndarray):
            assert combined.__getattribute__(attribute_name) is not None


def test_combine_parameters_with_increment_home_action_sequence() -> None:
    parameters = GBParameters(0.1, 0.01, 3, 2, 10, 20, 1, np.array([1.0, 2.0, 3.0]))
    increment = GBParameterIncrement(home_action_set=np.array([42.0, 43.0, 44.0]))

    combined = parameters.combine(increment)

    for attribute_name in ("sigma", "sigma_delta", "dim_act", "dim_obs", "len_sequence", "len_epoch", "epoch_sets"):
        assert combined.__getattribute__(attribute_name) is not None

    assert_array_equal(combined.home_action_set, increment.home_action_set)

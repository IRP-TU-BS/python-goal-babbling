from typing import Any

import pytest

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
def test_combine_parameters_with_increment(name: str, value: Any) -> None:
    parameters = GBParameters(0.1, 0.01, 3, 2, 10, 20, 1)
    increment = GBParameterIncrement()
    increment.__setattr__(name, value)

    parameters.update(increment)

    assert parameters.__getattribute__(name) == value

    for attribute_name in parameters.__match_args__:
        assert parameters.__getattribute__(attribute_name) is not None

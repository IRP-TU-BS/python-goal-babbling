from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pygb import GBParameterIncrement, GBParameters, GBParameterStore


def test_gb_parameters_eq() -> None:
    assert GBParameters(1.0, 2.0, 3, 4, 5, 6, 7, np.array([1.0])) == GBParameters(
        1.0, 2.0, 3, 4, 5, 6, 7, np.array([1.0])
    )
    assert not GBParameters(1.0, 2.0, 3, 4, 5, 6, 7, np.array([1.0])) == GBParameters(
        1.0, 42.0, 3, 4, 5, 6, 7, np.array([1.0])
    )


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


@pytest.mark.parametrize(
    ("gb_parameters", "parsed_list"),
    (
        (
            [
                GBParameters(0.1, 0.01, 1, 2, 10, 20, 1, np.array([1.0])),
                GBParameters(10.0, 20.0, 30, 20, 100, 200, 10, np.array([1.0])),
            ],
            [
                GBParameters(0.1, 0.01, 1, 2, 10, 20, 1, np.array([1.0])),
                GBParameters(10.0, 20.0, 30, 20, 100, 200, 10, np.array([1.0])),
            ],
        ),
        (
            [
                GBParameters(0.1, 0.01, 1, 2, 10, 20, 1, np.array([1.0])),
                GBParameters(10.0, 20.0, 30, 20, 100, 200, 10, np.array([1.0])),
                GBParameterIncrement(sigma=42.0),
            ],
            [
                GBParameters(0.1, 0.01, 1, 2, 10, 20, 1, np.array([1.0])),
                GBParameters(10.0, 20.0, 30, 20, 100, 200, 10, np.array([1.0])),
                GBParameters(42.0, 20.0, 30, 20, 100, 200, 10, np.array([1.0])),
            ],
        ),
        (
            GBParameters(0.1, 0.01, 1, 2, 10, 20, 1, np.array([1.0])),
            [GBParameters(0.1, 0.01, 1, 2, 10, 20, 1, np.array([1.0]))],
        ),
    ),
)
def test_parameter_store_combine_parameter_sets(
    gb_parameters: GBParameters | list[GBParameters | GBParameterIncrement], parsed_list: list[GBParameters]
) -> None:
    store = GBParameterStore(gb_parameters)

    assert store.gb_parameters == parsed_list


def test_parameter_store_raises_if_first_parameter_set_is_increment() -> None:
    with pytest.raises(ValueError):
        GBParameterStore([GBParameterIncrement(sigma=42.0)])

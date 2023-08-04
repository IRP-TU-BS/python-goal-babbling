from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pygb import (
    GBParameterIncrement,
    GBParameters,
    GBParameterStore,
    GoalBabblingContext,
)
from pygb.interfaces import AbstractStoppingCriteria


class DummyStoppingCriteria(AbstractStoppingCriteria[GoalBabblingContext]):
    def __init__(self, parameter: int) -> None:
        self.parameter = parameter

    def fulfilled(self, context: GoalBabblingContext) -> bool:
        return super().fulfilled(context)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, type(self)):
            return False

        return self.parameter == __o.parameter


def test_gb_parameters_eq() -> None:
    params = GBParameters(
        sigma=1.0,
        sigma_delta=2.0,
        dim_act=3,
        dim_obs=4,
        len_sequence=5,
        len_epoch=6,
        len_epoch_set=7,
        go_home_chance=0.1,
        home_action=np.array([1.0]),
        home_observation=np.array([20.0]),
        stopping_criteria=[DummyStoppingCriteria(parameter=42)],
    )
    assert params == params

    params2 = GBParameters(
        sigma=1.0,
        sigma_delta=2.0,
        dim_act=3,
        dim_obs=4,
        len_sequence=5,
        len_epoch=6,
        len_epoch_set=7,
        go_home_chance=0.1,
        home_action=np.array([42.0]),  # different
        home_observation=np.array([20.0]),
        stopping_criteria=[DummyStoppingCriteria(parameter=42)],
    )

    assert params != params2


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("sigma", 42.0),
        ("sigma_delta", 42.0),
        ("dim_act", 42),
        ("dim_obs", 42),
        ("len_sequence", 42),
        ("len_epoch", 42),
        ("len_epoch_set", 42),
        ("go_home_chance", 42.42),
    ],
)
def test_combine_parameters_with_increment_default_types(name: str, value: Any) -> None:
    parameters = GBParameters(
        sigma=1.0,
        sigma_delta=0.01,
        dim_act=3,
        dim_obs=2,
        len_sequence=10,
        len_epoch=20,
        len_epoch_set=10,
        go_home_chance=0.1,
        home_action=np.array([1.0, 2.0, 3.0]),
        home_observation=np.array([10.0]),
    )
    increment = GBParameterIncrement()
    increment.__setattr__(name, value)

    combined = parameters.combine(increment)

    assert combined.__getattribute__(name) == value

    for attribute_name in combined.__match_args__:
        attribute = combined.__getattribute__(attribute_name)

        if not isinstance(attribute, np.ndarray):
            assert combined.__getattribute__(attribute_name) is not None


def test_combine_parameters_with_increment_home_action_sequence() -> None:
    parameters = GBParameters(
        sigma=1.0,
        sigma_delta=0.01,
        dim_act=3,
        dim_obs=2,
        len_sequence=10,
        len_epoch=20,
        len_epoch_set=10,
        go_home_chance=0.1,
        home_action=np.array([1.0, 2.0, 3.0]),
        home_observation=np.array([10.0]),
    )
    increment_action = GBParameterIncrement(home_action=np.array([42.0, 43.0, 44.0]))

    combined = parameters.combine(increment_action)

    for attribute_name in (
        "sigma",
        "sigma_delta",
        "dim_act",
        "dim_obs",
        "len_sequence",
        "len_epoch",
        "home_observation",
    ):
        assert combined.__getattribute__(attribute_name) is not None

    assert_array_equal(combined.home_action, increment_action.home_action)

    increment_observation = GBParameterIncrement(home_observation=np.array([20.0]))

    combined = parameters.combine(increment_observation)

    for attribute_name in (
        "sigma",
        "sigma_delta",
        "dim_act",
        "dim_obs",
        "len_sequence",
        "len_epoch",
        "home_action",
    ):
        assert combined.__getattribute__(attribute_name) is not None

    assert_array_equal(combined.home_observation, increment_observation.home_observation)


@pytest.mark.parametrize(
    ("gb_parameters", "parsed_list"),
    (
        (
            [
                GBParameters(0.1, 0.01, 1, 2, 10, 20, 10, 0.1, np.array([1.0]), np.array([10.0])),
                GBParameters(10.0, 20.0, 30, 20, 100, 200, 10, 0.1, np.array([1.0]), np.array([10.0])),
            ],
            [
                GBParameters(0.1, 0.01, 1, 2, 10, 20, 10, 0.1, np.array([1.0]), np.array([10.0])),
                GBParameters(10.0, 20.0, 30, 20, 100, 200, 10, 0.1, np.array([1.0]), np.array([10.0])),
            ],
        ),
        (
            [
                GBParameters(0.1, 0.01, 1, 2, 10, 20, 10, 0.1, np.array([1.0]), np.array([10.0])),
                GBParameters(10.0, 20.0, 30, 20, 100, 200, 10, 0.1, np.array([1.0]), np.array([10.0])),
                GBParameterIncrement(sigma=42.0, stopping_criteria=[DummyStoppingCriteria(42)]),
            ],
            [
                GBParameters(0.1, 0.01, 1, 2, 10, 20, 10, 0.1, np.array([1.0]), np.array([10.0])),
                GBParameters(10.0, 20.0, 30, 20, 100, 200, 10, 0.1, np.array([1.0]), np.array([10.0])),
                GBParameters(
                    42.0,
                    20.0,
                    30,
                    20,
                    100,
                    200,
                    10,
                    0.1,
                    np.array([1.0]),
                    np.array([10.0]),
                    stopping_criteria=[DummyStoppingCriteria(42)],
                ),
            ],
        ),
        (
            GBParameters(0.1, 0.01, 1, 2, 10, 20, 10, 0.1, np.array([1.0]), np.array([10.0])),
            [GBParameters(0.1, 0.01, 1, 2, 10, 20, 10, 0.1, np.array([1.0]), np.array([10.0]))],
        ),
        (
            [
                GBParameters(0.1, 0.01, 1, 2, 10, 20, 10, 0.1, np.array([1.0]), np.array([10.0])),
                GBParameterIncrement(sigma=42.0),
                GBParameterIncrement(sigma_delta=42.42),
            ],
            [
                GBParameters(0.1, 0.01, 1, 2, 10, 20, 10, 0.1, np.array([1.0]), np.array([10.0])),
                GBParameters(42.0, 0.01, 1, 2, 10, 20, 10, 0.1, np.array([1.0]), np.array([10.0])),
                GBParameters(42.0, 42.42, 1, 2, 10, 20, 10, 0.1, np.array([1.0]), np.array([10.0])),
            ],
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

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np

from pygb import GBHomeWeightGenerator, GoalBabblingContext, RuntimeData


@patch("pygb._impl._gb_utils._gb_home_weight_generator.GBHomeWeightGenerator._calc_weights")
@patch("pygb._impl._gb_utils._gb_home_weight_generator.GBHomeWeightGenerator._choose_previous_data")
def test_generate(choose_previous_data_mock: MagicMock, calc_weights_mock: MagicMock) -> None:
    calc_weights_mock.return_value = (2.0, 4.0)
    choose_previous_data_mock.return_value = (np.array([1.0]), np.array([2.0]), np.array([3.0]))
    context_mock = MagicMock(spec=GoalBabblingContext, runtime_data=MagicMock(spec=RuntimeData))

    generator = GBHomeWeightGenerator()

    assert generator.generate(context_mock) == 4.0

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from pygb import (
    ActionSequence,
    GBHomeWeightGenerator,
    GoalBabblingContext,
    ObservationSequence,
    RuntimeData,
    SequenceType,
)

# @patch("pygb._impl._gb_utils._gb_home_weight_generator.GBHomeWeightGenerator._calc_weights")
# @patch("pygb._impl._gb_utils._gb_home_weight_generator.GBHomeWeightGenerator._choose_previous_data")
# def test_generate(choose_previous_data_mock: MagicMock, calc_weights_mock: MagicMock) -> None:
#     calc_weights_mock.return_value = (2.0, 4.0)
#     choose_previous_data_mock.return_value = (np.array([1.0]), np.array([2.0]), np.array([3.0]))
#     context_mock = MagicMock(spec=GoalBabblingContext, runtime_data=MagicMock(spec=RuntimeData))

#     generator = GBHomeWeightGenerator()

#     assert generator.generate(context_mock) == 4.0


@pytest.mark.parametrize(
    ("observation_index", "current_sequence", "previous_sequence", "expected_observation", "expected_action"),
    [
        (
            0,
            ActionSequence(None, None),
            ObservationSequence(
                None,
                None,
                observations=[np.array([0.0]), np.array([1.0])],
                predicted_actions=[np.array([-1.0]), np.array([-2.0])],
            ),
            np.array([1.0]),
            np.array([-2.0]),
        ),
        (
            0,
            ActionSequence(None, None),
            ActionSequence(
                None,
                None,
                observations=[np.array([2.0]), np.array([3.0])],
                actions=[np.array([-3.0]), np.array([-4.0])],
            ),
            np.array([3.0]),
            np.array([-4.0]),
        ),
        (
            1,
            ActionSequence(
                None,
                None,
                actions=[np.array([-5.5]), np.array([-6.0])],
                observations=[np.array([5.5]), np.array([6.0])],
            ),
            None,
            np.array([5.5]),
            np.array([-5.5]),
        ),
    ],
)
def test_choose_previous_data(
    observation_index: int,
    current_sequence: SequenceType,
    previous_sequence: SequenceType,
    expected_observation: np.ndarray,
    expected_action: np.ndarray,
) -> None:
    context_mock = MagicMock(
        spec=GoalBabblingContext,
        runtime_data=MagicMock(
            spec=RuntimeData,
            observation_index=observation_index,
            current_sequence=current_sequence,
            previous_sequence=previous_sequence,
        ),
    )

    generator = GBHomeWeightGenerator()

    observation, action = generator._choose_previous_data(context_mock)

    assert observation == expected_observation
    assert action == expected_action


# @patch("pygb._impl._gb_utils._gb_home_weight_generator.GBHomeWeightGenerator._choose_previous_data")
# def test_generate(choose_mock: MagicMock) -> None:
#     choose_mock.return_value = (np.array([1.0, 2.0]), np.array([0.1, 0.5, 1.0]))

#     context_mock = MagicMock(
#         spec=GoalBabblingContext, runtime_data=MagicMock(spec=RuntimeData, current_sequence=ActionSequence(None, None))
#     )

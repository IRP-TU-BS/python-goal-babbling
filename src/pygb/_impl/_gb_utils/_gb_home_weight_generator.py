import numpy as np

from pygb._impl._core._abstract_utils import AbstractWeightGenerator
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._runtime_data import ActionSequence, ObservationSequence


class GBHomeWeightGenerator(AbstractWeightGenerator[GoalBabblingContext]):
    def __init__(self, norm: int | None = None) -> None:
        self.norm = norm

    def _choose_previous_data(self, context: GoalBabblingContext) -> tuple[np.ndarray, np.ndarray]:
        seq = context.runtime_data.current_sequence

        if not isinstance(seq, ActionSequence):
            raise RuntimeError(
                """Failed to select previous data for go-home weight calculation: Current sequence must """
                f"""be of type {ActionSequence.__qualname__} (is {type(seq)})."""
            )
        observation_idx = context.runtime_data.observation_index

        if observation_idx < 1:
            if context.runtime_data.previous_sequence is None:
                raise NotImplementedError(f"Failed to generate weights for home sequence: No previous sequence found.")

            # no previous sequence -> training has just started
            if isinstance(context.runtime_data.previous_sequence, ObservationSequence):
                prev_observation = context.runtime_data.previous_sequence.observations[-1]
                prev_action = context.runtime_data.previous_sequence.predicted_actions[-1]
            else:
                prev_observation = context.runtime_data.previous_sequence.observations[-1]
                prev_action = context.runtime_data.previous_sequence.actions[-1]

        else:
            prev_observation = seq.observations[observation_idx - 1]
            prev_action = seq.actions[observation_idx - 1]

        return prev_observation, prev_action

    def generate(self, context: GoalBabblingContext) -> float:
        seq = context.runtime_data.current_sequence

        if not isinstance(seq, ActionSequence):
            raise RuntimeError(
                """Failed to calculate weights for a go-home-sequence: Current sequence is no """
                f""" {ActionSequence.__qualname__} instance."""
            )

        prev_observation, prev_action = self._choose_previous_data(context)

        observation_idx = context.runtime_data.observation_index

        observation_diff = seq.observations[observation_idx] - prev_observation
        action_diff = seq.actions[observation_idx] - prev_action

        observation_norm = np.linalg.norm(observation_diff, self.norm)
        action_norm = np.linalg.norm(action_diff, self.norm)

        if observation_norm == 0:
            observation_norm = 1.0

        if action_norm == 0:
            w_eff = 0.0
        else:
            w_eff = observation_norm / action_norm

        return min(w_eff, 1)

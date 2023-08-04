import numpy as np

from pygb._impl._core._abstract_utils import AbstractWeightGenerator
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._gb_utils._gb_weight_generator import GBWeightGenerator


class GBHomeWeightGenerator(GBWeightGenerator, AbstractWeightGenerator[GoalBabblingContext]):
    def __init__(self, norm: int | None = None) -> None:
        super().__init__(norm)

    def generate(self, context: GoalBabblingContext) -> float:
        prev_local_goal, prev_local_pred, prev_action = self._choose_previous_data(context)

        seq = context.runtime_data.current_sequence
        observation_idx = context.runtime_data.observation_index

        _, w_eff = self._calc_weights(
            local_goal=seq.local_goals[observation_idx],
            prev_local=prev_local_goal,
            local_goal_pred=seq.observations[observation_idx],
            prev_local_pred=prev_local_pred,
            action=seq.predicted_actions[observation_idx],
            prev_action=prev_action,
        )

        return w_eff

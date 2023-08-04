import numpy as np

from pygb._impl._core._abstract_utils import AbstractWeightGenerator
from pygb._impl._core._context import GoalBabblingContext


class GBWeightGenerator(AbstractWeightGenerator[GoalBabblingContext]):
    def __init__(self, norm: int | None = None) -> None:
        super().__init__()
        self.norm = norm

    def generate(self, context: GoalBabblingContext) -> float:
        seq = context.runtime_data.current_sequence
        observation_idx = context.runtime_data.observation_index

        if observation_idx < 1:
            raise RuntimeError(
                f"Failed to generate GB weight: Observation index must not be smaller than 1 (is {observation_idx})."
            )

        w_dir, w_eff = self._calc_weights(
            local_goal=seq.local_goals[observation_idx],
            prev_local=seq.local_goals[observation_idx - 1],
            local_goal_pred=seq.predicted_local_goals[observation_idx],
            prev_local_pred=seq.predicted_local_goals[observation_idx - 1],
            action=seq.predicted_actions[observation_idx],
            prev_action=seq.predicted_actions[observation_idx - 1],
        )

        return w_dir * w_eff

    def _calc_weights(
        self,
        local_goal: np.ndarray,
        prev_local: np.ndarray,
        local_goal_pred: np.ndarray,
        prev_local_pred: np.ndarray,
        action: np.ndarray,
        prev_action: np.ndarray,
    ) -> tuple[float, float]:
        prediction_diff = local_goal_pred - prev_local_pred
        action_diff = action - prev_action
        local_goal_diff = local_goal - prev_local

        prediction_norm = np.linalg.norm(prediction_diff, self.norm)
        action_norm = np.linalg.norm(action_diff, self.norm)
        local_goal_norm = np.linalg.norm(local_goal_diff, self.norm)

        if prediction_norm <= 1e-10:
            prediction_norm = 1  # TODO this seems questionable

        if action_norm <= 1e-10:
            w_eff = 0
        else:
            w_eff = prediction_norm / action_norm

        if local_goal_norm <= 1e-10:
            local_goal_norm = 1  # TODO

        goal_cosine = np.dot(prediction_diff / prediction_norm, local_goal_diff / local_goal_norm)

        w_dir = 0.5 + 0.5 * goal_cosine

        return w_dir, w_eff

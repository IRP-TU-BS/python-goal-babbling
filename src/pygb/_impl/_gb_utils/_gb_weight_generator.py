import numpy as np

from pygb._impl._core._abstract_utils import AbstractWeightGenerator
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._runtime_data import ActionSequence, ObservationSequence


class GBWeightGenerator(AbstractWeightGenerator[GoalBabblingContext]):
    """Weight generator which implemenst the original Goal Babbling weight from Rolf et. al (2010) for normal sequences,
    i.e. sequences between global goals in observation space."""

    def __init__(self, norm: int | None = None) -> None:
        """Constructor.

        Args:
            norm: Sets the norm type used. See numpy.norm documentation for details. Defaults to None.
        """
        super().__init__()
        self.norm = norm

    def generate(self, context: GoalBabblingContext) -> float:
        """Generates a sample weight based on the current Goal Babbling context.

        Args:
            context: Goal Babbling context.

        Raises:
            RuntimeError: In case the current observation index read from the context is less than 1.

        Returns:
            The calculated weight.
        """
        prev_local_goal, prev_local_pred, prev_action = self._choose_previous_data(context)

        seq = context.runtime_data.current_sequence
        observation_idx = context.runtime_data.observation_index

        w_dir, w_eff = self._calc_weights(
            local_goal=seq.local_goals[observation_idx],
            prev_local=prev_local_goal,
            local_goal_pred=seq.observations[observation_idx],
            prev_local_pred=prev_local_pred,
            action=seq.predicted_actions[observation_idx],
            prev_action=prev_action,
        )

        return w_dir * w_eff

    def _choose_previous_data(self, context: GoalBabblingContext) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        seq = context.runtime_data.current_sequence

        if isinstance(seq, ActionSequence):
            raise NotImplementedError(
                """Failed to choose previous data for weight calculation: Current sequence is an """
                f"""{ActionSequence.__qualname__} instance, but weights are calculated for an """
                f"""{ObservationSequence.__qualname__}."""
            )

        observation_idx = context.runtime_data.observation_index

        if observation_idx < 1:
            # start of a new sequence
            if context.runtime_data.previous_sequence is None or isinstance(
                context.runtime_data.previous_sequence, ActionSequence
            ):
                # training has just started/last sequence returned to home
                # -> use home action and observation as previous steps
                prev_local_goal = context.current_parameters.home_observation
                prev_local_pred = context.current_parameters.home_observation
                prev_action = context.current_parameters.home_action
            else:
                # start of the sequence, so there is no previous observation
                # -> use last sequence's observation
                prev_local_goal = context.runtime_data.previous_sequence.local_goals[-1]
                prev_local_pred = context.runtime_data.previous_sequence.observations[-1]
                prev_action = context.runtime_data.previous_sequence.predicted_actions[-1]
        else:
            # somewhere in the middle of a sequence, simply use the previous observation:
            prev_local_goal = seq.local_goals[observation_idx - 1]
            prev_local_pred = seq.observations[observation_idx - 1]
            prev_action = seq.predicted_actions[observation_idx - 1]

        return prev_local_goal, prev_local_pred, prev_action

    def _calc_weights(
        self,
        local_goal: np.ndarray,
        prev_local: np.ndarray,
        local_goal_pred: np.ndarray,
        prev_local_pred: np.ndarray,
        action: np.ndarray,
        prev_action: np.ndarray,
    ) -> tuple[float, float]:
        """Calculate directional and efficiency weights.

        Args:
            local_goal: Target local goal (ground truth).
            prev_local: Previous local goal (ground truth).
            local_goal_pred: Observation which is reached after the predicted action has been executed by the forward
                model.
            prev_local_pred: Previous observation prediction (similar to local_goal_pred).
            action: Predicted action.
            prev_action: Previous predicted action.

        Returns:
            Tuple (w_dir, w_eff).
        """
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

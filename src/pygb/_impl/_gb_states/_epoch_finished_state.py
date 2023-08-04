import numpy as np

from pygb._impl._core._abstract_state import AbstractState
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._events import EventSystem
from pygb._impl._core._model import AbstractForwardModel, AbstractInverseEstimator


class EpochFinishedState(AbstractState[GoalBabblingContext]):
    epoch_set_complete: str = "epoch_set_complete"
    epoch_set_not_complete: str = "epoch_set_not_complete"

    def __init__(self, context: GoalBabblingContext, event_system: EventSystem, name: str | None = None) -> None:
        super().__init__(context, event_system, name)

    def __call__(self) -> str:
        # calculate performance errors on test set and all optional goal sets:
        self.context.runtime_data.performance_error = self._evaluate(
            self.context.forward_model,
            self.context.inverse_estimate,
            self.context.current_goal_set.test,
        )
        if self.context.current_goal_set.optional_test is not None:
            for name, goals in self.context.current_goal_set.optional_test.items():
                self.context.runtime_data.opt_performance_errors[name] = self._evaluate(
                    self.context.forward_model, self.context.inverse_estimate, goals
                )

        self.events.emit("epoch-complete", self.context)

        # check if any of the stopping criteria is met:
        if any([criteria.fulfilled(self.context) for criteria in self.context.current_parameters.stopping_criteria]):
            return EpochFinishedState.epoch_set_complete

        if self.context.runtime_data.epoch_index < self.context.current_parameters.len_epoch_set:
            self.context.runtime_data.epoch_index += 1
            return EpochFinishedState.epoch_set_not_complete

        return EpochFinishedState.epoch_set_complete

    def _evaluate(
        self, forward_model: AbstractForwardModel, inverse_estimate: AbstractInverseEstimator, observations: np.ndarray
    ) -> float:
        predicted_actions = inverse_estimate.predict_batch(observations)
        predicted_actions = forward_model.clip(predicted_actions)
        predicted_observations = forward_model.forward_batch(predicted_actions)

        return np.sqrt(np.mean((predicted_observations - observations) ** 2))

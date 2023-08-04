import numpy as np

from pygb._impl._core._abstract_state import AbstractState
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._events import EventSystem
from pygb._impl._core._model import AbstractForwardModel, AbstractInverseEstimator


class EpochFinishedState(AbstractState[GoalBabblingContext]):
    """Class which represents the state machine state after an epoch has been completed. Decides whether the current
    epoch set is completed or another epoch needs to be executed."""

    epoch_set_complete: str = "epoch_set_complete"
    epoch_set_not_complete: str = "epoch_set_not_complete"

    def __init__(self, context: GoalBabblingContext, event_system: EventSystem, name: str | None = None) -> None:
        """Constructor.

        Args:
            context: State machine context.
            event_system: Event system singleton instance.
            name: State name. Defaults to None.
        """
        super().__init__(context, event_system, name)

    def __call__(self) -> str | None:
        """Executes the state.

        Steps:
            1) Calculate the RMSE on the test goal set
            2) Calculate the RMSE on optional test goal sets
            3) Emit an 'epoch-complete' event
            4) If any of the defined stopping criteria is fulfilled or the maximum number of epochs in the current epoch
                set is reached: Return an 'epoch_set_complete' transition
            5) Otherwise return an 'epoch_set_not_complete' transition

        Returns:
            Transition name.
        """
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

        # check if any of the stopping criteria is met or the number of epochs per epoch set is reached:
        if (
            any([criteria.fulfilled(self.context) for criteria in self.context.current_parameters.stopping_criteria])
            or self.context.runtime_data.epoch_index >= self.context.current_parameters.len_epoch_set
        ):
            return EpochFinishedState.epoch_set_complete

        # otherwise we are still in the epoch set and need to start a new epoch:
        self.context.runtime_data.epoch_index += 1
        return EpochFinishedState.epoch_set_not_complete

    def _evaluate(
        self, forward_model: AbstractForwardModel, inverse_estimate: AbstractInverseEstimator, observations: np.ndarray
    ) -> float:
        """Calculates the inverse estimate's performance on the specified observation set.

        Note: This function calculates the Performance Error rather than the Command Error (see Rolf et al, 2010). The
        Performance Error measures the error between the target observation set and the estimated observations, which
        are produced by executing the action estimations on the forward model. Specifically:

        Observation set O
        Estimated set of actions A* = g(O), where g() represents the inverse estimate
        Estimated set of observations O* = f(A*), where f() represent the forward model

        Performance Error: E = sqrt( mean( O - O* )^2 ) = sqrt( mean( O - f(g(O)) ) )


        Args:
            forward_model: Forward model instance.
            inverse_estimate: Inverse estimate instance.
            observations: Observation set, e.g. a test goal set.

        Returns:
            Root mean squared error on the specified observation set.
        """
        predicted_actions = inverse_estimate.predict_batch(observations)
        predicted_actions = forward_model.clip_batch(predicted_actions)
        predicted_observations = forward_model.forward_batch(predicted_actions)

        return np.sqrt(np.mean((predicted_observations - observations) ** 2))

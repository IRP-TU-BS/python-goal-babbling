import logging
from copy import deepcopy

import numpy as np

from pygb._impl._core._abstract_state import AbstractState
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._epoch_set_record import EpochSetRecord
from pygb._impl._core._event_system import EventSystem
from pygb._impl._core._events import Events
from pygb._impl._core._model import AbstractForwardModel, AbstractInverseEstimate
from pygb._impl._core._runtime_data import ActionSequence, ObservationSequence

_logger = logging.getLogger(__name__)


class EpochFinishedState(AbstractState[GoalBabblingContext]):
    """Class which represents the state machine state after an epoch has been completed. Decides whether the current
    epoch set is completed or another epoch needs to be executed."""

    epoch_set_complete: str = "epoch_set_complete"
    epoch_set_not_complete: str = "epoch_set_not_complete"

    def __init__(
        self, context: GoalBabblingContext, event_system: EventSystem = EventSystem.instance(), name: str | None = None
    ) -> None:
        """Constructor.

        Args:
            context: State machine context.
            event_system: Event system singleton instance.
            name: State name. Defaults to None.
        """
        if event_system is None:
            raise ValueError("Failed to initialize instance: No event system instance provided.")
        super().__init__(context, event_system, name)

    def __call__(self) -> str | None:
        """Executes the state.

        Steps:
            1) Calculate the RMSE on the test goal set
            2) Calculate the RMSE on optional test goal sets
            3) Emit an 'epoch-complete' event
            4) Store the current inverse estimate in case it beats the previous best inverse estimate
            5) Reset epoch-specific data such as recorded sequences and current sequence
            6) If any of the defined stopping criteria is fulfilled or the maximum number of epochs in the current epoch
                set is reached: Return an 'epoch_set_complete' transition
            7) Otherwise return an 'epoch_set_not_complete' transition

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

        self.events.emit(Events.EPOCH_COMPLETE, self.context)
        stop_reason = self._evaluate_stop(self.context)

        self._update_record(self.context, stop_reason)

        # reset epoch specific runtime data
        self.context.runtime_data.current_sequence = None
        self.context.runtime_data.sequences = []
        self.context.runtime_data.sequence_index = 0

        # check if any of the stopping criteria is met or the number of epochs per epoch set is reached:
        if stop_reason is not None:
            return EpochFinishedState.epoch_set_complete

        # otherwise we are still in the epoch set and need to start a new epoch:
        self.context.runtime_data.epoch_index += 1
        return EpochFinishedState.epoch_set_not_complete

    def transitions(self) -> list[str]:
        return [EpochFinishedState.epoch_set_complete, EpochFinishedState.epoch_set_not_complete]

    def _update_record(self, context: GoalBabblingContext, stop_reason: str | None) -> None:
        if len(context.epoch_set_records) < context.runtime_data.epoch_set_index + 1:
            context.epoch_set_records.append(EpochSetRecord())

        observation_sequence_count = 0
        action_sequence_count = 0
        for sequence in context.runtime_data.sequences:
            if isinstance(sequence, ActionSequence):
                action_sequence_count += 1
            elif isinstance(sequence, ObservationSequence):
                observation_sequence_count += 1

        obs_sample_count = observation_sequence_count * context.current_parameters.len_sequence
        act_sample_count = action_sequence_count * context.current_parameters.len_sequence
        perf = {"test": context.runtime_data.performance_error} | context.runtime_data.opt_performance_errors

        context.epoch_set_records[-1].stop_reason = stop_reason
        context.epoch_set_records[-1].total.epoch_count += 1
        context.epoch_set_records[-1].total.observation_sequence_count += observation_sequence_count
        context.epoch_set_records[-1].total.action_sequence_count += action_sequence_count
        context.epoch_set_records[-1].total.observation_sample_count += obs_sample_count
        context.epoch_set_records[-1].total.action_sample_count += act_sample_count
        context.epoch_set_records[-1].total.performance.update(perf)

        update_best = False

        # If a model cache is specified it sets the criteria which determines the 'best' estimate. If not, simply use
        # the test performance as a metric.
        if context.estimate_cache is not None and context.estimate_cache.conditional_save(
            context.inverse_estimate, context.runtime_data.epoch_set_index, context
        ):
            update_best = True
            _logger.info(f"Stored new best inverse estimate [{context.runtime_data.performance_error}]")

        elif context.estimate_cache is None and (
            len(context.epoch_set_records[-1].best.performance) == 0
            or context.epoch_set_records[-1].best.performance["test"] > context.runtime_data.performance_error
        ):
            update_best = True

        if update_best:
            context.epoch_set_records[-1].best = deepcopy(context.epoch_set_records[-1].total)

    def _evaluate_stop(self, context: GoalBabblingContext) -> str | None:
        if context.runtime_data.epoch_index >= context.current_parameters.len_epoch_set - 1:
            return "epoch_count_reached"

        for criteria in context.current_parameters.stopping_criteria:
            if criteria.fulfilled(context):
                return str(criteria)

        return None

    def _evaluate(
        self, forward_model: AbstractForwardModel, inverse_estimate: AbstractInverseEstimate, observations: np.ndarray
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

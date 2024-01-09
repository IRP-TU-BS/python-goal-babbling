import logging

import numpy as np

from pygb._impl._core._abstract_state import AbstractState
from pygb._impl._core._abstract_utils import (
    AbstractGoalSelector,
    AbstractNoiseGenerator,
    AbstractSequenceGenerator,
    AbstractWeightGenerator,
)
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._event_system import EventSystem
from pygb._impl._core._model import AbstractForwardModel, AbstractInverseEstimate
from pygb._impl._core._runtime_data import ActionSequence, ObservationSequence

_logger = logging.getLogger(__name__)


class GenerateSequenceState(AbstractState[GoalBabblingContext]):
    sequence_finished: str = "sequence_finished"

    def __init__(
        self,
        context: GoalBabblingContext,
        goal_selector: AbstractGoalSelector,
        goal_sequence_generator: AbstractSequenceGenerator,
        noise_generator: AbstractNoiseGenerator,
        weight_generator: AbstractWeightGenerator,
        event_system: EventSystem = EventSystem.instance(),
        name: str | None = None,
    ) -> None:
        super().__init__(context, event_system, name)

        self.goal_selector = goal_selector
        self.goal_sequence_generator = goal_sequence_generator
        self.weight_generator = weight_generator
        self.noise_generator = noise_generator

    def __call__(self) -> str | None:
        """Trains the inverse estimate on a sequence of observations between two global goals.

        Steps:
            1) Select a global target goal
            2) Generate a sequence of local goals in between start goal (the previous end goal) and end goal
            3) Traverse the generated sequence, training the inverse estimator using samples created by the Goal
                Babbling method

        Returns:
            Transition name once the sequence has been finished.
        """
        # generate sequence between previous stop goal and new target goal:
        target_goal_index, target_goal = self.goal_selector.select(self.context)
        _logger.debug("Selected sequence target: Goal index %d" % target_goal_index)

        sequence = self._generate_new_sequence(target_goal, target_goal_index, self.context)

        # update current sequence and previous sequence:
        self.context.runtime_data.current_sequence = sequence

        # traverse local goals and fit them using the generated samples + weights:
        for observation_index, local_goal in enumerate(sequence.local_goals):
            self.context.runtime_data.observation_index = observation_index

            action = self.context.inverse_estimate.predict(local_goal)
            action += self.noise_generator.generate(local_goal.T)
            action = self.context.forward_model.clip(action)
            observation = self.context.forward_model.forward(action)

            self.noise_generator.update()

            # important: WeightGenerator uses the Goal Babbling context, so update it here
            sequence.predicted_actions.append(action)
            sequence.observations.append(observation)

            weight = self.weight_generator.generate(self.context)
            sequence.weights.append(weight)

            self.context.inverse_estimate.fit(observation, action, weight)

        # increase stop goal's visit count:
        self.context.runtime_data.train_goal_visit_count[target_goal_index] += 1
        _logger.debug(
            "Train goal %d visit count update: %d->%d"
            % (
                target_goal_index,
                self.context.runtime_data.train_goal_visit_count[target_goal_index] - 1,
                self.context.runtime_data.train_goal_visit_count[target_goal_index],
            )
        )

        # note down performance error on sequence's target goal:
        self.context.runtime_data.train_goal_error[target_goal_index] = self._calc_performance_error(
            target_goal, self.context.forward_model, self.context.inverse_estimate
        )
        _logger.debug("Train goal %d error update: %.8f" % (target_goal_index, rmse))

        return GenerateSequenceState.sequence_finished

    def transitions(self) -> list[str]:
        return [GenerateSequenceState.sequence_finished]

    def _generate_new_sequence(
        self, target_goal: np.ndarray, target_goal_index: int, context: GoalBabblingContext
    ) -> ObservationSequence:
        if context.runtime_data.previous_sequence is None or isinstance(
            context.runtime_data.previous_sequence, ActionSequence
        ):
            # start of epoch set/previous sequence was to home -> no previous target goal, so we start at the home
            # observation
            start_goal = context.current_parameters.home_observation
        else:
            start_goal = context.runtime_data.previous_sequence.stop_goal

        local_goal_sequence = self.goal_sequence_generator.generate(
            start=start_goal, stop=target_goal, len_sequence=context.current_parameters.len_sequence
        )

        return ObservationSequence(
            start_goal=start_goal,
            stop_goal=target_goal,
            stop_goal_index=target_goal_index,
            local_goals=local_goal_sequence,
        )

    def _calc_performance_error(
        self, observation: np.ndarray, forward_model: AbstractForwardModel, estimate: AbstractInverseEstimate
    ) -> float:
        action_prediction = estimate.predict(observation)
        action_prediction = forward_model.clip(action_prediction)
        obs_prediction = forward_model.forward(action_prediction)

        return np.sqrt(np.mean((observation - obs_prediction)) ** 2)

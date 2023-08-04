from pygb._impl._core._abstract_state import AbstractState
from pygb._impl._core._abstract_utils import (
    AbstractGoalSelector,
    AbstractLocalGoalGenerator,
    AbstractNoiseGenerator,
    AbstractWeightGenerator,
)
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._events import EventSystem
from pygb._impl._core._runtime_data import SequenceData


class GenerateSequenceState(AbstractState[GoalBabblingContext]):
    sequence_finished: str = "sequence_finished"

    def __init__(
        self,
        context: GoalBabblingContext,
        goal_selector: AbstractGoalSelector,
        local_goal_generator: AbstractLocalGoalGenerator,
        noise_generator: AbstractNoiseGenerator,
        weight_generator: AbstractWeightGenerator,
        event_system: EventSystem | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(context, event_system, name)

        self.goal_selector = goal_selector
        self.local_goal_selector = local_goal_generator
        self.weight_generator = weight_generator
        self.noise_generator = noise_generator

    def __call__(self) -> str:
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
        sequence = self._generate_new_sequence(self.context)

        # update current sequence and previous sequence:
        self.context.runtime_data.update_current_sequence(sequence)

        # traverse local goals and fit them using the generated samples + weights:
        for observation_index, local_goal in enumerate(sequence.local_goals):
            self.context.runtime_data.observation_index = observation_index

            action = self.context.inverse_estimate.predict(local_goal)
            action += self.noise_generator.generate(local_goal)
            action = self.context.forward_model.clip(action)
            observation = self.context.forward_model.forward(action)

            # important: WeightGenerator uses the Goal Babbling context, so update it here
            sequence.predicted_actions.append(action)
            sequence.observations.append(observation)

            weight = self.weight_generator.generate(self.context)
            sequence.weights.append(weight)

            self.context.inverse_estimate.fit(observation, action, weight)

        # add sequence to completed sequences:
        self.context.runtime_data.sequences.append(sequence)

        # increase stop goal's visit count:
        self.context.runtime_data.train_goal_visit_count[sequence.stop_glob_goal_idx] += 1

        return GenerateSequenceState.sequence_finished

    def _generate_new_sequence(self, context: GoalBabblingContext) -> SequenceData:
        target_goal_index, target_goal = self.goal_selector.select(context)

        if context.runtime_data.previous_sequence is None:
            # start of epoch set -> no previous sequence, so we start at the home observation
            start_goal = context.current_parameters.home_observation
            start_index = -1
        else:
            start_index = context.runtime_data.previous_sequence.stop_glob_goal_idx
            start_goal = context.current_goal_set.train[start_index]

        local_goal_sequence = self.local_goal_selector.generate(
            start_goal=start_goal, stop_goal=target_goal, len_sequence=context.current_parameters.len_sequence
        )

        return SequenceData(
            start_glob_goal_idx=start_index, stop_glob_goal_idx=target_goal_index, local_goals=local_goal_sequence
        )
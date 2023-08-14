import numpy as np

from pygb._impl._core._abstract_state import AbstractState
from pygb._impl._core._abstract_utils import (
    AbstractSequenceGenerator,
    AbstractWeightGenerator,
)
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._events import EventSystem
from pygb._impl._core._runtime_data import ActionSequence, ObservationSequence


class GenerateHomeSequenceState(AbstractState[GoalBabblingContext]):
    sequence_finished: str = "sequence_finished"

    def __init__(
        self,
        context: GoalBabblingContext,
        home_sequence_generator: AbstractSequenceGenerator,
        weight_generator: AbstractWeightGenerator,
        event_system: EventSystem | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(context, event_system, name)

        self.home_sequence_generator = home_sequence_generator
        self.weight_generator = weight_generator

    def __call__(self) -> str | None:
        sequence = self._generate_sequence(target_action=self.context.current_parameters.home_action)

        self.context.runtime_data.update_current_sequence(sequence)

        for index, action in enumerate(sequence.actions):
            self.context.runtime_data.observation_index = index

            observation = self.context.forward_model.forward(action)
            sequence.observations.append(observation)
            weight = self.weight_generator.generate(self.context)
            sequence.weights.append(weight)

            self.context.inverse_estimate.fit(observation, action, weight)

        self.context.runtime_data.sequences.append(sequence)

        return GenerateHomeSequenceState.sequence_finished

    def _generate_sequence(self, target_action: np.ndarray) -> ActionSequence:
        if self.context.runtime_data.previous_sequence is None:
            raise NotImplementedError(f"Failed to generate home sequence: No previous sequence found.")

        if isinstance(self.context.runtime_data.previous_sequence, ObservationSequence):
            start_action = self.context.runtime_data.previous_sequence.predicted_actions[-1]
        elif isinstance(self.context.runtime_data.previous_sequence, ActionSequence):
            start_action = self.context.runtime_data.previous_sequence.stop_action

        home_action_sequence = self.home_sequence_generator.generate(
            start=start_action,
            stop=target_action,
            len_sequence=self.context.current_parameters.len_sequence,
        )

        return ActionSequence(start_action=start_action, stop_action=target_action, actions=home_action_sequence)

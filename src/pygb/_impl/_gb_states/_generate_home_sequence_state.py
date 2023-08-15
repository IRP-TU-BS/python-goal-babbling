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
    """Class which represents the state machine state when a sequence to the home action/observation pair is
    generated.
    """

    sequence_finished: str = "sequence_finished"

    def __init__(
        self,
        context: GoalBabblingContext,
        home_sequence_generator: AbstractSequenceGenerator,
        weight_generator: AbstractWeightGenerator,
        event_system: EventSystem | None = None,
        name: str | None = None,
    ) -> None:
        """Constructor.

        Args:
            context: Goal Babbling context.
            home_sequence_generator: Sequence generator instance used to generate a sequence in action space.
            weight_generator: Weight generator instance.
            event_system: Event system singleton instance. Defaults to None.
            name: Optional name. Defaults to None.
        """
        super().__init__(context, event_system, name)

        self.home_sequence_generator = home_sequence_generator
        self.weight_generator = weight_generator

    def __call__(self) -> str | None:
        """Exectues the state.

        Steps:
            1) Generate a sequence to the home action in action space using the sequence generator instance
            2) Update the context's runtime data
            3) Traverse the generated sequence, training the inverse estimator using observations (generated by the
                forward model) and actions (from the sequence)

        Returns:
            Transition name 'sequence_finished' once the sequence is finished.
        """
        sequence = self._generate_sequence(target_action=self.context.current_parameters.home_action)

        self.context.runtime_data.current_sequence = sequence

        for index, action in enumerate(sequence.actions):
            self.context.runtime_data.observation_index = index

            observation = self.context.forward_model.forward(action)
            sequence.observations.append(observation)
            weight = self.weight_generator.generate(self.context)
            sequence.weights.append(weight)

            self.context.inverse_estimate.fit(observation, action, weight)

        return GenerateHomeSequenceState.sequence_finished

    def _generate_sequence(self, target_action: np.ndarray) -> ActionSequence:
        """Generates an action sequence towards a target action.

        Args:
            target_action: Final action.

        Raises:
            NotImplementedError: In case the context's previous sequence is None. It does not make sense to return to
                home when the inverse estimate has not been trained at least once before.

        Returns:
            The generated sequence in action space.
        """
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

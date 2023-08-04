from pygb._impl._core._abstract_state import AbstractState
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._events import EventSystem


class SequenceFinishedState(AbstractState[GoalBabblingContext]):
    """Class which represents the state machine state after a sequence has been completed. Decides whether the current
    epoch is completed or another sequence is to be executed."""

    epoch_finished: str = "epoch_finished"
    epoch_not_finished: str = "epoch_not_finished"

    def __init__(self, context: GoalBabblingContext, event_system: EventSystem, name: str | None = None) -> None:
        """Constructor.

        Args:
            context: Goal Babbling context.
            event_system: Event system.
            name: Optional name. Defaults to None.
        """
        super().__init__(context, event_system, name)

    def __call__(self) -> str | None:
        """Execute the state.

        - Steps:
            1) Emit 'sequence-finished' event
            2) If the current epoch is not completed, increase the context's sequence index
            3) Return an 'epoch_finished' transition if the epoch is completed, else return an 'epoch_not_finished'
                tranisition

        Returns:
            Transition. Either 'epoch_finished' or 'epoch_not_finished'.
        """
        self.events.emit("sequence-finished", self.context)

        if self.context.runtime_data.sequence_index < self.context.current_parameters.len_epoch - 1:
            self.context.runtime_data.sequence_index += 1

            return SequenceFinishedState.epoch_not_finished

        return SequenceFinishedState.epoch_finished

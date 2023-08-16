import logging

from pygb._impl._core._abstract_state import AbstractState
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._events import EventSystem

_logger = logging.getLogger(__name__)


class EpochSetFinishedState(AbstractState[GoalBabblingContext]):
    """Class which represents the state machine state after an epoch set has been completed. Decides whether to stop
    training or training is to be continued."""

    continue_training: str = "continue_training"
    stop_training: str = "stop_training"

    def __init__(
        self,
        context: GoalBabblingContext,
        event_system: EventSystem,
        load_previous_best: bool = True,
        name: str | None = None,
    ) -> None:
        """Constructor.

        Args:
            context: Goal Babbling context.
            event_system: Event system singleton instance.
            name: State name. Defaults to None.
        """
        super().__init__(context, event_system, name)

        self.load_previous = load_previous_best

        if self.load_previous and self.context.model_store is None:
            raise RuntimeError(
                f"""Failed to initialize {self.__class__.__qualname__}: If 'load_previous' is set to True, """
                """the Goal Babbling context must contain a model storage."""
            )

    def __call__(self) -> str | None:
        """Executes the state.

        Steps:
            1) Emit an 'epoch-set-complete' event
            2) If the target number of epoch sets has not been reached, increase the epoch set index and return a
                'continue_training' transition
            3) If the target number of epoch sets has been reaches, return 'stop_training' transition

        Returns:
            Transition name.
        """
        self.events.emit("epoch-set-complete", self.context)

        # reset epoch-set-specific runtime data:
        train_goal_count = self.context.current_goal_set.train.shape[0]

        self.context.runtime_data.train_goal_error = [0.0] * train_goal_count
        self.context.runtime_data.train_goal_visit_count = [0] * train_goal_count
        self.context.runtime_data.epoch_index = 0

        if self.context.runtime_data.epoch_set_index < self.context.num_epoch_sets - 1:
            if self.load_previous:
                # context.model_store is not None due to check in __init__
                self.context.inverse_estimate = self.context.model_store.load(self.context.runtime_data.epoch_set_index)
                _logger.info(
                    f"""Loaded epoch set {self.context.runtime_data.epoch_set_index} best estimate for upcoming """
                    f"""epoch set {self.context.runtime_data.epoch_set_index + 1}"""
                )

            self.context.runtime_data.epoch_set_index += 1
            return EpochSetFinishedState.continue_training

        return EpochSetFinishedState.stop_training

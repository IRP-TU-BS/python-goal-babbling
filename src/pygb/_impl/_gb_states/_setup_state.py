import logging

from pygb._impl._core._abstract_state import AbstractState
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._event_system import EventSystem

_logger = logging.getLogger(__name__)


class SetupState(AbstractState[GoalBabblingContext]):
    setup_complete: str = "setup_complete"

    def __init__(
        self, context: GoalBabblingContext, event_system: EventSystem = EventSystem.instance(), name: str | None = None
    ) -> None:
        super().__init__(context, event_system, name)

    def __call__(self) -> str | None:
        train_goal_count = self.context.current_goal_set.train.shape[0]

        self.context.runtime_data.train_goal_error = [0.0] * train_goal_count
        self.context.runtime_data.train_goal_visit_count = [0] * train_goal_count

        _logger.debug("Initialized training goal stats for %d training goals" % train_goal_count)

        return SetupState.setup_complete

    def transitions(self) -> list[str]:
        return [SetupState.setup_complete]

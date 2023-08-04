from pygb._impl._core._abstract_state import AbstractState
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._events import EventSystem


class SetupState(AbstractState[GoalBabblingContext]):
    setup_complete: str = "setup_complete"

    def __init__(
        self, context: GoalBabblingContext, event_system: EventSystem | None = None, name: str | None = None
    ) -> None:
        super().__init__(context, event_system, name)

    def __call__(self) -> str | None:
        train_goal_count = self.context.current_goal_set.train.shape[0]

        self.context.runtime_data.train_goal_error = [0.0] * train_goal_count
        self.context.runtime_data.train_goal_visit_count = [0] * train_goal_count

        return SetupState.setup_complete

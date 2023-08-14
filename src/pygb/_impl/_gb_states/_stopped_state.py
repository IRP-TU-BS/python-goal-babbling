from pygb._impl._core._abstract_state import AbstractState
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._events import EventSystem


class StoppedState(AbstractState[GoalBabblingContext]):
    def __init__(
        self, context: GoalBabblingContext, event_system: EventSystem | None = None, name: str | None = None
    ) -> None:
        super().__init__(context, event_system, name)

    def __call__(self) -> str | None:
        self.context.set_stopped()
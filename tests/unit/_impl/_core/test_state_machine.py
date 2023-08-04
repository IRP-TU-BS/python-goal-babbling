from pygb import AbstractContext, AbstractState, EventSystem, StateMachine


class DummyContext(AbstractContext):
    def __init__(self) -> None:
        super().__init__()

        self.running = False
        self.counter = 0

    def is_running(self) -> bool:
        return self.running

    def set_running(self) -> None:
        self.running = True


class DummyState(AbstractState):
    def __init__(self, context: AbstractContext, event_system: EventSystem, name: str | None = None) -> None:
        super().__init__(context, event_system, name)

    def __call__(self) -> None:
        self.context.co

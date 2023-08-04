import pytest

from pygb import EventSystem, StateMachine
from pygb.interfaces import AbstractContext, AbstractState


class DummyContext(AbstractContext):
    def __init__(self) -> None:
        super().__init__()

        self.running = False
        self.state_calls: list[str] = []

    @property
    def num_epoch_sets(self) -> int:
        return 42

    def is_running(self) -> bool:
        return self.running

    def set_running(self) -> None:
        self.running = True

    def set_stopped(self) -> None:
        self.running = False


class DummyState(AbstractState[DummyContext]):
    def __init__(self, context: DummyContext, event_system: EventSystem | None = None, name: str | None = None) -> None:
        super().__init__(context, event_system, name)

    def __call__(self) -> str | None:
        self.context.state_calls.append(self.name)
        if len(self.context.state_calls) >= 3:
            self.context.running = False

        return self.name


class DummyFinalState(AbstractState[DummyContext]):
    def __init__(self, context: DummyContext, event_system: EventSystem | None = None, name: str | None = None) -> None:
        super().__init__(context, event_system, name)

    def __call__(self) -> str | None:
        return None


def test_init() -> None:
    context = DummyContext()
    state = DummyState(context)
    sm = StateMachine(context, initial_state=state)

    assert sm._transition_table == {}
    assert state.name == "DummyState"


def test_set_initial_state() -> None:
    context = DummyContext()
    state = DummyState(context, name="dummy_state")

    sm = StateMachine(context)
    sm.context.running = True

    with pytest.raises(RuntimeError):
        sm.initial_state = state

    sm.context.running = False
    sm.initial_state = state

    assert sm.initial_state == state
    assert sm.current_state == state


def test_set_current_state() -> None:
    context = DummyContext()
    state = DummyState(context, name="dummy_state")
    sm = StateMachine(context)

    with pytest.raises(NotImplementedError):
        sm.current_state = state

    sm.initial_state = state
    assert sm.current_state == state


def test_add() -> None:
    context = DummyContext()
    state = DummyState(context, name="dummy_state")
    sm = StateMachine(context)

    sm.add("test_transition", state)

    assert sm._transition_table == {"test_transition": state}

    with pytest.raises(ValueError):
        sm.add("test_transition", state)

    state2 = DummyState(context, name="dummy_state2")
    sm.add("test_transition", state2, no_raise=True)

    assert sm._transition_table == {"test_transition": state2}


def test_pop() -> None:
    context = DummyContext()
    sm = StateMachine(context)
    state1 = DummyState(context, name="state1")
    state2 = DummyState(context, name="state2")

    sm._transition_table = {"transition1": state1, "transition2": state2}

    with pytest.raises(KeyError):
        sm.pop("unknown_transition", no_raise=False)

    removed_state = sm.pop("unknown_transition", no_raise=True)
    assert removed_state is None

    removed_state = sm.pop("transition2")
    assert removed_state == state2
    assert sm._transition_table == {"transition1": state1}


def test_run() -> None:
    context = DummyContext()
    state1 = DummyState(context, name="state1")
    state2 = DummyState(context, name="state2")
    sm = StateMachine(context, initial_state=state1)

    sm.add(state1.name, state2)
    sm.add(state2.name, state1)

    assert sm._transition_table == {"state1": state2, "state2": state1}

    sm.run()

    assert not sm.context.is_running()
    assert context.state_calls == ["state1", "state2", "state1"]


def test_run_raises_if_initial_state_not_set() -> None:
    context = DummyContext()

    sm = StateMachine(context)

    with pytest.raises(RuntimeError, match="Failed to start state machine: No initial state set."):
        sm.run()


def test_state_machine_stops_after_receiving_none_as_transition() -> None:
    context = DummyContext()
    init_state = DummyState(context, name="init_state")
    final_state = DummyFinalState(context, name="final_state")

    sm = StateMachine(context, initial_state=init_state)

    sm.add(init_state.name, final_state)

    sm.run()

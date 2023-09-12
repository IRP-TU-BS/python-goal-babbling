from typing import Generator

import pytest

from pygb import EventSystem, observes
from pygb.interfaces import AbstractContext


class DummyContext(AbstractContext):
    def __init__(self) -> None:
        super().__init__()

        self.value = 42

    def num_epoch_sets(self) -> int:
        return 24

    def is_running(self) -> bool:
        return super().is_running()

    def set_running(self) -> None:
        return super().set_running()

    def set_stopped(self) -> None:
        return super().set_stopped()


def test_get_singleton_instance() -> None:
    events = EventSystem.instance()
    assert events is not None

    events2 = EventSystem.instance()

    assert id(events) == id(events2)


def test_emit() -> None:
    events = EventSystem.instance()
    context_value = None
    called = False

    def test_observer(context: DummyContext) -> None:
        nonlocal context_value
        context_value = context.value

    def test_not_called_observer(context: DummyContext):
        nonlocal called
        called = True

    events.event_observers["test_event"].append(test_observer)
    events.event_observers["test_not_called_event"].append(test_not_called_observer)
    events.emit("test_event", DummyContext())

    assert context_value == 42
    assert not called


def test_register_observer() -> None:
    events = EventSystem.instance()

    events.register_observer("test-event", lambda context: None)
    assert len(events.event_observers["test-event"]) == 1


def test_remove_observer() -> None:
    events = EventSystem.instance()

    def observer1(context: AbstractContext) -> None:
        pass

    def observer2(context: AbstractContext) -> None:
        pass

    events.event_observers["test-event-1"].append(observer1)
    events.event_observers["test-event-1"].append(observer2)

    events.event_observers["test-event-2"].append(observer1)

    with pytest.raises(RuntimeError):
        events.remove_observer("test-event-2", observer2, no_raise=False)

    events.remove_observer("test-event-2", observer2, no_raise=True)

    events.remove_observer("test-event-1", observer1)
    assert dict(events.event_observers) == {"test-event-1": [observer2], "test-event-2": [observer1]}

    events.remove_observer("test-event-2", observer1)
    assert dict(events.event_observers) == {"test-event-1": [observer2], "test-event-2": []}


def test_observes_decorator() -> None:
    events = EventSystem.instance()

    called = False

    @observes("test-event")
    def observer(context: AbstractContext) -> None:
        nonlocal called
        called = True

    assert dict(events.event_observers) == {"test-event": [observes("test-event")(observer)]}


def test_clear() -> None:
    events = EventSystem.instance()

    @observes("test-event")
    def observer(context: AbstractContext) -> None:
        ...

    assert len(events.event_observers.items()) == 1

    events.clear()

    assert len(events.event_observers.items()) == 0

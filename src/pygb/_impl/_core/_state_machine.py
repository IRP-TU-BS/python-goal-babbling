from typing import Any

from pygb._impl._core._context import AbstractContext
from pygb._impl._core._states import AbstractState


class StateMachine:
    def __init__(self, context: AbstractContext, initial_state: AbstractState | None = None) -> None:
        self.context = context

        self._current_state = initial_state
        self._initial_state = initial_state
        self._transition_table: dict[str, AbstractState] = dict()

    @property
    def current_state(self) -> AbstractState | None:
        return self._current_state

    @current_state.setter
    def current_state(self, _v: Any) -> None:
        raise NotImplementedError("Failed to set the current state: Setting the state is not allowed.")

    @property
    def initial_state(self) -> AbstractState | None:
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state: AbstractState) -> None:
        if self.context.is_running():
            raise RuntimeError("Failed to set the initial state: State machine is already running.")

        self._current_state = state
        self._initial_state = state

    def run(self) -> None:
        self.context.set_running()

        while self.context.is_running():
            transition = self.current_state()
            if transition not in self._transition_table:
                raise RuntimeError(
                    f"""State Machine failure: State '{self._current_state.name}' returned unknown transition """
                    f"""'{transition}'."""
                )

            self._current_state = self._transition_table[transition]

    def add(self, transition_name: str, state: AbstractState, no_raise: bool = False) -> None:
        if transition_name in self._transition_table and not no_raise:
            raise ValueError(
                f"""Failed to add state '{state.name}': Transition '{transition_name}' already has a target state """
                f""" ('{self._transition_table[transition_name].name}'). To override existing transitions use the """
                """'no_raise' flag."""
            )

        self._transition_table[transition_name] = state

    def pop(self, transition_name: str, no_raise: bool = True) -> AbstractState | None:
        if transition_name in self._transition_table:
            return self._transition_table.pop(transition_name)
        elif not no_raise:
            return None

        raise KeyError(
            f"""Failed to remove transition: '{transition_name}' does not exist. Use the 'no_raise' flag """
            """to suppress this exception."""
        )

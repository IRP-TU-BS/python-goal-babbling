"""Module which contains the state machine."""

from typing import Any

from pygb._impl._core._context import AbstractContext
from pygb._impl._core._states import AbstractState


class StateMachine:
    """State machine class."""

    def __init__(self, context: AbstractContext, initial_state: AbstractState | None = None) -> None:
        """Constructor.

        Args:
            context: Context instance.
            initial_state: Initial state. Defaults to None.
        """
        self.context = context

        self._current_state = initial_state
        self._initial_state = initial_state
        self._transition_table: dict[str, AbstractState] = dict()

    @property
    def current_state(self) -> AbstractState | None:
        """Current state property. Is updated with each state machine cycle.

        Returns:
            The currently active state.
        """
        return self._current_state

    @current_state.setter
    def current_state(self, _v: Any) -> None:
        """Current state setter, which is prohibited."""
        raise NotImplementedError("Failed to set the current state: Setting the state is not allowed.")

    @property
    def initial_state(self) -> AbstractState | None:
        """Initial state getter. The initial state remains the same once the state machine is started.

        Returns:
            The initial state.
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state: AbstractState) -> None:
        """Initial state setter. The initial state can only be set as long as the state machine is not running.

        Args:
            state: State instance.

        Raises:
            RuntimeError: If used while the state machine is running.
        """
        if self.context.is_running():
            raise RuntimeError("Failed to set the initial state: State machine is already running.")

        self._current_state = state
        self._initial_state = state

    def run(self) -> None:
        """Starts the state machine. The state machine can be stopped by setting the running flag inside the context."""
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
        """Add a state transition.

        Args:
            transition_name: Name of the transition.
            state: Target state. It is loaded as the next state if another state returns 'transition_name'.
            no_raise: Disables raising an error if the transition name already exists. Defaults to False.

        Raises:
            ValueError: If the 'transition_name' already exists in the transition table and 'no_raise' is set to False.
        """
        if transition_name in self._transition_table and not no_raise:
            raise ValueError(
                f"""Failed to add state '{state.name}': Transition '{transition_name}' already has a target state """
                f""" ('{self._transition_table[transition_name].name}'). To override existing transitions use the """
                """'no_raise' flag."""
            )

        self._transition_table[transition_name] = state

    def pop(self, transition_name: str, no_raise: bool = True) -> AbstractState | None:
        """Removes a transition from the transition table.

        Args:
            transition_name: The transition to remove
            no_raise: Disables raising an error if 'tranisition_name' is unknown. Defaults to True.

        Raises:
            KeyError: If 'transition_name' is unknown and 'no_raise' is set to False.

        Returns:
            The target state of the removed transition or None if the transition is unknown.
        """
        if transition_name in self._transition_table:
            return self._transition_table.pop(transition_name)

        if no_raise:
            return None

        raise KeyError(
            f"""Failed to remove transition: '{transition_name}' does not exist. Use the 'no_raise' flag """
            """to suppress this exception."""
        )

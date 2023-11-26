import logging
import traceback
from pathlib import Path
from typing import Any, Generic, Literal

from pygb._impl._core._abstract_context import AbstractContext
from pygb._impl._core._abstract_state import AbstractState, ContextType

try:
    import pydot

    HAS_PYDOT = True
except ImportError:
    HAS_PYDOT = False

_logger = logging.getLogger(__name__)


class StateMachine(Generic[ContextType]):
    """State machine class."""

    def __init__(self, context: ContextType, initial_state: AbstractState | None = None) -> None:
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
    def current_state(self) -> AbstractState:
        """Current state property. Is updated with each state machine cycle.

        Returns:
            The currently active state.
        """
        if self._current_state is None:
            raise RuntimeError(
                """Failed to retrieve current state: Current state is unset. Set one by specifying an initial state """
                """ when setting up the state machine."""
            )

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
        """Starts the state machine. The state machine can be stopped by setting the running flag inside the context.

        Raises:
            RuntimeError: If no initial state is set.
            RuntimeError: If a state returns an unknown (i.e. unregistered) transition.
        """
        _logger.info("Starting state machine")

        if self._initial_state is None:
            raise RuntimeError("Failed to start state machine: No initial state set.")

        self.context.set_running()
        transition = None

        while self.context.is_running():
            try:
                _logger.debug("Executing transition: %s -> %s" % (transition or "-", self.current_state.name))
                transition = self.current_state()

                if transition is None:
                    break

                if transition not in self._transition_table:
                    raise RuntimeError(
                        f"""State Machine failure: State '{self.current_state.name}' returned unknown transition """
                        f"""'{transition}'."""
                    )

                self._current_state = self._transition_table[transition]
            except Exception as error:
                _logger.error("State machine stopped due to error: %s" % error)
                _logger.error("Full traceback: %s" % traceback.format_exc())
                break

        _logger.info("State machine stopped")

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
        _logger.info("Added transition: %s->%s" % (transition_name, state.name))

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
            state = self._transition_table.pop(transition_name)
            _logger.info("Removed transition: %s->%s" % (transition_name, state.name))
            return state

        if no_raise:
            return None

        raise KeyError(
            f"""Failed to remove transition: '{transition_name}' does not exist. Use the 'no_raise' flag """
            """to suppress this exception."""
        )

    def graph(
        self,
        directory: Path,
        type_: Literal["svg", "png"] = "png",
        name: str = "goal-babbling-state-machine",
        no_raise: bool = False,
    ) -> None:
        """Plots the current state machine graph and saves it to the specified directory.

        Note: This method requires the 'pydot' Python package ('pip install pydot') as well as the 'graphviz' tool
        installed on your system.

        Args:
            directory: Target directory.
            type_: File type (either "png" or "svg"). Defaults to "png".
            name: File name (excluding file ending). Defaults to "goal-babbling-state-machine".
            no_raise: Wether or not to raise an exception if 'pydot' is not installed. Defaults to False.

        Raises:
            ImportError: If the 'pydot' package is missing.
        """
        if HAS_PYDOT:
            graph = self._generate_graph(name)

            if type_ == "png":
                graph.write_png(directory.joinpath(f"{name}.png"))
            else:
                graph.write_svg(directory.joinpath(f"{name}.svg"))

            return

        msg = (
            "Failed to plot state machine graph: Python package 'pydot' not found. Install it via 'pip install pydot'."
        )

        if no_raise:
            _logger.warning(msg)
            return

        raise ImportError(msg)

    def _generate_graph(self, name: str) -> "pydot.Dot":
        if self.initial_state is None:
            raise RuntimeError("Failed to plot graph: No initial state set.")

        graph = pydot.Dot(name, graph_type="digraph")
        states = list(self._transition_table.values())

        if self.initial_state not in states:  # we assert that initial_state is set in the calling method
            states.append(self.initial_state)

        for state in states:
            node = (
                pydot.Node(state.name, shape="rectangle", color="green", style="filled, rounded")
                if state == self.initial_state
                else pydot.Node(state.name, shape="rectangle", style="rounded")
            )
            graph.add_node(node)

        transitions = set()
        for transition, target_state in self._transition_table.items():
            for source_name in [s.name for s in states if transition in s.transitions()]:
                transitions.add((source_name, target_state.name, transition))

        for t in transitions:
            graph.add_edge(pydot.Edge(t[0], t[1], label=t[2]))

        return graph

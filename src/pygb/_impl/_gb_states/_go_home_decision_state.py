import numpy as np

from pygb._impl._core._abstract_state import AbstractState
from pygb._impl._core._context import GoalBabblingContext
from pygb._impl._core._events import EventSystem
from pygb._impl._core._runtime_data import ActionSequence


class GoHomeDecisionState(AbstractState[GoalBabblingContext]):
    """Class which represents the state machine state before a new sequence is started. Decides whether to return to
    home or to generate a sequence to a new goal."""

    go_home: str = "go_home"
    generate_sequence: str = "generate_sequence"

    def __init__(
        self,
        context: GoalBabblingContext,
        event_system: EventSystem | None = None,
        name: str | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Constructor.

        Args:
            context: Goal Babbling context
            event_system: Event system singleton instance. Defaults to None.
            name: State name. If not specified, the class name is chosen as a default name. Defaults to None.
            random_seed: Random seed which is used to initialize a numpy random number generator. Defaults to None.
        """
        super().__init__(context, event_system, name)

        self._rng = np.random.default_rng(seed=random_seed)

    def __call__(self) -> str | None:
        """Execute the state.

        Steps:
            1) Return a 'go_home' transition in P% of cases, where P is specified in the current epoch set's parameter
                set. Otherwise return 'generate_sequence'

        Returns:
            Transition name.
        """
        if isinstance(self.context.runtime_data.previous_sequence, ActionSequence):
            return GoHomeDecisionState.generate_sequence

        if self._rng.random() <= self.context.current_parameters.go_home_chance:
            return GoHomeDecisionState.go_home

        return GoHomeDecisionState.generate_sequence

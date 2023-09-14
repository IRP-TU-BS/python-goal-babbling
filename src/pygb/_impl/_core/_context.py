from pygb._impl._core._abstract_context import AbstractContext
from pygb._impl._core._abstract_estimate_cache import AbstractEstimateCache
from pygb._impl._core._epoch_set_record import EpochSetRecord
from pygb._impl._core._goals import GoalSet, GoalStore
from pygb._impl._core._model import AbstractForwardModel, AbstractInverseEstimate
from pygb._impl._core._parameters import GBParameters, GBParameterStore
from pygb._impl._core._runtime_data import RuntimeData


class GoalBabblingContext(AbstractContext):
    def __init__(
        self,
        param_store: GBParameterStore,
        goal_store: GoalStore,
        forward_model: AbstractForwardModel,
        inverse_estimate: AbstractInverseEstimate,
        estimate_cache: AbstractEstimateCache | None = None,
        runtime_data: RuntimeData = RuntimeData(),
    ) -> None:
        self.gb_param_store = param_store
        self.goal_store = goal_store
        self.runtime_data = runtime_data
        self.forward_model = forward_model
        self.inverse_estimate = inverse_estimate
        self.estimate_cache = estimate_cache

        self.running = False
        self.epoch_set_records: list[EpochSetRecord] = []

    @property
    def num_epoch_sets(self) -> int:
        return len(self.goal_store)

    @property
    def current_goal_set(self) -> GoalSet:
        """Return the current epoch set's goal set.

        Returns:
            Current epoch set's goal set.
        """
        return self.goal_store[self.runtime_data.epoch_set_index]

    @property
    def current_parameters(self) -> GBParameters:
        """Returns the current epoch set's goal babbling paramters.

        Returns:
            Current epoch set's goal babbling paramters.
        """
        return self.gb_param_store[self.runtime_data.epoch_set_index]

    def is_running(self) -> bool:
        return self.running

    def set_running(self) -> None:
        self.running = True

    def set_stopped(self) -> None:
        self.running = False

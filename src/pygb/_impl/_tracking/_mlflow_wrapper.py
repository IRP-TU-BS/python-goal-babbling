import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import mlflow
import numpy as np

from pygb import GoalBabblingContext


class MLFlowWrapper:
    """Wrapper for the MLFlow Python (fluent) library. Provides convenience functions and callbacks to log training
    progress.

    Usage: Wrap your training state machine in a context manager:

        >>> state_machine = StateMachine(my_context)
        >>> wrapper = MLFlowWrapper("my_experiment", "my_parent_run")
        >>> with wrapper:
        ...     state_machine.run()

    This way the parent run is properly initialized.
    """

    def __init__(
        self, experiment_name: str, parent_run: str, directory: Path | None = None, parent_run_description: str = "-"
    ) -> None:
        self.experiment_name = experiment_name
        self.parent_run = parent_run
        self.parent_run_description = parent_run_description

        if directory is None:
            mlflow.set_tracking_uri(directory)

        self._active_run: mlflow.ActiveRun | None = None
        self._parent_run: mlflow.ActiveRun | None = None

    def __enter__(self) -> None:
        self._parent_run = mlflow.start_run(
            experiment_id=self.experiment_id, run_name=self.parent_run, description=self.parent_run_description
        )

    def __exit__(self, *args: tuple, **kwargs: dict) -> None:
        while mlflow.active_run() is not None:
            mlflow.end_run()

    @property
    def experiment_id(self) -> str:
        _experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if _experiment is None:
            _experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            _experiment_id = _experiment.experiment_id

        return _experiment_id

    def epoch_complete_callback(self, context: GoalBabblingContext) -> None:
        """Callback which logs parameters (fixed, e.g. hyper-parameters) and training metrics at the end of an epoch.

        Logging is structured in the following way:

            - runs are logged as part of a parent experiment
            - each run is composed of mutliple sub-runs, one for each epoch set
            - parameters are logged the first time this function is called, while metrics are logged wth every call
            - whenever a new epoch set is detected, the currently active run is ended and a new one is started

        This function is supposed to be used as a callback used with the event system.

        Args:
            context: Goal Babbling context.
        """
        if self._active_run is None:
            self._active_run = mlflow.start_run(
                run_name=f"epochSet{context.runtime_data.epoch_set_index}",
                experiment_id=self.experiment_id,
                nested=True,
            )
            self._log_epoch_set_statics(context, epoch_set_index=0)

        mlflow.log_metrics(context.runtime_data.metrics(), step=context.runtime_data.epoch_index)
        mlflow.log_metrics(context.forward_model.metrics(), step=context.runtime_data.epoch_index)
        mlflow.log_metrics(context.inverse_estimate.metrics(), step=context.runtime_data.epoch_index)

    def epoch_set_complete_callback(self, context: GoalBabblingContext) -> None:
        """Ends the currently active run (for an epoch set) and starts a new run. Logs the best model from the epoch set
        and logs static data of the upcoming epoch set.

        Args:
            context: Goal Babbling context.
        """
        model = context.model_store.load(epoch_set_index=context.runtime_data.epoch_set_index)
        self.log_pickle(model, name=f"best_llm_es{context.runtime_data.epoch_set_index}_pickle")

        if self._active_run is not None:
            mlflow.end_run()

        if context.num_epoch_sets - 1 > context.runtime_data.epoch_set_index:
            next_epoch_set = context.runtime_data.epoch_set_index + 1

            self._active_run = mlflow.start_run(
                run_name=f"epochSet{next_epoch_set}",
                experiment_id=self.experiment_id,
                nested=True,
            )

            self._log_epoch_set_statics(context, next_epoch_set)

    def log_numpy_array(self, data: np.ndarray, name: str) -> None:
        with NamedTemporaryFile(prefix=name, suffix=".csv") as named_file:
            path = Path(named_file.name)

            with open(path, mode="w") as file:
                np.savetxt(file, data, delimiter=",")

            mlflow.log_artifact(path)

    def log_pickle(self, obj: Any, name: str) -> None:
        with NamedTemporaryFile(prefix=name) as named_file:
            path = Path(named_file.name)

            with open(path, mode="wb") as file:
                pickle.dump(obj, file)

            mlflow.log_artifact(path)

    def _log_epoch_set_statics(self, context: GoalBabblingContext, epoch_set_index: int) -> None:
        self.log_numpy_array(context.goal_store[epoch_set_index].train, name="train_goals")
        self.log_numpy_array(context.goal_store[epoch_set_index].test, name="test_goals")
        mlflow.log_params(context.gb_param_store[epoch_set_index].parameters())
        mlflow.log_params(context.gb_param_store[epoch_set_index].parameters())
        mlflow.log_params(context.forward_model.parameters())
        mlflow.log_params(context.inverse_estimate.parameters())

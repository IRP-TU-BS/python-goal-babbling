from pathlib import Path
from tempfile import NamedTemporaryFile

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

    def __init__(self, experiment_name: str, parent_run: str, directory: Path | None = None) -> None:
        self.experiment_name = experiment_name
        self.parent_run = parent_run

        if directory is None:
            mlflow.set_tracking_uri(directory)

        self._current_epoch_set: int | None = None
        self._active_run: mlflow.ActiveRun | None = None
        self._parent_run: mlflow.ActiveRun | None = None

    def __enter__(self) -> None:
        self._parent_run = mlflow.start_run(experiment_id=self.experiment_id, run_name=self.parent_run)

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

    def log(self, context: GoalBabblingContext) -> None:
        """Callback which logs parameters (fixed, e.g. hyper-parameters) and training metrics.

        Logging is structured in the following way:

            - runs are logged as part of a parent experiment
            - each run is composed of mutliple sub-runs, one for each epoch set
            - parameters are logged the first time this function is called, while metrics are logged wth every call
            - whenever a new epoch set is detected, the currently active run is ended and a new one is started

        This function is supposed to be used as a callback used with the event system.

        Args:
            context: Goal Babbling context.
        """
        if self._active_run is None or self._current_epoch_set != context.runtime_data.epoch_set_index:
            if self._active_run is not None and self._current_epoch_set != context.runtime_data.epoch_set_index:
                mlflow.end_run()

            self._current_epoch_set = context.runtime_data.epoch_set_index
            self._active_run = mlflow.start_run(
                run_name=f"epochSet{self._current_epoch_set}", experiment_id=self.experiment_id, nested=True
            )

            self.log_numpy_array(context.current_goal_set.train, name="train_goals")
            self.log_numpy_array(context.current_goal_set.test, name="test_goals")
            mlflow.log_params(context.current_parameters.parameters())
            mlflow.log_params(context.current_goal_set.parameters())
            mlflow.log_params(context.forward_model.parameters())
            mlflow.log_params(context.inverse_estimate.parameters())

        mlflow.log_metrics(context.runtime_data.metrics(), step=context.runtime_data.epoch_index)
        mlflow.log_metrics(context.forward_model.metrics(), step=context.runtime_data.epoch_index)
        mlflow.log_metrics(context.inverse_estimate.metrics(), step=context.runtime_data.epoch_index)

    def log_numpy_array(self, data: np.ndarray, name: str) -> None:
        with NamedTemporaryFile(prefix=name, suffix=".csv") as named_file:
            path = Path(named_file.name)

            with open(path, mode="w") as file:
                np.savetxt(file, data, delimiter=",")

            mlflow.log_artifact(path)

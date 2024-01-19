from pygb._impl._core._context import GoalBabblingContext


def epoch_complete_callback(context: GoalBabblingContext) -> None:
    s = "---------------------------\n"
    s += f"Epoch set {context.runtime_data.epoch_set_index} | Epoch {context.runtime_data.epoch_index}\n"

    for heading, metrics in zip(
        ("Performance metrics", "Forward model metrics", "Inverse estimate metrics"),
        (context.runtime_data.metrics(), context.forward_model.metrics(), context.inverse_estimate.metrics()),
    ):
        s += f"\t {heading}\n"
        for label, value in metrics.items():
            s += f"\t\t {label}: {value}\n"

    s += "---------------------------\n\n"

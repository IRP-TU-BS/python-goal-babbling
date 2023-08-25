from typing import Any

import pytest

from pygb import RuntimeData


def test_runtime_data_init() -> None:
    runtime_data = RuntimeData()

    assert runtime_data.current_sequence is None
    assert runtime_data.previous_sequence is None
    assert runtime_data.performance_error is None
    assert len(runtime_data.sequences) == 0
    assert runtime_data.observation_index == 0


@pytest.mark.parametrize(
    ("runtime_data", "expected_metrics"),
    [
        (
            RuntimeData(performance_error=0.5, opt_performance_errors={"foo": 1.0, "bar": 2.0}),
            {"performance_error": 0.5, "foo_performance_error": 1.0, "bar_performance_error": 2.0},
        ),
        (
            RuntimeData(performance_error=0.5),
            {"performance_error": 0.5},
        ),
    ],
)
def test_runtime_data_metrics(runtime_data: RuntimeData, expected_metrics: dict[str, Any]) -> None:
    assert runtime_data.metrics() == expected_metrics

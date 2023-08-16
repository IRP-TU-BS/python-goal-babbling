from pygb import ObservationSequence, RuntimeData


def test_runtime_data_init() -> None:
    runtime_data = RuntimeData()

    assert runtime_data.current_sequence is None
    assert runtime_data.previous_sequence is None
    assert runtime_data.performance_error is None
    assert len(runtime_data.sequences) == 0
    assert runtime_data.observation_index == 0

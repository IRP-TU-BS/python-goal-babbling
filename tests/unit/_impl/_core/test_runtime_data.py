from pygb import RuntimeData, SequenceData


def test_runtime_data_init() -> None:
    runtime_data = RuntimeData()

    assert runtime_data.current_sequence is None
    assert runtime_data.previous_sequence is None
    assert runtime_data.performance_error is None
    assert len(runtime_data.sequences) == 0
    assert runtime_data.observation_index == 0


def test_runtime_data_update_current_sequence() -> None:
    runtime_data = RuntimeData()
    sequence = SequenceData(0, 1)

    runtime_data.update_current_sequence(sequence)

    assert runtime_data.previous_sequence is None
    assert runtime_data.current_sequence == sequence
    assert runtime_data.sequences == []

    sequence2 = SequenceData(1, 3)
    runtime_data.update_current_sequence(sequence2)

    assert runtime_data.previous_sequence == sequence
    assert runtime_data.current_sequence == sequence2
    assert runtime_data.sequences == []

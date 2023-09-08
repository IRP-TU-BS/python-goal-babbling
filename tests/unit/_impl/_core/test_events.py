from pygb import Events


def test_default_events() -> None:
    assert Events.SEQUENCE_FINISHED == "sequence-finished"
    assert Events.EPOCH_COMPLETE == "epoch-complete"
    assert Events.EPOCH_SET_COMPLETE == "epoch-set-complete"

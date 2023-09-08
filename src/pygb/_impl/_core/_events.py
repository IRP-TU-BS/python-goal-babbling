from enum import Enum


class Events(str, Enum):
    SEQUENCE_FINISHED = "sequence-finished"
    EPOCH_COMPLETE = "epoch-complete"
    EPOCH_SET_COMPLETE = "epoch-set-complete"

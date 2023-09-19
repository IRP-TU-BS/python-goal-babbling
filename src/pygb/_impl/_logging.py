import logging
import sys
from pathlib import Path

DEFAULT_LOG_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"


def setup_logging(
    log_level: int = logging.INFO,
    log_format: str = DEFAULT_LOG_FORMAT,
    log_file: Path | None = None,
    log_file_level: int = logging.DEBUG,
) -> None:
    """Sets up a StreamHandler for logs to stderr and an optional FileHandler.

    Args:
        log_level: The StreamHandler's log level. Defaults to logging.INFO.
        log_format: Global log format. Defaults to DEFAULT_LOG_FORMAT.
        log_file: Path to a log file. Defaults to None.
        log_file_level: The FileHandler's log level. Defaults to logging.DEBUG.
    """
    logging.getLogger().setLevel(logging.DEBUG)

    formatter = logging.Formatter(log_format)

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(logging.Formatter(log_format))

    logging.getLogger().addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(funcName)s][%(levelname)s] %(message)s"))
        file_handler.setLevel(log_file_level)
        logging.getLogger().addHandler(file_handler)

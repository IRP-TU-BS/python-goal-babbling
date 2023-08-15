import logging
from pathlib import Path

DEFAULT_LOG_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"


def setup_logging(
    log_level: int = logging.INFO,
    log_format: str = DEFAULT_LOG_FORMAT,
    log_file: Path | None = None,
) -> None:
    logging.basicConfig(level=log_level, format=log_format)
    logging.getLogger().setLevel(log_level)

    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

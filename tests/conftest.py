from typing import Generator
from unittest.mock import patch

import pytest

from pygb import EventSystem


@pytest.fixture(autouse=True)
def mock_event_system() -> Generator[None, None, None]:
    """Mocks the event system.

    The unit test using this fixture has a guaranteed fresh EventSystem which is globally accessible via
    EventSystem.instance().

    Yields:
        Nothing.
    """
    with patch("pygb._impl._core._events.EventSystem.instance") as mock:
        mock.return_value = EventSystem()
        yield

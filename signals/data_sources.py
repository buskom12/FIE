import os
import sys

_FIE_PATH = os.path.join(os.path.dirname(__file__), "..", "future-intelligence-engine")
if _FIE_PATH not in sys.path:
    sys.path.insert(0, os.path.abspath(_FIE_PATH))

from events.event_discovery import discover_events  # noqa: E402


def collect_text_stream() -> list:
    return discover_events()

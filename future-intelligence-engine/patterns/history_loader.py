"""
History Loader — загружает данные из data.loader в PatternEngine.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from data.loader import load_historical_data

if TYPE_CHECKING:
    from patterns.pattern_engine import PatternEngine


def load_history(engine: "PatternEngine", path: Optional[Path] = None) -> int:
    """
    Загружает исторические кейсы в PatternEngine.

    Возвращает количество загруженных кейсов.
    """
    cases = load_historical_data(path)
    engine.add_cases(cases)
    return len(cases)


def load_raw(path: Optional[Path] = None) -> list:
    """Возвращает сырые данные без загрузки в движок."""
    return load_historical_data(path)

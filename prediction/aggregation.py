import sys
import os

_FIE_PATH = os.path.join(os.path.dirname(__file__), "..", "future-intelligence-engine")
if _FIE_PATH not in sys.path:
    sys.path.insert(0, os.path.abspath(_FIE_PATH))

from prediction.aggregation import aggregate_predictions as _aggregate  # noqa: E402


def aggregate_predictions(agent_results: list[dict]) -> dict:
    """
    Агрегирует результаты агентов.

    Возвращает:
        probability  — итоговая взвешенная вероятность
        confidence   — уверенность (1 - разброс мнений)
        agents_count — количество агентов
    """
    result = _aggregate(agent_results)
    return {
        "probability": result["final_probability"],
        "confidence": result["confidence"],
        "agents_count": result["agents_count"],
    }

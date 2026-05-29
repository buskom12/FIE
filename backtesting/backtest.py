"""
Упрощённый бэктест на основе паттернов: find_patterns → compute_probability → предсказание.
Используется для тестирования PatternEngine отдельно от полного пайплайна FIE.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Бэктест
# ---------------------------------------------------------------------------

def run_backtest(
    engine,
    dataset: list[dict],
    threshold: float = 0.5,
) -> list[dict]:
    """
    Прогоняет dataset через engine на основе паттернов.

    Каждый элемент dataset должен содержать:
        - "signals"  : данные для поиска паттернов
        - "outcome"  : фактический исход (0 или 1)

    Args:
        threshold: Порог вероятности для бинарного предсказания (по умолчанию 0.5).

    Возвращает список словарей {"predicted", "actual", "probability"}.
    """
    if not dataset:
        return []

    prior = getattr(engine, "prior", 0.5)
    results = []
    for case in dataset:
        market_context = case.get("market_context")
        matches = engine.find_patterns(case["signals"], market_context=market_context)
        hybrid = engine.compute_final_probability(
            case["signals"],
            matches=matches,
            market_context=market_context,
        )
        prob = hybrid["final_probability"]
        effective_prob = prob if prob is not None else prior
        results.append({
            "predicted": 1 if effective_prob > threshold else 0,
            "actual": case["outcome"],
            "probability": effective_prob,
            "has_signal": prob is not None,
            "pattern_probability": hybrid["pattern_probability"],
            "signal_score": hybrid["signal_score"],
        })

    return results


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def calculate_accuracy(results: list[dict]) -> float:
    """Доля правильных предсказаний."""
    if not results:
        raise ValueError("Список результатов пуст — невозможно вычислить точность.")
    correct = sum(1 for r in results if r["predicted"] == r["actual"])
    return correct / len(results)


def calculate_metrics(results: list[dict]) -> dict:
    """
    Расширенный набор метрик:
        accuracy  — общая точность
        precision — точность положительных предсказаний
        recall    — полнота (чувствительность)
        f1        — гармоническое среднее precision и recall
    """
    if not results:
        raise ValueError("Список результатов пуст.")

    tp = sum(1 for r in results if r["predicted"] == 1 and r["actual"] == 1)
    fp = sum(1 for r in results if r["predicted"] == 1 and r["actual"] == 0)
    fn = sum(1 for r in results if r["predicted"] == 0 and r["actual"] == 1)

    accuracy = calculate_accuracy(results)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total": len(results),
        "correct": sum(1 for r in results if r["predicted"] == r["actual"]),
    }

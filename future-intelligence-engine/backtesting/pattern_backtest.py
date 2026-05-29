"""
Pattern Backtest — Leave-One-Out Loop

Для каждого события из истории:
  1. Обучаем PatternEngine на всех остальных кейсах (исключая тестовый)
  2. Прогоняем тестовый кейс через движок
  3. Сравниваем предсказание с реальным исходом

Leave-One-Out гарантирует честную оценку без утечки данных.
"""

from patterns.pattern_engine import PatternEngine
from patterns.history_loader import load_raw


def run_pattern_backtest(threshold: float = 0.5) -> dict:
    """
    Запускает leave-one-out бэктест на data/historical_events.json.

    Args:
        threshold: порог вероятности для бинарной классификации (default 0.5)

    Returns:
        Словарь с метриками: accuracy, mean_error, results
    """
    all_cases = load_raw()
    results = []

    for i, test_case in enumerate(all_cases):
        train_cases = [c for j, c in enumerate(all_cases) if j != i]

        engine = PatternEngine()
        engine.add_cases(train_cases)

        analysis = engine.analyze(test_case["signals"])
        predicted_prob = analysis["probability"]
        predicted_outcome = int(predicted_prob >= threshold)
        real_outcome = test_case["outcome"]

        error = abs(predicted_prob - real_outcome)
        correct = predicted_outcome == real_outcome

        results.append({
            "event": test_case["event"],
            "signals": [s["type"] for s in test_case["signals"]],
            "predicted_prob": round(predicted_prob, 4),
            "predicted_outcome": predicted_outcome,
            "real_outcome": real_outcome,
            "error": round(error, 4),
            "correct": correct,
            "matched_cases": analysis["matched_cases"],
            "confidence": analysis["confidence"],
        })

    accuracy = sum(r["correct"] for r in results) / len(results)
    mean_error = sum(r["error"] for r in results) / len(results)

    _print_report(results, accuracy, mean_error)

    return {
        "accuracy": round(accuracy, 4),
        "mean_error": round(mean_error, 4),
        "total_events": len(results),
        "results": results,
    }


def _print_report(results: list[dict], accuracy: float, mean_error: float) -> None:
    print("\n" + "=" * 60)
    print("PATTERN BACKTEST — LEAVE-ONE-OUT")
    print("=" * 60)

    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(
            f"{status} [{r['confidence']:6}] "
            f"P={r['predicted_prob']:.2f} | Real={r['real_outcome']} | "
            f"Matches={r['matched_cases']} | {r['event']}"
        )

    print("=" * 60)
    print(f"Events tested : {len(results)}")
    print(f"Accuracy      : {accuracy:.1%}")
    print(f"Mean Error    : {mean_error:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_pattern_backtest()

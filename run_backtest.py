"""
Robust Backtesting — полный цикл с проверкой на переобучение.

Схема:
  1. Загрузка + shuffle данных
  2. Train 70% / Validation 10% / Test 20%
  3. Обучение PatternEngine на train
  4. Подбор порога на validation
  5. Финальная оценка на test
  6. Проверка на переобучение: train_accuracy - test_accuracy > 0.10 → OVERFITTING
  7. Полный отчёт метрик (Brier, ROC AUC, ECE, Calibration)
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.datasets.builder import build_dataset
from data.loader import load_historical_data
from patterns.pattern_engine import PatternEngine
from backtesting.backtest import run_backtest
from backtesting.metrics import full_report, calibration, brier_score, roc_auc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split(data: list, train: float = 0.7, val: float = 0.1) -> tuple:
    """Разбивает данные на train / val / test."""
    n = len(data)
    t = int(n * train)
    v = int(n * (train + val))
    return data[:t], data[t:v], data[v:]


def _find_best_threshold(engine: PatternEngine, val_data: list) -> float:
    """Подбирает порог вероятности на validation-сете по максимуму F1."""
    best_threshold, best_f1 = 0.5, 0.0
    for threshold in [i / 20 for i in range(3, 18)]:  # 0.15 … 0.85
        results = run_backtest(engine, val_data, threshold=threshold)
        if not results:
            continue
        tp = sum(1 for r in results if r["predicted"] == 1 and r["actual"] == 1)
        fp = sum(1 for r in results if r["predicted"] == 1 and r["actual"] == 0)
        fn = sum(1 for r in results if r["predicted"] == 0 and r["actual"] == 1)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    return best_threshold


def _print_section(title: str) -> None:
    width = 56
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _verdict(train_acc: float, test_acc: float) -> str:
    gap = train_acc - test_acc
    if gap > 0.10:
        return f"ПЕРЕОБУЧЕНИЕ (gap={gap:.2%}) — модель учит шум, не паттерн"
    if test_acc >= 0.65:
        return f"EDGE НАЙДЕН  (gap={gap:.2%}) — система работает"
    if test_acc >= 0.60:
        return f"ИНТЕРЕСНО    (gap={gap:.2%}) — нужно больше/лучших данных"
    return f"ШУМ          (gap={gap:.2%}) — улучшить сигналы или паттерны"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Загрузка (приоритет: реальный market dataset; fallback: historical synthetic)
    try:
        horizon = int(os.environ.get("FIE_HORIZON", "1") or "1")
        dataset = build_dataset(horizon=horizon)
        print(f"Загружено market-кейсов : {len(dataset)}")
    except Exception as exc:
        print(f"Не удалось собрать market dataset ({exc}). Использую historical_events.json")
        dataset = load_historical_data("data/historical_events.json")
        print(f"Загружено событий : {len(dataset)}")

    # 2. Split 70 / 10 / 20
    train_data, val_data, test_data = _split(dataset)
    print(f"Train: {len(train_data)}  |  Val: {len(val_data)}  |  Test: {len(test_data)}")

    # 3. Обучение
    engine = PatternEngine(similarity_threshold=0.3)
    for case in train_data:
        engine.add_case(
            case["signals"],
            case["outcome"],
            market_context=case.get("market_context"),
        )
    print(f"PatternEngine обучен: {engine}")

    # 4. Подбор порога на validation
    best_threshold = _find_best_threshold(engine, val_data)
    print(f"Лучший порог (по Val F1): {best_threshold:.2f}")

    # 5. Train-метрики (для проверки переобучения)
    train_results = run_backtest(engine, train_data, threshold=best_threshold)
    train_report = full_report(train_results)

    # 6. Test-метрики
    test_results = run_backtest(engine, test_data, threshold=best_threshold)
    if not test_results:
        print("Нет результатов на test-сете.")
        return
    test_report = full_report(test_results)

    # ---------------------------------------------------------------------------
    # Вывод: Train vs Test — проверка на переобучение
    # ---------------------------------------------------------------------------
    _print_section("TRAIN vs TEST — ПРОВЕРКА НА ПЕРЕОБУЧЕНИЕ")
    print(f"  {'Метрика':<22} {'Train':>10} {'Test':>10}  {'Delta':>8}")
    print("-" * 56)
    for key in ("accuracy", "precision", "recall", "f1", "brier_score", "roc_auc"):
        tr = train_report[key]
        te = test_report[key]
        if isinstance(tr, float) and isinstance(te, float):
            delta = tr - te
            flag = "  ⚠" if abs(delta) > 0.10 else ""
            print(f"  {key:<22} {tr:>10.4f} {te:>10.4f}  {delta:>+8.4f}{flag}")
        else:
            print(f"  {key:<22} {str(tr):>10} {str(te):>10}")

    # ---------------------------------------------------------------------------
    # Полный отчёт по Test
    # ---------------------------------------------------------------------------
    _print_section("ПОЛНЫЙ ОТЧЁТ — TEST SET")
    print(f"  Всего предсказаний : {test_report['total']}")
    print(f"  Правильных         : {test_report['correct']}")
    print(f"  Accuracy           : {test_report['accuracy']:.2%}")
    print(f"  Precision          : {test_report['precision']:.2%}")
    print(f"  Recall             : {test_report['recall']:.2%}")
    print(f"  F1                 : {test_report['f1']:.2%}")
    print(f"  Brier Score        : {test_report['brier_score']:.4f}  (0=идеал, 0.25=случай)")
    print(f"  Brier Skill Score  : {test_report['brier_skill_score']:.4f}  (>0 лучше случая)")
    print(f"  ROC AUC            : {test_report['roc_auc']}          (0.5=случай, 1.0=идеал)")
    print(f"  ECE (калибровка)   : {test_report['ece']:.4f}  (ближе к 0 — лучше)")

    # ---------------------------------------------------------------------------
    # Калибровочная таблица
    # ---------------------------------------------------------------------------
    _print_section("КАЛИБРОВКА — насколько вероятности честны?")
    probs = [r["probability"] for r in test_results]
    actuals = [r["actual"] for r in test_results]
    bins = calibration(probs, actuals, n_bins=5)
    print(f"  {'Bin':>6} {'Pred%':>8} {'Actual%':>9} {'Count':>7} {'Error':>8}")
    print("-" * 44)
    for b in bins:
        bar = "█" * int(b["calibration_error"] * 40)
        print(f"  {b['bin_center']:>6.0%} {b['mean_predicted']:>8.1%} "
              f"{b['mean_actual']:>9.1%} {b['count']:>7}  {b['calibration_error']:>6.4f} {bar}")

    # ---------------------------------------------------------------------------
    # Вердикт
    # ---------------------------------------------------------------------------
    _print_section("ВЕРДИКТ")
    verdict = _verdict(train_report["accuracy"], test_report["accuracy"])
    print(f"  {verdict}")
    print("=" * 56)


if __name__ == "__main__":
    main()

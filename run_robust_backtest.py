"""
Запуск Robust Backtesting с регуляризацией PatternEngine.

Режимы:
  1. Grid search по конфигурациям — ищем лучший баланс gap/accuracy
  2. Финальный запуск лучшей конфигурации с калибровкой
  3. K-Fold для проверки стабильности
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.datasets.builder import build_dataset
from data.loader import load_historical_data
from patterns.pattern_engine import PatternEngine, PatternEngineConfig
from backtesting.robust_backtest import run_robust_backtest, run_k_fold_backtest

W = 64


def _section(title: str) -> None:
    print(f"\n{'=' * W}\n  {title}\n{'=' * W}")


def _bar(v: float, w: int = 16) -> str:
    filled = int(max(0.0, min(1.0, v)) * w)
    return "█" * filled + "░" * (w - filled)


# ---------------------------------------------------------------------------
# 1. Grid Search
# ---------------------------------------------------------------------------

CONFIGS = [
    # label, PatternEngineConfig
    ("No reg (baseline)",
     PatternEngineConfig(similarity_threshold=0.3, laplace_alpha=0.0, min_matches=1,
                         confidence_threshold=1, top_k=None)),

    ("min=5",
     PatternEngineConfig(similarity_threshold=0.3, laplace_alpha=0.0, min_matches=5,
                         confidence_threshold=1, top_k=None)),

    ("min=5 + alpha=1",
     PatternEngineConfig(similarity_threshold=0.3, laplace_alpha=1.0, min_matches=5,
                         confidence_threshold=1, top_k=None)),

    ("min=5 + alpha=1 + conf=20",
     PatternEngineConfig(similarity_threshold=0.3, laplace_alpha=1.0, min_matches=5,
                         confidence_threshold=20, top_k=None)),

    ("min=5 + alpha=1 + conf=10",
     PatternEngineConfig(similarity_threshold=0.3, laplace_alpha=1.0, min_matches=5,
                         confidence_threshold=10, top_k=None)),

    ("min=3 + alpha=1 + conf=15",
     PatternEngineConfig(similarity_threshold=0.3, laplace_alpha=1.0, min_matches=3,
                         confidence_threshold=15, top_k=None)),

    ("FULL REG (рекоменд.)",
     PatternEngineConfig(similarity_threshold=0.3, laplace_alpha=1.0, min_matches=5,
                         confidence_threshold=20, top_k=20)),
]


def run_grid_search(dataset: list) -> PatternEngineConfig:
    _section("GRID SEARCH — поиск конфигурации с минимальным gap")
    print(f"\n  {'Конфигурация':<24} {'TrainAcc':>9} {'TestAcc':>9} {'Gap':>7} {'Brier':>7} {'AUC':>7}")
    print(f"  {'-' * 66}")

    best_cfg = None
    best_score = float("inf")  # минимизируем: gap + brier

    for label, cfg in CONFIGS:
        engine = PatternEngine(config=cfg)
        result = run_robust_backtest(
            engine, dataset,
            split_mode="walk_forward",
            calibration_method=None,
        )
        gap = result["overfitting_gap"]
        acc = result["accuracy"]
        bs = result["brier_score"]
        auc = result["roc_auc"]
        flag = "⚠" if result["is_overfit"] else "✓"

        print(f"  {label:<24} {result['train_accuracy']:>9.2%} {acc:>9.2%} "
              f"{gap:>+7.2%} {bs:>7.4f} {str(auc):>7}  {flag}")

        # Балансируем: хотим низкий gap И высокую точность
        score = gap * 2 + bs - acc
        if score < best_score and not result["is_overfit"]:
            best_score = score
            best_cfg = cfg

    if best_cfg is None:
        # Если всё переобучено — берём с минимальным gap
        best_cfg = CONFIGS[-1][1]
        print(f"\n  Все конфигурации переобучены. Берём наиболее регуляризованную.")
    else:
        label = next(l for l, c in CONFIGS if c is best_cfg)
        print(f"\n  Лучшая конфигурация: {label}")

    return best_cfg


# ---------------------------------------------------------------------------
# 2. Финальный отчёт
# ---------------------------------------------------------------------------

def print_final_report(result: dict) -> None:
    _section("ФИНАЛЬНЫЙ ОТЧЁТ — лучшая конфигурация + Platt Calibration")

    gap = result["overfitting_gap"]
    overfit_str = "⚠  ПЕРЕОБУЧЕНИЕ" if result["is_overfit"] else "✓  в норме"
    print(f"\n  OVERFITTING")
    print(f"  Train Accuracy : {result['train_accuracy']:.2%}")
    print(f"  Test  Accuracy : {result['raw_accuracy']:.2%}  (до калибровки)")
    print(f"  Gap            : {gap:+.2%}  {overfit_str}")

    if result.get("calibration"):
        cal = result["calibration"]
        print(f"\n  КАЛИБРОВКА  ({cal['method']})")
        print(f"  Brier  {cal['before']['brier_score']:.4f} → {cal['after']['brier_score']:.4f}  "
              f"(Δ {cal['improvement']['brier_delta']:+.4f})")
        print(f"  ECE    {cal['before']['ece']:.4f} → {cal['after']['ece']:.4f}  "
              f"(Δ {cal['improvement']['ece_delta']:+.4f})")

        print(f"\n  {'Bin':>5} {'Pred':>7} {'Actual':>8} {'Error':>8}")
        print(f"  {'-' * 34}")
        for b in result["calibration"]["bins_after"]:
            bar = "█" * int(b["calibration_error"] * 30)
            print(f"  {b['bin_center']:>5.0%} {b['mean_predicted']:>7.1%} "
                  f"{b['mean_actual']:>8.1%}  {b['calibration_error']:>6.4f} {bar}")

    # --- Confidence stats ---
    cov = result.get("signal_coverage", 1.0)
    mean_conf = result.get("mean_confidence", 1.0)
    hca = result.get("high_confidence_accuracy")
    print(f"\n  УВЕРЕННОСТЬ СИСТЕМЫ")
    print(f"  Signal coverage   : {cov:.1%}  (кейсов с ≥ min_matches совпадений)")
    print(f"  Mean confidence   : {mean_conf:.2f}  (0=нет данных, 1=полная уверенность)")
    if hca is not None:
        print(f"  High-conf accuracy: {hca:.2%}  (только кейсы с confidence ≥ 0.5)")

    print(f"\n  FIE vs BASELINE")
    for key, lb, label in [
        ("accuracy", False, "Accuracy"),
        ("brier_score", True, "Brier Score"),
        ("roc_auc", False, "ROC AUC"),
    ]:
        fie = result[key] if isinstance(result[key], float) else 0.5
        base_key = f"baseline_{key}" if key != "roc_auc" else "baseline_roc_auc"
        base = result.get(base_key, 0.5)
        if not isinstance(base, float):
            base = 0.5
        beats = (fie < base) if lb else (fie > base)
        icon = "✓" if beats else "✗"
        print(f"  {icon}  {label:<14} FIE={fie:.4f}   Base={base:.4f}   Δ={fie - base:+.4f}")

    print(f"\n  МЕТРИКИ  (Test, n={result['test_size']}, после калибровки)")
    for key, label in [("accuracy","Accuracy"),("precision","Precision"),
                       ("recall","Recall"),("f1","F1")]:
        v = result[key]
        print(f"  {label:<14} {v:.2%}  {_bar(v)}")
    print(f"  {'Brier Score':<14} {result['brier_score']:.4f}")
    print(f"  {'Brier Skill':<14} {result['brier_skill_score']:.4f}  (> 0 = лучше случая)")
    print(f"  {'ROC AUC':<14} {result['roc_auc']}  (> 0.5 = лучше случая)")
    print(f"  {'ECE':<14} {result['ece']:.4f}  (< 0.05 = честная калибровка)")

    acc = result["accuracy"]
    overfit = result["is_overfit"]
    beats = result["beats_baseline_accuracy"] and result["beats_baseline_brier"]

    if overfit and not beats:
        verdict = "ПЕРЕОБУЧЕНИЕ + нет edge → нужны новые сигналы"
    elif overfit:
        verdict = "EDGE ЕСТЬ, но переобучение → собрать больше данных"
    elif not beats:
        verdict = "ШУМ — не лучше монеты"
    elif acc >= 0.65:
        verdict = "EDGE НАЙДЕН — система работает"
    elif acc >= 0.60:
        verdict = "СЛАБЫЙ EDGE — продолжаем улучшать"
    else:
        verdict = "СЛАБЫЙ СИГНАЛ"

    print(f"\n  ВЕРДИКТ: {verdict}")
    print("=" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Приоритет: рыночные данные (OHLCV → реальные continuous signals). Fallback: synthetic historical events.
    horizons_env = os.environ.get("FIE_HORIZONS", "1,3,6,12")
    horizons: list[int] = []
    for part in horizons_env.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            horizons.append(int(part))
        except ValueError:
            continue
    horizons = sorted(set([h for h in horizons if h >= 1])) or [1]

    try:
        # Держим датасет компактным по умолчанию, чтобы прогон был быстрым.
        min_candles = int(os.environ.get("FIE_MIN_CANDLES", "4000") or "4000")
        dataset = build_dataset(horizon=horizons[0], horizons=horizons, min_candles=min_candles)
        print(f"Загружено market-кейсов: {len(dataset)}  (horizons={horizons})")
    except Exception as exc:
        print(f"Не удалось собрать market dataset ({exc}). Использую historical_events.json")
        dataset = load_historical_data("data/historical_events.json")
        print(f"Загружено событий: {len(dataset)}")

    # 1. Grid search (делаем один раз на базовом датасете)
    best_cfg = run_grid_search(dataset)

    # 2. Финальный запуск лучшей конфигурации для каждого горизонта
    # (Если dataset умеет outcomes — просто переписываем outcome под нужный horizon)
    _section("MULTI-HORIZON RESULTS  (AUC / Brier / ECE)")
    print(f"\n  {'H':>3} {'Thr':>5} {'AUC':>8} {'Brier':>8} {'ECE':>7} {'Acc':>7}")
    print(f"  {'-' * 46}")

    for h in horizons:
        if dataset and isinstance(dataset[0], dict) and "outcomes" in dataset[0]:
            ds_h = []
            for case in dataset:
                oc = case.get("outcomes", {})
                if isinstance(oc, dict) and h in oc:
                    cc = dict(case)
                    cc["outcome"] = int(oc[h])
                    cc["horizon"] = h
                    ds_h.append(cc)
            dataset_h = ds_h
        else:
            dataset_h = dataset

        engine = PatternEngine(config=best_cfg)
        result = run_robust_backtest(
            engine,
            dataset_h,
            split_mode="walk_forward",
            calibration_method="platt",
            threshold=None,            # обязателен подбор best_threshold
            optimize_threshold=True,
        )

        print(
            f"  {h:>3} {result['threshold']:>5.2f} "
            f"{str(result['roc_auc']):>8} {result['brier_score']:>8.4f} "
            f"{result['ece']:>7.4f} {result['accuracy']:>7.2%}"
        )

    # 3. K-Fold стабильность на лучшей конфигурации (только для первого горизонта, чтобы не удлинять прогон)
    def engine_factory():
        return PatternEngine(config=best_cfg)

    kfold = run_k_fold_backtest(engine_factory, dataset, k=5)

    _section(f"K-FOLD  (k={kfold['k']})  — стабильность модели (H={horizons[0]})")
    print(f"\n  {'Фолд':>5} {'Accuracy':>10} {'Brier':>8} {'AUC':>9}")
    print(f"  {'-' * 36}")
    for f in kfold["folds"]:
        print(f"  {f['fold']:>5} {f['accuracy']:>10.4f} {f['brier_score']:>8.4f} {str(f['roc_auc']):>9}")
    print(f"  {'-' * 36}")
    print(f"  Mean  {kfold['mean_accuracy']:>10.4f}  ± {kfold['std_accuracy']:.4f}")
    print(f"  Range [{kfold['min_accuracy']:.4f}, {kfold['max_accuracy']:.4f}]")
    stable = "✓ СТАБИЛЬНО" if kfold["is_stable"] else "⚠ НЕСТАБИЛЬНО"
    print(f"\n  {stable}")
    print("=" * W)


if __name__ == "__main__":
    main()

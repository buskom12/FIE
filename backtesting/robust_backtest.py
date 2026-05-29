"""
Robust Backtesting Engine.

Полный цикл:
    Train phase       → обучение PatternEngine + эволюция весов агентов
    Calibration phase → фитинг CalibrationEngine на train-вероятностях
    Test phase        → честная оценка на невиданных данных (с/без калибровки)
    Baseline          → сравнение с random (монетой)
    Overfitting       → детектирование gap train/test > порога

Поддерживает два режима split:
    "walk_forward" — хронологический (для рыночных данных, рекомендуется)
    "random"       — перемешанный (для i.i.d. данных)
"""

from __future__ import annotations

from backtesting.split import (
    walk_forward_split,
    train_test_split,
    random_baseline,
    k_fold_split,
    temporal_k_fold_split,
)
from backtesting.metrics import brier_score, roc_auc, full_report
from learning.evolution_engine import update_agent_weights
from learning.calibration_engine import CalibrationEngine


# ---------------------------------------------------------------------------
# Ядро бэктеста
# ---------------------------------------------------------------------------

def run_robust_backtest(
    engine,
    dataset: list[dict],
    agents: list[dict] | None = None,
    *,
    split_mode: str = "walk_forward",
    test_size: float = 0.2,
    threshold: float | None = None,
    optimize_threshold: bool = True,
    threshold_grid: list[float] | None = None,
    overfitting_threshold: float = 0.10,
    calibration_method: str | None = "platt",
    weight_mode: str = "batch",
    ema_alpha: float = 0.1,
) -> dict:
    """
    Запускает полный цикл robust backtesting с опциональной калибровкой.

    Args:
        engine:               PatternEngine с методами add_case / find_patterns / compute_probability
        dataset:              Список кейсов {"signals": [...], "outcome": int}
        agents:               Список dict-агентов {"probability", "weight"} для эволюции (опционально)
        split_mode:           "walk_forward" (рекомендуется) или "random"
        test_size:            Доля тестовых данных
        threshold:            Порог бинарного предсказания. Если None — будет подобран на val.
        optimize_threshold:   Если True и threshold=None — подбираем порог на val (части train)
        threshold_grid:       Сетка порогов (если None — дефолт 0.05..0.95)
        overfitting_threshold: Максимальный допустимый gap train-test accuracy
        calibration_method:   "platt" | "isotonic" | "binning" | None (без калибровки)
        weight_mode:          Шаг 6 — режим обучения весов сигналов:
                                "none"       — веса не обучаются (все сигналы равнозначны, baseline)
                                "batch"      — один раз после всего train-сета (быстро, стабильно)
                                "online_ema" — EMA после каждого кейса (адаптивно, медленнее сходится)
        ema_alpha:            Скорость EMA для weight_mode="online_ema" (0.1 = медленно и стабильно)

    Returns:
        Полный отчёт метрик + результаты калибровки + флаги качества.
    """
    if weight_mode not in ("batch", "online_ema", "none"):
        raise ValueError(f"Неизвестный weight_mode: '{weight_mode}'. Используй 'batch', 'online_ema' или 'none'.")

    if len(dataset) < 10:
        raise ValueError("Слишком мало данных для бэктеста (минимум 10 кейсов).")

    # ----- Split -----
    if split_mode == "walk_forward":
        train, test = walk_forward_split(dataset, split=1 - test_size)
    elif split_mode == "random":
        train, test = train_test_split(dataset, test_size=test_size)
    else:
        raise ValueError(f"Неизвестный split_mode: '{split_mode}'. Используй 'walk_forward' или 'random'.")

    # ----- Inner split: train_fit / val_threshold (честный подбор порога) -----
    if len(train) >= 20:
        cut = int(len(train) * 0.8)
        train_fit = train[:cut]
        val_thr = train[cut:]
    else:
        train_fit = train
        val_thr = []

    # ----- Train phase (только train_fit) -----
    _supports_online = hasattr(engine, "update_weights_online")
    _supports_batch  = hasattr(engine, "fit_signal_weights")

    for case in train_fit:
        engine.add_case(
            case["signals"],
            case["outcome"],
            market_context=case.get("market_context"),
        )

        # Шаг 6: online EMA — обновляем веса сразу после каждого кейса
        if weight_mode == "online_ema" and _supports_online:
            engine.update_weights_online(case, alpha=ema_alpha)

        if agents is not None:
            for agent in agents:
                matches = engine.find_patterns(
                    case["signals"],
                    market_context=case.get("market_context"),
                )
                agent["probability"] = engine.compute_probability(matches)
            update_agent_weights(agents, case["outcome"])

    # Шаг 6: batch — одно финальное обучение на всём train-сете (без утечки в test)
    # weight_mode="none" — пропускаем, веса остаются дефолтными (честный baseline)
    if weight_mode == "batch" and _supports_batch:
        engine.fit_signal_weights(train_fit)

    # ----- Threshold selection (на val_thr) -----
    if threshold is None:
        if not optimize_threshold or not val_thr:
            threshold = 0.5
        else:
            threshold = _find_best_threshold(engine, val_thr, threshold_grid=threshold_grid)

    # Сырые train-вероятности для калибровки и overfitting check (на train_fit)
    train_results_raw = _evaluate(engine, train_fit, threshold)
    train_report = full_report(train_results_raw)

    # ----- Calibration phase (фитинг на train) -----
    calibration_report = None
    calibrator = None
    if calibration_method is not None:
        train_probs = [r["probability"] for r in train_results_raw]
        train_outcomes = [r["actual"] for r in train_results_raw]
        calibrator = CalibrationEngine(method=calibration_method)
        calibrator.fit(train_probs, train_outcomes)

    # ----- Test phase (сырые вероятности) -----
    test_results_raw = _evaluate(engine, test, threshold)
    test_report_raw = full_report(test_results_raw)

    # ----- Test phase (калиброванные вероятности) -----
    if calibrator is not None:
        raw_test_probs = [r["probability"] for r in test_results_raw]
        test_outcomes = [r["actual"] for r in test_results_raw]
        cal_probs = calibrator.transform(raw_test_probs)
        test_results_cal = [
            {
                "predicted": 1 if p > threshold else 0,
                "actual": o,
                "probability": p,
            }
            for p, o in zip(cal_probs, test_outcomes)
        ]
        test_report_cal = full_report(test_results_cal)
        calibration_report = calibrator.calibration_report(raw_test_probs, test_outcomes)
    else:
        test_report_cal = test_report_raw

    # Итоговые метрики — берём калиброванные (если есть)
    test_report = test_report_cal

    # ----- Baseline (random) -----
    baseline_probs = random_baseline(len(test), seed=0)
    baseline_outcomes = [r["actual"] for r in test_results_raw]
    baseline_bs = brier_score(baseline_probs, baseline_outcomes)
    baseline_auc = roc_auc(baseline_probs, baseline_outcomes)
    baseline_acc = sum(
        (1 if p > 0.5 else 0) == o
        for p, o in zip(baseline_probs, baseline_outcomes)
    ) / len(test)

    # ----- Overfitting check (по сырым метрикам — честнее) -----
    gap = train_report["accuracy"] - test_report_raw["accuracy"]
    is_overfit = gap > overfitting_threshold

    return {
        "split_mode": split_mode,
        "weight_mode": weight_mode,
        "train_size": len(train_fit),
        "test_size": len(test),
        "threshold": threshold,
        "calibration_method": calibration_method,

        # Test метрики (калиброванные)
        "accuracy": test_report["accuracy"],
        "precision": test_report["precision"],
        "recall": test_report["recall"],
        "f1": test_report["f1"],
        "brier_score": test_report["brier_score"],
        "brier_skill_score": test_report["brier_skill_score"],
        "roc_auc": test_report["roc_auc"],
        "ece": test_report["ece"],

        # Сырые test метрики (до калибровки)
        "raw_accuracy": test_report_raw["accuracy"],
        "raw_brier_score": test_report_raw["brier_score"],
        "raw_ece": test_report_raw["ece"],

        # Статистика уверенности
        "signal_coverage": round(
            sum(1 for r in test_results_raw if r.get("has_signal", True)) / len(test_results_raw), 4
        ),
        "mean_confidence": round(
            sum(r.get("confidence", 1.0) for r in test_results_raw) / len(test_results_raw), 4
        ),
        "high_confidence_accuracy": _high_confidence_accuracy(test_results_raw, min_conf=0.5),

        # Train (только для диагностики)
        "train_accuracy": train_report["accuracy"],
        "train_brier_score": train_report["brier_score"],

        # Калибровочный отчёт
        "calibration": calibration_report,

        # Baseline сравнение
        "baseline_accuracy": round(baseline_acc, 4),
        "baseline_brier_score": round(baseline_bs, 4),
        "baseline_roc_auc": round(baseline_auc, 4) if isinstance(baseline_auc, float) else "n/a",

        # Флаги
        "overfitting_gap": round(gap, 4),
        "is_overfit": is_overfit,
        "beats_baseline_accuracy": test_report["accuracy"] > baseline_acc,
        "beats_baseline_brier": test_report["brier_score"] < baseline_bs,
    }


def _find_best_threshold(engine, val_data: list[dict], *, threshold_grid: list[float] | None = None) -> float:
    """
    Подбирает порог по максимуму F1 на val-сете.

    Почему так:
      - AUC меряет ранжирование, но для решения "buy/sell" нужен порог
      - при AUC ~0.53 оптимальный порог почти никогда не равен 0.5
    """
    grid = threshold_grid
    if grid is None:
        grid = [i / 100 for i in range(5, 96, 5)]  # 0.05..0.95

    best_t = 0.5
    best_f1 = -1.0

    # Считаем вероятности один раз, чтобы не гонять engine многократно
    eval_05 = _evaluate(engine, val_data, threshold=0.5)
    probs = [r["probability"] for r in eval_05]
    actuals = [r["actual"] for r in eval_05]

    for t in grid:
        tp = fp = fn = 0
        for p, a in zip(probs, actuals):
            pred = 1 if p > t else 0
            if pred == 1 and a == 1:
                tp += 1
            elif pred == 1 and a == 0:
                fp += 1
            elif pred == 0 and a == 1:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t


def run_k_fold_backtest(
    engine_factory,
    dataset: list[dict],
    k: int = 5,
    threshold: float = 0.5,
    weight_mode: str = "batch",
    ema_alpha: float = 0.1,
) -> dict:
    """
    K-fold cross-validation для оценки стабильности модели.

    engine_factory — callable, возвращающий новый пустой PatternEngine.
    Высокая дисперсия accuracy между фолдами = нестабильная модель.

    Args:
        weight_mode: "batch" — fit_signal_weights в конце train-фазы каждого фолда;
                     "online_ema" — update_weights_online после каждого кейса.
        ema_alpha:   Скорость EMA (только для weight_mode="online_ema").
    """
    # Temporal CV по умолчанию: train = прошлое, test = будущее.
    # Это снижает риск "fake alpha" из-за утечки информации из будущего.
    folds = temporal_k_fold_split(dataset, k=k)
    fold_results = []

    for i, (train, test) in enumerate(folds):
        engine = engine_factory()
        _supports_online = hasattr(engine, "update_weights_online")
        _supports_batch  = hasattr(engine, "fit_signal_weights")

        for case in train:
            engine.add_case(
                case["signals"],
                case["outcome"],
                market_context=case.get("market_context"),
            )
            if weight_mode == "online_ema" and _supports_online:
                engine.update_weights_online(case, alpha=ema_alpha)

        # Шаг 6: обучаем веса только на train-части каждого фолда
        if weight_mode == "batch" and _supports_batch:
            engine.fit_signal_weights(train)

        results = _evaluate(engine, test, threshold)
        report = full_report(results)
        fold_results.append({
            "fold": i + 1,
            "accuracy": report["accuracy"],
            "brier_score": report["brier_score"],
            "roc_auc": report["roc_auc"],
        })

    accuracies = [f["accuracy"] for f in fold_results]
    mean_acc = sum(accuracies) / len(accuracies)
    variance = sum((a - mean_acc) ** 2 for a in accuracies) / len(accuracies)
    std_dev = variance ** 0.5

    return {
        "k": k,
        "folds": fold_results,
        "mean_accuracy": round(mean_acc, 4),
        "std_accuracy": round(std_dev, 4),
        "min_accuracy": round(min(accuracies), 4),
        "max_accuracy": round(max(accuracies), 4),
        "is_stable": std_dev < 0.05,  # стандартное отклонение < 5% = стабильно
    }


# ---------------------------------------------------------------------------
# Внутренние хелперы
# ---------------------------------------------------------------------------

def _high_confidence_accuracy(results: list[dict], min_conf: float = 0.5) -> float | None:
    """Accuracy только на кейсах с высокой уверенностью."""
    high_conf = [r for r in results if r.get("confidence", 1.0) >= min_conf]
    if not high_conf:
        return None
    correct = sum(1 for r in high_conf if r["predicted"] == r["actual"])
    return round(correct / len(high_conf), 4)

def _evaluate(engine, data: list[dict], threshold: float) -> list[dict]:
    """
    Прогоняет данные через PatternEngine.

    Если compute_probability возвращает None (недостаточно данных) —
    кейс помечается флагом low_confidence. В метриках используется
    prior движка как fallback-вероятность (честнее, чем 0.5).
    """
    results = []
    prior = getattr(engine, "prior", 0.5)

    for case in data:
        market_context = case.get("market_context")
        matches = engine.find_patterns(case["signals"], market_context=market_context)
        result = engine.compute_result(matches)
        hybrid = engine.compute_final_probability(
            case["signals"],
            matches=matches,
            market_context=market_context,
        )
        prob = result["probability"]
        confidence = result["confidence"]
        has_signal = result["has_signal"]

        # Нет данных → используем prior, но помечаем как low_confidence
        effective_prob = hybrid["final_probability"] if hybrid["final_probability"] is not None else prior

        results.append({
            "predicted": 1 if effective_prob > threshold else 0,
            "actual": case["outcome"],
            "probability": effective_prob,
            "confidence": confidence,
            "has_signal": has_signal,
            "matches": result["matches"],
            "pattern_probability": hybrid["pattern_probability"],
            "signal_score": hybrid["signal_score"],
        })
    return results

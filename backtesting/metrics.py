"""
Профессиональные метрики качества предсказаний.

Модули:
    brier_score     — качество вероятностей (лучше accuracy для prob-моделей)
    calibration     — соответствие предсказанных вероятностей реальным частотам
    roc_auc         — Area Under ROC Curve
    full_report     — полный отчёт по всем метрикам
"""

from __future__ import annotations

import math
from collections import defaultdict


# ---------------------------------------------------------------------------
# Brier Score
# ---------------------------------------------------------------------------

def brier_score(predictions: list[float], outcomes: list[int]) -> float:
    """
    Brier Score = среднее квадратичное отклонение вероятности от исхода.

    Диапазон: [0.0, 1.0]
        0.0  — идеальные предсказания
        0.25 — случайное угадывание (монета)
        1.0  — идеально неправильные предсказания

    Лучше accuracy: штрафует за уверенные ошибки сильнее, чем за осторожные.
    """
    if len(predictions) != len(outcomes):
        raise ValueError("predictions и outcomes должны быть одинаковой длины.")
    if not predictions:
        raise ValueError("Пустые списки.")

    return sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / len(predictions)


def brier_skill_score(bs: float, bs_reference: float = 0.25) -> float:
    """
    Brier Skill Score (BSS) — насколько модель лучше случайного угадывания.

    BSS > 0  → модель лучше случая
    BSS = 0  → модель = случайное угадывание
    BSS < 0  → модель хуже случайного угадывания
    """
    return 1 - bs / bs_reference


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration(
    predictions: list[float],
    outcomes: list[int],
    n_bins: int = 5,
) -> list[dict]:
    """
    Проверяет калибровку модели: если говорим 70% — реально ли происходит 70%?

    Возвращает список бинов с:
        bin_center     — центр интервала вероятностей
        mean_predicted — средняя предсказанная вероятность в бине
        mean_actual    — реальная частота исходов в бине
        count          — количество кейсов в бине
        calibration_error — |mean_predicted - mean_actual|
    """
    bins: dict[int, list] = defaultdict(list)

    for p, o in zip(predictions, outcomes):
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bins[bin_idx].append((p, o))

    result = []
    for i in range(n_bins):
        items = bins.get(i, [])
        if not items:
            continue
        mean_pred = sum(x[0] for x in items) / len(items)
        mean_act = sum(x[1] for x in items) / len(items)
        result.append({
            "bin_center": round((i + 0.5) / n_bins, 2),
            "mean_predicted": round(mean_pred, 4),
            "mean_actual": round(mean_act, 4),
            "count": len(items),
            "calibration_error": round(abs(mean_pred - mean_act), 4),
        })

    return result


def expected_calibration_error(
    predictions: list[float],
    outcomes: list[int],
    n_bins: int = 10,
) -> float:
    """
    ECE — взвешенная средняя ошибка калибровки.
    Чем ближе к 0 — тем лучше откалибрована модель.
    """
    bins = calibration(predictions, outcomes, n_bins=n_bins)
    total = len(predictions)
    return sum(b["calibration_error"] * b["count"] / total for b in bins)


# ---------------------------------------------------------------------------
# ROC AUC (без сторонних библиотек)
# ---------------------------------------------------------------------------

def roc_auc(predictions: list[float], outcomes: list[int]) -> float:
    """
    Area Under ROC Curve, вычисленная методом трапеций.

    0.5 — случайный классификатор
    1.0 — идеальный классификатор
    < 0.5 — хуже случайного (инвертированный сигнал)
    """
    paired = sorted(zip(predictions, outcomes), key=lambda x: -x[0])

    tp, fp = 0, 0
    pos = sum(outcomes)
    neg = len(outcomes) - pos

    if pos == 0 or neg == 0:
        return float("nan")

    auc = 0.0
    prev_fp = 0

    for _, outcome in paired:
        if outcome == 1:
            tp += 1
        else:
            fp += 1
            auc += tp  # прямоугольник под кривой

    return auc / (pos * neg)


# ---------------------------------------------------------------------------
# Полный отчёт
# ---------------------------------------------------------------------------

def full_report(results: list[dict]) -> dict:
    """
    Полный отчёт по результатам бэктеста.

    results — список словарей с ключами:
        "predicted"   : int (0 или 1)
        "actual"      : int (0 или 1)
        "probability" : float (вероятность исхода=1)
    """
    if not results:
        raise ValueError("Список результатов пуст.")

    preds_binary = [r["predicted"] for r in results]
    actuals = [r["actual"] for r in results]
    probs = [r.get("probability", float(r["predicted"])) for r in results]

    tp = sum(1 for p, a in zip(preds_binary, actuals) if p == 1 and a == 1)
    fp = sum(1 for p, a in zip(preds_binary, actuals) if p == 1 and a == 0)
    fn = sum(1 for p, a in zip(preds_binary, actuals) if p == 0 and a == 1)
    correct = sum(1 for p, a in zip(preds_binary, actuals) if p == a)

    accuracy = correct / len(results)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    bs = brier_score(probs, actuals)
    bss = brier_skill_score(bs)
    auc = roc_auc(probs, actuals)
    ece = expected_calibration_error(probs, actuals)

    return {
        "total": len(results),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "brier_score": round(bs, 4),
        "brier_skill_score": round(bss, 4),
        "roc_auc": round(auc, 4) if not math.isnan(auc) else "n/a",
        "ece": round(ece, 4),
    }

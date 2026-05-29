"""
Веса и нормализация сигналов для Pattern Weighting + Signal Intelligence Layer.

Семантика весов (Шаг 6 — пересмотрена):
    weight[s] = P(outcome=1 | signal s присутствует)  ∈ [0.0, 1.0]

    0.5  — нейтральный сигнал (не даёт предсказательной силы)
    0.7  — умеренно позитивный предиктор
    0.9  — сильный позитивный предиктор
    0.2  — негативный предиктор (предсказывает outcome=0)

Дефолт 0.5 = "мы ничего не знаем об этом сигнале".
После обучения значения сдвигаются от 0.5 в сторону реальных вероятностей.
"""
from __future__ import annotations

from typing import Any

# --- Signal Intelligence Layer, шаг 1: структура наблюдаемых сигналов ---
# Пример: signals = {"low_volume": 1, "whale_absence": 1, "volatility_compression": 1}

# --- Шаг 2 — хранилище весов (дефолты до обучения; обновляется fit_signal_weights / EMA) ---
# Дефолт 0.5 = нейтральный prior (P(outcome=1) = 50%)
signal_weights = {
    "low_volume": 0.5,
    "whale_absence": 0.5,
    "volatility_compression": 0.5,
    "momentum_up": 0.5,
    "momentum_strength": 0.5,
    # Volume imbalance (реальные режимы объёма)
    "volume_spike": 0.5,
    "volume_dry": 0.5,
    # Momentum (градации, не просто up/down)
    "momentum_up_1": 0.5,
    "momentum_down_1": 0.5,
    "momentum_up_5": 0.5,
    "momentum_down_5": 0.5,
    "return_1h": 0.5,
    "return_4h": 0.5,
    "return_24h": 0.5,
    "atr_ratio": 0.5,
    "std_returns": 0.5,
    # Volatility regime усиление
    "range_expansion": 0.5,
    "volume_ratio": 0.5,
    "breakout": 0.5,
    "breakout_up": 0.5,
    # Mean reversion
    "mean_reversion_long": 0.5,
    "mean_reversion_short": 0.5,
    # Derivatives (asymmetric information)
    "funding_long_crowded": 0.5,
    "funding_short_crowded": 0.5,
    "long_liquidations_spike": 0.5,
    "short_liquidations_spike": 0.5,
    "oi_up_price_up": 0.5,
    "oi_down_price_up": 0.5,
    "oi_up_price_down": 0.5,
    "oi_down_price_down": 0.5,
    # Multi-timeframe derivatives (builder → multi_timeframe_derivatives_signals)
    "deriv_funding_stress_12": 0.5,
    "deriv_funding_stress_48": 0.5,
    "deriv_funding_stress_96": 0.5,
    "deriv_oi_stress_4": 0.5,
    "deriv_oi_stress_24": 0.5,
    "deriv_liq_spike_12": 0.5,
    "deriv_liq_spike_60": 0.5,
    # Scenario signals (second-order interaction features)
    "scen_long_trap": 0.5,
    "scen_short_squeeze": 0.5,
    "scen_capitulation": 0.5,
    "scen_breakout_confirmed": 0.5,
    "scen_breakout_suspicious": 0.5,
    "scen_accumulation": 0.5,
    "scen_momentum_exhaustion": 0.5,
    "scen_liq_reversal": 0.5,
    "scen_trend_flow": 0.5,
    "regime_high_volatility": 0.5,
    "regime_low_volatility": 0.5,
    "regime_trend": 0.5,
    "regime_range": 0.5,
}

CANONICAL_SIGNAL_KEYS: tuple[str, ...] = tuple(signal_weights.keys())


def full_presence_signals() -> dict[str, int]:
    """Все канонические сигналы с силой присутствия 1 (удобный шаблон для тестов/дампов)."""
    return {k: 1 for k in CANONICAL_SIGNAL_KEYS}


def normalize_signals_to_canonical(raw_signals: Any) -> dict[str, float]:
    """
    Приводит вход к каноническому виду (значения — float для скоринга):
        { "low_volume": 1.0, "whale_absence": 1.0, "volatility_compression": 1.0 }

    Исходный формат наблюдений Signal Intelligence Layer — dict с целыми 1:
        signals = {"low_volume": 1, "whale_absence": 1, "volatility_compression": 1}

    Поддерживаем 2 формата:
      - list[{"type": <str>, "strength": <float?>}, ...] (текущий датасет FIE)
      - dict[str, int | float] — канонический слой

    Возвращаем только "присутствующие" сигналы (value > 0),
    чтобы compute_signal_weights считал total корректно.
    """
    if raw_signals is None:
        return {}

    # Уже канонический dict-формат
    if isinstance(raw_signals, dict):
        out: dict[str, float] = {}
        for k, v in raw_signals.items():
            if k not in CANONICAL_SIGNAL_KEYS:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv > 0.0:
                out[k] = fv
        return out

    # list[{"type": ...}]
    if not isinstance(raw_signals, list):
        return {}

    # Снимаем strength если он есть, иначе считаем 1.0
    # Map "сырой тип" -> "канонический сигнал"
    type_to_canonical: dict[str, str] = {
        "low_volume": "low_volume",
        "no_whales": "whale_absence",
        # "volatility_compression" в реальных данных часто прокси:
        "low_volatility": "volatility_compression",
        "consolidation": "volatility_compression",
        # Контекст режима рынка (шаг 3: context features)
        "high_volatility": "regime_high_volatility",
        "low_volatility_regime": "regime_low_volatility",
        "trend": "regime_trend",
        "range": "regime_range",
    }

    # Если на один канонический сигнал приходится несколько типов,
    # берём max strength.
    out: dict[str, float] = {}
    for item in raw_signals:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if not isinstance(t, str):
            continue
        canonical = type_to_canonical.get(t)
        if canonical is None:
            continue
        raw_strength = item.get("strength", 1.0)
        try:
            v = float(raw_strength)
        except (TypeError, ValueError):
            v = 1.0
        if v <= 0.0:
            continue
        out[canonical] = max(out.get(canonical, 0.0), v)

    return out


def compute_signal_weights(dataset: list[dict]) -> dict[str, float]:
    """
    Шаг 3 — считаем вес сигнала из истории.

    Семантика (Шаг 6):
        weight[s] = P(outcome=1 | signal s присутствует)

    Формула:
        prob = (pos + 1) / (total + 2)   # Laplace smoothing → избегаем 0% и 100%

    Интерпретация:
        0.5  — нейтральный сигнал (нет предсказательной силы)
        0.7  — 70% кейсов с этим сигналом заканчивались outcome=1
        0.2  — негативный предиктор (редко ведёт к outcome=1)

    ``dataset``: элементы ``{"signals": {...}, "outcome": 0|1}``.
    Учитываются только канонические ключи и пары с ``value > 0`` (присутствие сигнала).
    Сигналы без статистики получают дефолтный вес 0.5 (нейтральный prior).
    """
    stats: dict[str, dict[str, float]] = {}

    for case in dataset:
        try:
            outcome = float(case["outcome"])
        except (KeyError, TypeError, ValueError):
            continue

        signals = case.get("signals")
        if not isinstance(signals, dict):
            continue

        for signal, value in signals.items():
            if signal not in CANONICAL_SIGNAL_KEYS:
                continue
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            if fv <= 0.0:
                continue

            if signal not in stats:
                stats[signal] = {"pos": 0.0, "total": 0.0}
            stats[signal]["total"] += 1.0
            stats[signal]["pos"] += outcome

    weights: dict[str, float] = {}
    for signal, s in stats.items():
        # P(outcome=1 | signal) с Laplace smoothing
        prob = (s["pos"] + 1.0) / (s["total"] + 2.0)
        weights[signal] = prob

    for key in CANONICAL_SIGNAL_KEYS:
        if key not in weights:
            weights[key] = float(signal_weights[key])  # 0.5 = нейтральный prior

    return weights


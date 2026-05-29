"""
Генератор синтетического датасета для бэктестинга.

Сигналы намеренно зашумлены: ни один из них не даёт 100% точность.
Исходы определяются комбинациями сигналов + случайный шум (~20%).
"""

import json
import math
import random
from pathlib import Path

random.seed(2024)

# ---------------------------------------------------------------------------
# Параметры "анти-комфорта" датасета (целенаправленная деградация сигнала)
# ---------------------------------------------------------------------------
LABEL_NOISE_RATE = 0.26          # доля случаев, где outcome инвертируется
CONTRADICTION_RATE = 0.38        # доля случаев с конфликтными (bull+bear) сигналами
FALSE_PATTERN_RATE = 0.18        # доля случаев с искусственными ложными паттернами

FALSE_PATTERNS = [
    # Паттерн выглядит бычьим, но outcome будет чаще медвежий
    {
        "signals": ["high_volume", "whale_buy", "rsi_oversold"],
        "forced_base_prob": 0.18,
    },
    # Паттерн выглядит медвежьим, но outcome будет чаще бычий
    {
        "signals": ["low_volume", "whale_sell", "rsi_overbought"],
        "forced_base_prob": 0.82,
    },
]

# ---------------------------------------------------------------------------
# Сигнальная схема
# ---------------------------------------------------------------------------

BULLISH_SIGNALS = [
    "high_volume", "whale_buy", "rsi_oversold", "macd_crossover_up",
    "support_bounce", "funding_negative", "exchange_outflow",
    "fear_greed_low", "open_interest_spike", "spot_premium",
]

BEARISH_SIGNALS = [
    "low_volume", "whale_sell", "rsi_overbought", "macd_crossover_down",
    "resistance_reject", "funding_positive", "exchange_inflow",
    "fear_greed_high", "open_interest_drop", "futures_premium",
]

NEUTRAL_SIGNALS = [
    "rsi_neutral", "no_whales", "sideways_volume", "low_volatility",
    "mixed_funding", "consolidation",
]

ALL_SIGNALS = BULLISH_SIGNALS + BEARISH_SIGNALS + NEUTRAL_SIGNALS


def _generate_ohlcv_window(length: int = 24) -> list[dict]:
    """
    Генерирует синтетическое OHLCV-окно.
    Внутри скрыто выбирается рыночный режим, чтобы индикаторы не были рандомом.
    """
    hidden_regime = random.choice(["trend", "range"])
    hidden_volatility = random.choice(["high", "low"])

    price = random.uniform(90.0, 110.0)
    bars: list[dict] = []

    for _ in range(length):
        drift = random.gauss(0.25, 0.2) if hidden_regime == "trend" else random.gauss(0.0, 0.08)
        vol_scale = random.uniform(1.3, 2.2) if hidden_volatility == "high" else random.uniform(0.5, 1.1)
        shock = random.gauss(0.0, 0.45 * vol_scale)

        open_p = price
        close_p = max(1.0, open_p + drift + shock)
        spread = abs(random.gauss(0.0, 0.35 * vol_scale)) + 0.02

        high_p = max(open_p, close_p) + spread
        low_p = min(open_p, close_p) - spread
        volume = random.uniform(600.0, 2400.0) * (1.0 + 0.4 * vol_scale)

        bars.append({
            "open": round(open_p, 4),
            "high": round(high_p, 4),
            "low": round(low_p, 4),
            "close": round(close_p, 4),
            "volume": round(volume, 2),
        })
        price = close_p

    return bars


def _compute_atr(ohlcv: list[dict], period: int = 14) -> float:
    if len(ohlcv) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(ohlcv)):
        h = ohlcv[i]["high"]
        l = ohlcv[i]["low"]
        prev_c = ohlcv[i - 1]["close"]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    tail = trs[-period:]
    return sum(tail) / len(tail)


def _compute_adx(ohlcv: list[dict], period: int = 14) -> float:
    if len(ohlcv) < period + 1:
        return 0.0

    plus_dm = []
    minus_dm = []
    tr_values = []

    for i in range(1, len(ohlcv)):
        up_move = ohlcv[i]["high"] - ohlcv[i - 1]["high"]
        down_move = ohlcv[i - 1]["low"] - ohlcv[i]["low"]

        plus = up_move if up_move > down_move and up_move > 0 else 0.0
        minus = down_move if down_move > up_move and down_move > 0 else 0.0
        plus_dm.append(plus)
        minus_dm.append(minus)

        h = ohlcv[i]["high"]
        l = ohlcv[i]["low"]
        prev_c = ohlcv[i - 1]["close"]
        tr_values.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))

    atr = sum(tr_values[-period:]) / period if period else 0.0
    if atr <= 1e-9:
        return 0.0

    plus_di = 100.0 * (sum(plus_dm[-period:]) / period) / atr
    minus_di = 100.0 * (sum(minus_dm[-period:]) / period) / atr
    denom = plus_di + minus_di
    if denom <= 1e-9:
        return 0.0

    dx = 100.0 * abs(plus_di - minus_di) / denom
    return dx


def _compute_bollinger_width(ohlcv: list[dict], period: int = 20, n_std: float = 2.0) -> float:
    if len(ohlcv) < period:
        return 0.0
    closes = [c["close"] for c in ohlcv[-period:]]
    mean = sum(closes) / period
    var = sum((x - mean) ** 2 for x in closes) / period
    std = math.sqrt(var)
    upper = mean + n_std * std
    lower = mean - n_std * std
    if mean <= 1e-9:
        return 0.0
    return (upper - lower) / mean


def _build_market_context_from_ohlcv(ohlcv: list[dict]) -> dict:
    """Реальный context из ATR/ADX/Bollinger width."""
    atr = _compute_atr(ohlcv)
    adx = _compute_adx(ohlcv)
    bb_width = _compute_bollinger_width(ohlcv)

    volatility = "high" if atr >= 1.35 else "low"
    regime = "trend" if adx >= 25.0 else "range"
    compression = "high" if bb_width <= 0.06 else "low"

    return {
        "volatility": volatility,
        "regime": regime,
        "compression": compression,
        "atr": round(atr, 4),
        "adx": round(adx, 4),
        "bollinger_width": round(bb_width, 4),
    }


def _compute_base_probability(signals: list[dict], market_context: dict) -> float:
    """Вычисляет базовую вероятность бычьего исхода на основе сигналов и режима."""
    types = {s["type"] for s in signals}

    bull_hits = sum(1 for s in BULLISH_SIGNALS if s in types)
    bear_hits = sum(1 for s in BEARISH_SIGNALS if s in types)
    total = bull_hits + bear_hits

    if total == 0:
        return 0.5  # нет сигналов — монета

    base_prob = bull_hits / total

    # Контекст = фильтр: некоторые сигналы меняют эффект в разных режимах.
    # Пример (смысл как в user request):
    #   low_volume: trend -> P(outcome=1|low_volume) выше, range -> ниже
    regime = str(market_context.get("regime", "")).lower()
    volatility = str(market_context.get("volatility", "")).lower()

    if "low_volume" in types:
        if regime == "trend":
            base_prob += 0.15
        elif regime == "range":
            base_prob -= 0.15

    # Доп. эффект волатильности: в high volatility импульсные сигналы чуть сильнее.
    if volatility == "high" and ("high_volume" in types or "macd_crossover_up" in types):
        base_prob += 0.05

    # Клипы, чтобы оставаться в [0..1]
    return max(0.02, min(0.98, base_prob))


def generate_case(case_id: int) -> dict:
    """Генерирует один кейс с зашумленным исходом."""
    ohlcv = _generate_ohlcv_window(length=24)
    market_context = _build_market_context_from_ohlcv(ohlcv)

    # Количество сигналов: 1–4
    n_signals = random.choices([1, 2, 3, 4], weights=[10, 40, 35, 15])[0]
    signal_types = random.sample(ALL_SIGNALS, n_signals)

    # Добавляем противоречивые сигналы: одновременно bull+bear в одном кейсе.
    if random.random() < CONTRADICTION_RATE:
        bull = random.choice(BULLISH_SIGNALS)
        bear = random.choice(BEARISH_SIGNALS)
        signal_types = list(set(signal_types + [bull, bear]))

    # Смешиваем бычьи/медвежьи сигналы с небольшим уклоном в случайную сторону
    bias = random.gauss(0, 0.08)
    base_prob = _compute_base_probability(
        [{"type": s} for s in signal_types],
        market_context=market_context,
    )
    noisy_prob = max(0.05, min(0.95, base_prob + bias))

    # Ложные паттерны: искажаем вероятность для части кейсов
    # и тем самым создаём "обманчивую" структуру в данных.
    if random.random() < FALSE_PATTERN_RATE:
        false_pattern = random.choice(FALSE_PATTERNS)
        signal_types = list(set(signal_types + false_pattern["signals"]))
        noisy_prob = false_pattern["forced_base_prob"]

    # Исход: вероятностный (не детерминированный — это важно для реализма)
    outcome = 1 if random.random() < noisy_prob else 0

    # Доп. label noise — ломает слишком "чистую" зависимость сигнал→исход.
    if random.random() < LABEL_NOISE_RATE:
        outcome = 1 - outcome

    return {
        "id": case_id,
        "event": f"BTC signal pattern #{case_id}",
        "signals": [{"type": s} for s in signal_types],
        "ohlcv": ohlcv,
        "market_context": market_context,
        "base_probability": round(noisy_prob, 3),
        "outcome": outcome,
    }


def main(n: int = 600, output: str = "data/historical_events.json") -> None:
    dataset = [generate_case(i) for i in range(1, n + 1)]

    outcomes = [c["outcome"] for c in dataset]
    bull_pct = sum(outcomes) / len(outcomes)

    print(f"Сгенерировано кейсов : {len(dataset)}")
    print(f"Исходов '1' (бычьих): {sum(outcomes)} ({bull_pct:.1%})")
    print(f"Исходов '0' (медв.)  : {len(outcomes) - sum(outcomes)} ({1-bull_pct:.1%})")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Сохранено в {out_path}")


if __name__ == "__main__":
    main()

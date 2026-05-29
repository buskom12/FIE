from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return None if math.isnan(v) else v
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            v = float(s)
            return None if math.isnan(v) else v
        except ValueError:
            return None
    return None


def _last_n_valid(xs: Iterable[float | None], n: int) -> list[float]:
    out: list[float] = []
    for x in reversed(list(xs)):
        if isinstance(x, (int, float)) and not math.isnan(float(x)):
            out.append(float(x))
            if len(out) >= n:
                break
    out.reverse()
    return out


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


# ---------------------------------------------------------------------
# Binance-style adapters (чтобы совместить с кодом из ТЗ)
# ---------------------------------------------------------------------


def funding_signal(funding_rates: list[Any]) -> dict[str, float]:
    """
    Принимает либо Binance fundingRate JSON (list[dict] с ключом fundingRate),
    либо список чисел.

    Важно: эта функция по смыслу "на последней точке". Для обучения на истории
    используйте `funding_signal_from_series()` внутри builder, чтобы не было leakage.
    """
    fr: list[float] = []
    for x in funding_rates:
        if isinstance(x, dict):
            v = _to_float(x.get("fundingRate"))
        else:
            v = _to_float(x)
        if v is not None:
            fr.append(v)
    if len(fr) < 5:
        return {"funding_extreme": 0.0, "funding_positive": 0.0, "funding_negative": 0.0}

    current = fr[-1]
    avg = float(np.mean(fr))
    std = float(np.std(fr)) if len(fr) > 1 else 0.0
    z = (current - avg) / std if std > 1e-12 else 0.0

    # continuous + нормализация (bounded)
    return {
        "funding_extreme": _clip(z, -8.0, 8.0),
        "funding_positive": max(0.0, current),
        "funding_negative": abs(min(0.0, current)),
    }


def oi_signal(klines: list[Any]) -> dict[str, float]:
    """
    Binance klines: list[list] где volume = kline[5].
    По смыслу в ТЗ это прокси OI; для обучения на истории используйте
    `oi_signal_from_series()` (volume/OI series без утечки).
    """
    vols: list[float] = []
    for x in klines:
        if isinstance(x, (list, tuple)) and len(x) > 5:
            v = _to_float(x[5])
        elif isinstance(x, dict):
            v = _to_float(x.get("volume"))
        else:
            v = None
        if v is not None:
            vols.append(v)
    if len(vols) < 6:
        return {"oi_increasing": 0.0, "oi_decreasing": 0.0}

    delta = vols[-1] - vols[-5]
    return {
        "oi_increasing": max(0.0, float(delta)),
        "oi_decreasing": abs(min(0.0, float(delta))),
    }


def liquidation_proxy(prices: list[float]) -> dict[str, float]:
    if len(prices) < 7:
        return {"liquidation_spike": 0.0}
    returns = np.diff(np.array(prices, dtype=float))
    spike = float(np.max(np.abs(returns[-5:]))) if returns.size else 0.0
    return {"liquidation_spike": spike}


# ---------------------------------------------------------------------
# Safe-for-training helpers (rolling, no future leakage)
# ---------------------------------------------------------------------


def funding_signal_from_series(
    funding_series: list[float | None],
    i: int,
    *,
    window: int = 100,
) -> dict[str, float]:
    """
    Continuous signals по funding, используя только 0..i (без утечки будущего).
    Возвращает "raw magnitude" + bounded extreme score.
    """
    if i < 0 or i >= len(funding_series):
        return {"funding_extreme": 0.0, "funding_positive": 0.0, "funding_negative": 0.0}

    hist = funding_series[max(0, i - window + 1) : i + 1]
    fr = _last_n_valid(hist, window)
    cur = funding_series[i]
    cur_f = float(cur) if isinstance(cur, (int, float)) and not math.isnan(float(cur)) else None
    # Funding часто обновляется реже, чем 1h бары (например каждые 8h).
    # Если на текущем баре значение отсутствует, используем последнее валидное
    # из окна (leakage-free: это значение из прошлого/текущего hist).
    if cur_f is None and fr:
        cur_f = float(fr[-1])
    # Для hourly рядов с редкими апдейтами (funding ~ каждые 8h) 48h окно даёт ~6 точек.
    # 8 точек как минимум слишком жёстко и превращает фичу в константу 0.
    min_points = 4
    if cur_f is None or len(fr) < min_points:
        return {"funding_extreme": 0.0, "funding_positive": 0.0, "funding_negative": 0.0}

    avg = float(np.mean(fr))
    std = float(np.std(fr)) if len(fr) > 1 else 0.0
    z = (cur_f - avg) / std if std > 1e-12 else 0.0
    return {
        "funding_extreme": _clip(z, -8.0, 8.0),
        "funding_positive": max(0.0, cur_f),
        "funding_negative": abs(min(0.0, cur_f)),
    }


def oi_signal_from_series(
    proxy_series: list[float | None],
    i: int,
    *,
    lookback: int = 5,
    scale: float | None = None,
) -> dict[str, float]:
    """
    Continuous OI proxy delta без бинари.
    Если scale задан, нормализуем delta -> [-1,1] через tanh(delta/scale).
    """
    if i < 0 or i >= len(proxy_series):
        return {"oi_increasing": 0.0, "oi_decreasing": 0.0}
    if i - lookback < 0:
        return {"oi_increasing": 0.0, "oi_decreasing": 0.0}

    cur = proxy_series[i]
    prev = proxy_series[i - lookback]
    if cur is None or prev is None:
        return {"oi_increasing": 0.0, "oi_decreasing": 0.0}

    delta = float(cur) - float(prev)
    if scale is not None and scale > 1e-12:
        # bounded continuous magnitude, но знак сохраняем через split
        delta = math.tanh(delta / scale)
    return {"oi_increasing": max(0.0, delta), "oi_decreasing": abs(min(0.0, delta))}


def _oi_scale_median(
    proxy_series: list[float | None],
    i: int,
    lookback: int,
    *,
    tail: int = 20,
) -> float | None:
    """Медиана |ΔOI| за последние `tail` баров (без утечки вперёд)."""
    if i < lookback:
        return None
    deltas: list[float] = []
    for j in range(max(0, i - tail + 1), i + 1):
        if j - lookback < 0:
            continue
        cur = proxy_series[j]
        prev = proxy_series[j - lookback]
        if cur is None or prev is None:
            continue
        deltas.append(abs(float(cur) - float(prev)))
    if not deltas:
        return None
    deltas.sort()
    return deltas[len(deltas) // 2]


def liquidation_spike_normalized(
    closes: list[float],
    *,
    tail: int = 60,
    ref_frac: float = 0.005,
) -> float:
    """
    Нормализует liquidation_proxy к [0, 1]: крупный 1h-ход относительно цены.
    """
    if not closes:
        return 0.0
    w = closes[-tail:] if len(closes) >= tail else closes
    lp = liquidation_proxy(w).get("liquidation_spike", 0.0)
    p = float(closes[-1])
    if p <= 1e-12:
        return 0.0
    denom = p * max(ref_frac, 1e-8)
    return _clip(lp / denom, 0.0, 1.0)


def multi_timeframe_derivatives_signals(
    funding_series: list[float | None],
    oi_series: list[float | None],
    closes: list[float],
    i: int,
) -> dict[str, float]:
    """
    Multi-timeframe (на 1h барах): короткие / средние / длинные окна funding,
    разные горизонты OI, два горизонта liquidation proxy.

    Все значения в [0, 1] — совместимо с PatternEngine (presence strength).
    """
    out: dict[str, float] = {}

    # Funding: окна в барах (= часах для 1h)
    for window, tag in ((12, "12"), (48, "48"), (96, "96")):
        d = funding_signal_from_series(funding_series, i, window=window)
        z = float(d.get("funding_extreme", 0.0))
        # stress = насколько экстрален funding (в любую сторону)
        # Важно: избегаем жёсткой сатурации в 1.0 из-за clip(abs(z)/k),
        # иначе downstream (gate_score через max(...)) легко схлопывается в константу.
        az = abs(z)
        k = 4.0
        out[f"deriv_funding_stress_{tag}"] = float(az / (az + k)) if az > 0 else 0.0

    # OI: разные lookback в барах
    for lb, tag in ((4, "4"), (24, "24")):
        sc = _oi_scale_median(oi_series, i, lookback=lb, tail=20)
        d = oi_signal_from_series(oi_series, i, lookback=lb, scale=sc)
        inc = float(d.get("oi_increasing", 0.0))
        dec = float(d.get("oi_decreasing", 0.0))
        out[f"deriv_oi_stress_{tag}"] = _clip(max(inc, dec), 0.0, 1.0)

    # Liquidation proxy: короткое и длинное окно по цене
    out["deriv_liq_spike_12"] = liquidation_spike_normalized(closes, tail=12)
    out["deriv_liq_spike_60"] = liquidation_spike_normalized(closes, tail=60)

    return out


"""
Сигналы для PatternEngine: моментум, доходности, волатильность, объём, пробой.

Все числовые признаки приводятся к [0, 1] для канонического слоя (value > 0 = участвует).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Минимальный индекс бара (нужна история для return_24h, breakout 20, ATR 14)
MIN_SIGNAL_INDEX = 30


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _scale_return_simple(r: float) -> float:
    """
    Legacy scaler (ret -> [0,1]) оставлен для совместимости.

    Важно: для PatternEngine "value > 0" означает присутствие сигнала.
    Поэтому плотные признаки (всегда >0) убивают структуру.
    Ниже в generate_signals() мы используем разреженные strength-фичи.
    """
    return _clip01((r + 0.03) / 0.06)


def _return_strength(r: float, *, deadzone: float, scale: float) -> float:
    """
    Разреженный continuous сигнал для доходностей:
      - |r| <= deadzone -> 0
      - дальше растёт линейно до 1 при |r| ~= scale
    """
    ar = abs(r)
    if ar <= deadzone:
        return 0.0
    if scale <= deadzone:
        return 0.0
    return _clip01((ar - deadzone) / (scale - deadzone))


def _true_range(
    high_i: float,
    low_i: float,
    close_prev: float,
) -> float:
    return max(
        high_i - low_i,
        abs(high_i - close_prev),
        abs(low_i - close_prev),
    )


def _atr_sma(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int,
    i: int,
) -> float:
    """ATR ≈ SMA(True Range, period) на баре i."""
    if i < period:
        return 0.0
    trs: list[float] = []
    for j in range(i - period + 1, i + 1):
        trs.append(_true_range(highs[j], lows[j], closes[j - 1]))
    return sum(trs) / float(len(trs))


def _mean(xs: list[float]) -> float:
    return sum(xs) / float(len(xs)) if xs else 0.0


def _std(xs: list[float]) -> float:
    if not xs:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / float(len(xs))
    return math.sqrt(var)


def _ret_to_strength_up_down(r: float, *, scale: float) -> tuple[float, float]:
    """
    Преобразует доходность r в разреженные силы up/down.
    scale: при |r|≈scale получаем strength≈1.
    """
    if scale <= 1e-12:
        return 0.0, 0.0
    if r > 0.0:
        return _clip01(r / scale), 0.0
    if r < 0.0:
        return 0.0, _clip01((-r) / scale)
    return 0.0, 0.0


def generate_signals(
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float],
    i: int,
    *,
    funding_rate: float | None = None,
    open_interest: float | None = None,
    liq_long_usd: float | None = None,
    liq_short_usd: float | None = None,
    # engineered derivatives features (preferred)
    funding_z: float | None = None,
    oi_ret_1: float | None = None,
    long_liq_ratio: float | None = None,
    short_liq_ratio: float | None = None,
) -> dict[str, float]:
    """
    Сигналы на баре i (закрытые свечи 0..i).

    Требование: i >= MIN_SIGNAL_INDEX, len(closes) == len(highs) == len(lows) == len(volumes).
    """
    if i < MIN_SIGNAL_INDEX or i >= len(closes):
        return {}

    c = closes
    h = highs
    lo = lows
    v = volumes

    signals: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 0) Derivatives: funding / OI / liquidations (asymmetric information)
    # ------------------------------------------------------------------
    # Эти значения приходят "по времени бара". Мы не строим плотные признаки,
    # только отклонения/спайки.

    # Funding extremes (z-score over rolling window when available; fallback to abs funding)
    #  - funding_z >> 0 => crowding longs
    #  - funding_z << 0 => crowding shorts
    if isinstance(funding_z, (int, float)):
        fz = float(funding_z)
        if fz >= 1.5:
            signals["funding_long_crowded"] = _clip01((fz - 1.5) / 2.5)
        elif fz <= -1.5:
            signals["funding_short_crowded"] = _clip01((abs(fz) - 1.5) / 2.5)
    elif funding_rate is not None and isinstance(funding_rate, (int, float)):
        fr = float(funding_rate)
        fr_abs = abs(fr)
        fr_s = 0.0
        if fr_abs > 0.0002:
            fr_s = _clip01((fr_abs - 0.0002) / 0.0008)
        if fr_s > 0:
            if fr > 0:
                signals["funding_long_crowded"] = fr_s
            elif fr < 0:
                signals["funding_short_crowded"] = fr_s

    # Liquidations spike: ratio(current / avg_window)
    if isinstance(long_liq_ratio, (int, float)):
        r = float(long_liq_ratio)
        # 1.5x..5.0x -> 0..1
        if r >= 1.5:
            signals["long_liquidations_spike"] = _clip01((r - 1.5) / 3.5)
    elif liq_long_usd is not None and isinstance(liq_long_usd, (int, float)):
        ll = float(liq_long_usd)
        if ll > 0:
            s = _clip01(ll / 50_000_000.0)
            if s > 0.05:
                signals["long_liquidations_spike"] = s

    if isinstance(short_liq_ratio, (int, float)):
        r = float(short_liq_ratio)
        if r >= 1.5:
            signals["short_liquidations_spike"] = _clip01((r - 1.5) / 3.5)
    elif liq_short_usd is not None and isinstance(liq_short_usd, (int, float)):
        sl = float(liq_short_usd)
        if sl > 0:
            s = _clip01(sl / 50_000_000.0)
            if s > 0.05:
                signals["short_liquidations_spike"] = s

    # OI + price move decomposition (uses oi_ret_1 when available)
    if isinstance(oi_ret_1, (int, float)):
        oi_r = float(oi_ret_1)
        # считаем "существенным" изменение OI от 0.2% и выше (для 1h это уже движение)
        if abs(oi_r) > 0.002:
            # сила по масштабу изменения OI
            oi_s = _clip01((abs(oi_r) - 0.002) / 0.02)
            # направление цены — по r1 (у нас он ниже вычисляется, но тут можно пересчитать быстро)
            price_ret_1 = (closes[i] / closes[i - 1]) - 1.0 if closes[i - 1] > 0 else 0.0
            if price_ret_1 > 0 and oi_r > 0:
                signals["oi_up_price_up"] = oi_s
            elif price_ret_1 > 0 and oi_r < 0:
                signals["oi_down_price_up"] = oi_s
            elif price_ret_1 < 0 and oi_r > 0:
                signals["oi_up_price_down"] = oi_s
            elif price_ret_1 < 0 and oi_r < 0:
                signals["oi_down_price_down"] = oi_s

    # --- legacy: low volume (окно 10 баров до i включительно) ---
    w10 = v[i - 9 : i + 1]
    avg_vol_10 = sum(w10) / len(w10)
    if avg_vol_10 > 0:
        # continuous: чем ниже относительно среднего, тем сильнее
        lv = _clip01((0.7 - (v[i] / avg_vol_10)) / 0.7)
        if lv > 0:
            signals["low_volume"] = lv

    # --- Volume imbalance: volume_now / avg_volume (20 баров до i-1) ---
    avg_v20_ref = float(np.mean(v[i - 20 : i]))
    vr_raw = (v[i] / avg_v20_ref) if avg_v20_ref > 0 else 1.0

    # Непрерывный feature, но разреженный: "насколько далеко от нормы 1.0"
    # 1.0±0.1 -> 0; 1.0±1.1 -> ~1
    if avg_v20_ref > 0:
        vr_dev = abs(vr_raw - 1.0)
        vr_s = _clip01((vr_dev - 0.1) / 1.0)
        if vr_s > 0:
            signals["volume_ratio"] = vr_s

    # Дискретные режимы (spike/dry) со strength
    if vr_raw >= 1.5:
        signals["volume_spike"] = _clip01((vr_raw - 1.5) / 1.5)
    if vr_raw <= 0.7:
        signals["volume_dry"] = _clip01((0.7 - vr_raw) / 0.7)

    # --- legacy: краткий моментум (делаем continuous) ---
    if c[i - 5] > 0:
        r5_for_m = (c[i] / c[i - 5]) - 1.0
        if r5_for_m > 0:
            signals["momentum_up"] = _clip01(r5_for_m / 0.05)

    # --- volatility compression (continuous без "if") ---
    span5 = max(c[i - 4 : i + 1]) - min(c[i - 4 : i + 1])
    span_all = max(c[i - 9 : i + 1]) - min(c[i - 9 : i + 1])
    if span_all > 1e-12:
        ratio = span5 / span_all
        # ratio <= 0.2 -> ~1, ratio >= 0.8 -> 0
        signals["volatility_compression"] = _clip01((0.8 - ratio) / 0.6)
    else:
        signals["volatility_compression"] = 0.0

    # --- доходности (1h = 1 бар при интервале 1h) ---
    r1 = (c[i] / c[i - 1]) - 1.0 if c[i - 1] > 0 else 0.0
    r4 = (c[i] / c[i - 4]) - 1.0 if c[i - 4] > 0 else 0.0
    r5 = (c[i] / c[i - 5]) - 1.0 if c[i - 5] > 0 else 0.0
    r24 = (c[i] / c[i - 24]) - 1.0 if c[i - 24] > 0 else 0.0

    # --- return strength (continuous + разреженность) ---
    rs1 = _return_strength(r1, deadzone=0.0015, scale=0.02)
    rs4 = _return_strength(r4, deadzone=0.0025, scale=0.04)
    rs24 = _return_strength(r24, deadzone=0.006, scale=0.08)
    if rs1 > 0:
        signals["return_1h"] = rs1
    if rs4 > 0:
        signals["return_4h"] = rs4
    if rs24 > 0:
        signals["return_24h"] = rs24

    # --- Momentum strength (градации) на ret_1 и ret_5 ---
    up1, down1 = _ret_to_strength_up_down(r1, scale=0.02)
    up5, down5 = _ret_to_strength_up_down(r5, scale=0.05)
    if up1 > 0:
        signals["momentum_up_1"] = up1
    if down1 > 0:
        signals["momentum_down_1"] = down1
    if up5 > 0:
        signals["momentum_up_5"] = up5
    if down5 > 0:
        signals["momentum_down_5"] = down5

    # --- momentum strength: |текущий лог-рет| / std лог-рет за 20 баров ---
    log_prices = np.log(np.maximum(np.array(c[i - 20 : i + 1], dtype=float), 1e-12))
    lr = np.diff(log_prices)
    std_lr = float(np.std(lr)) if lr.size else 0.0
    cur_lr = math.log(c[i] / c[i - 1]) if c[i - 1] > 0 else 0.0
    if std_lr > 1e-12:
        z = abs(cur_lr) / std_lr
        ms = _clip01(z / 4.0)
        if ms > 0:
            signals["momentum_strength"] = ms

    # --- ATR(14) / цена ---
    atr = _atr_sma(h, lo, c, 14, i)
    atr_ratio = atr / c[i] if c[i] > 0 else 0.0
    # Высокий ATR относительно цены (режим) — 0 в "норме"
    # 0.01..0.05 -> 0..1
    atr_s = 0.0
    if atr_ratio > 0.01:
        atr_s = _clip01((atr_ratio - 0.01) / 0.04)
    if atr_s > 0:
        signals["atr_ratio"] = atr_s

    # --- std простых доходностей за 20 баров (нормировка) ---
    rets = []
    for j in range(i - 19, i + 1):
        if c[j - 1] > 0:
            rets.append((c[j] / c[j - 1]) - 1.0)
    std_r = float(np.std(rets)) if rets else 0.0
    # Высокая дисперсия доходностей — 0 в "норме"
    # 0.005..0.025 -> 0..1
    sr_s = 0.0
    if std_r > 0.005:
        sr_s = _clip01((std_r - 0.005) / 0.02)
    if sr_s > 0:
        signals["std_returns"] = sr_s

    # --- Range expansion: True Range / avg True Range(20) ---
    tr_now = _true_range(h[i], lo[i], c[i - 1])
    trs20 = [_true_range(h[j], lo[j], c[j - 1]) for j in range(i - 19, i + 1)]
    avg_tr20 = _mean(trs20)
    if avg_tr20 > 1e-12:
        tr_ratio = tr_now / avg_tr20
        if tr_ratio > 1.0:
            # 1.0..3.0 -> 0..1
            signals["range_expansion"] = _clip01((tr_ratio - 1.0) / 2.0)

    # --- breakout / structure (continuous strength в ATR-единицах) ---
    prior_max = max(c[i - 20 : i])
    if c[i] > prior_max:
        # 0..2 ATR -> 0..1
        denom = atr if atr > 1e-12 else (prior_max * 0.01 if prior_max > 0 else 1.0)
        strength = (c[i] - prior_max) / denom
        s01 = _clip01(strength / 2.0)
        signals["breakout"] = s01
        signals["breakout_up"] = s01
    else:
        signals["breakout"] = 0.0
        signals["breakout_up"] = 0.0

    # --- Mean reversion: z-score по окну 20 ---
    w20 = c[i - 19 : i + 1]
    m20 = _mean(w20)
    s20 = _std(w20)
    if s20 > 1e-12:
        z = (c[i] - m20) / s20
        if z <= -1.0:
            signals["mean_reversion_long"] = _clip01((abs(z) - 1.0) / 3.0)
        elif z >= 1.0:
            signals["mean_reversion_short"] = _clip01((abs(z) - 1.0) / 3.0)

    return signals


def generate_signals_from_row_window(
    window_rows: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Упрощение для тестов: window_rows — последние (MIN_SIGNAL_INDEX+1)+ баров,
    сигнал на последнем баре.
    """
    if len(window_rows) <= MIN_SIGNAL_INDEX:
        return {}
    closes = [float(r["price"]) for r in window_rows]
    highs = [float(r.get("high", r["price"])) for r in window_rows]
    lows = [float(r.get("low", r["price"])) for r in window_rows]
    volumes = [float(r["volume"]) for r in window_rows]
    i = len(window_rows) - 1
    return generate_signals(closes, highs, lows, volumes, i)

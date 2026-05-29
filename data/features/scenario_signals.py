"""
Scenario Signals — second-order features.

Каждый сценарий = мягкое произведение нескольких уже готовых сигналов.
Это interaction-термы: PatternEngine учится не на отдельных битах,
а на ситуациях (Long Trap, Capitulation, Accumulation, и т.д.).

Важно:
  - принимаем уже готовый dict signals (из generate_signals + MTF derivatives)
  - не обращаемся к будущим барам → leakage-free
  - все выходные значения в [0, 1], continuous
  - если хотя бы один компонент сценария == 0 → сценарий = 0 (нет ситуации)
"""

from __future__ import annotations

import math


def _g2(a: float, b: float) -> float:
    """Геометрическое среднее двух значений [0,1] -> [0,1]."""
    return math.sqrt(max(0.0, a) * max(0.0, b))


def _g3(a: float, b: float, c: float) -> float:
    """
    Soft-агрегация трёх сигналов (вместо жёсткого geometric mean).

    Геометрическое среднее (a*b*c)^(1/3) обнуляет сценарий, если хотя бы один
    компонент == 0. В реальном пайплайне многие входы дискретны/редки (часто
    ровно 0 из-за deadzone/резких порогов), что превращает сценарии и gate_score
    в state-machine и далее даёт few unique p_model.

    Здесь берём среднее парных произведений:
      (a*b + a*c + b*c) / 3
    Это делает сценарий зависимым от двух любых компонентов и сохраняет
    boundedness в [0..1] при входах [0..1].
    """
    aa = max(0.0, float(a))
    bb = max(0.0, float(b))
    cc = max(0.0, float(c))
    return (aa * bb + aa * cc + bb * cc) / 3.0


def _s(signals: dict[str, float], key: str) -> float:
    return float(signals.get(key, 0.0))


def compute_scenario_signals(signals: dict[str, float]) -> dict[str, float]:
    """
    Вычисляет 9 сценарных сигналов из уже готового dict signals.

    Сценарии:
      scen_long_trap          — толпа в лонгах + цена падает
      scen_short_squeeze      — толпа в шортах + цена растёт
      scen_capitulation       — ликвидационный спайк + нисходящий импульс + высокая ATR
      scen_breakout_confirmed — пробой + объёмное подтверждение
      scen_breakout_suspicious— пробой при сухом объёме (возможный ложный)
      scen_accumulation       — сжатие волатильности + сухой объём (накопление)
      scen_momentum_exhaustion— сильный импульс + лик спайк (перегрев = разворот)
      scen_liq_reversal       — ценовой спайк-ликвидация + зоны разворота
      scen_trend_flow         — подтверждённый тренд: импульс + объём + OI
    """
    out: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 1. Long Trap (crowded longs + price falling)
    # Когда у всех лонги, и цена начинает падать — капкан.
    # Outcome=0-biased сценарий.
    # ------------------------------------------------------------------
    funding_lc  = _s(signals, "funding_long_crowded")
    mom_down1   = _s(signals, "momentum_down_1")
    mom_down5   = _s(signals, "momentum_down_5")
    down_conf   = max(mom_down1, mom_down5 * 0.7)  # приоритет 1h, 5h как поддержка
    out["scen_long_trap"] = _g2(funding_lc, down_conf)

    # ------------------------------------------------------------------
    # 2. Short Squeeze (crowded shorts + price rising)
    # ------------------------------------------------------------------
    funding_sc  = _s(signals, "funding_short_crowded")
    mom_up1     = _s(signals, "momentum_up_1")
    mom_up5     = _s(signals, "momentum_up_5")
    up_conf     = max(mom_up1, mom_up5 * 0.7)
    out["scen_short_squeeze"] = _g2(funding_sc, up_conf)

    # ------------------------------------------------------------------
    # 3. Capitulation
    # Большой ликвидационный спайк + нисходящий импульс + высокий ATR
    # (вынос лонгов при волатильном падении)
    # ------------------------------------------------------------------
    liq_long    = _s(signals, "long_liquidations_spike")
    liq_proxy   = max(_s(signals, "deriv_liq_spike_12"), liq_long)
    atr_r       = _s(signals, "atr_ratio")
    out["scen_capitulation"] = _g3(liq_proxy, down_conf, atr_r)

    # ------------------------------------------------------------------
    # 4. Breakout Confirmed
    # Пробой вверх + объём выше нормы → деньги идут на пробой.
    # ------------------------------------------------------------------
    bo_up       = _s(signals, "breakout_up")
    vol_spike   = _s(signals, "volume_spike")
    vol_ratio   = _s(signals, "volume_ratio")
    vol_conf    = max(vol_spike, vol_ratio * 0.5)
    out["scen_breakout_confirmed"] = _g2(bo_up, vol_conf)

    # ------------------------------------------------------------------
    # 5. Breakout Suspicious (ложный пробой)
    # Пробой вверх при сухом объёме — вероятно, ложный.
    # ------------------------------------------------------------------
    vol_dry     = _s(signals, "volume_dry")
    out["scen_breakout_suspicious"] = _g2(bo_up, vol_dry)

    # ------------------------------------------------------------------
    # 6. Accumulation Zone
    # Компрессия волатильности + сухой объём + нейтральный funding stress
    # = рынок "в спящем режиме" перед движением.
    # ------------------------------------------------------------------
    vc          = _s(signals, "volatility_compression")
    # Считаем нейтральность фандинга: стресс по 48h окну низкий → нейтрален
    f_stress    = _s(signals, "deriv_funding_stress_48")
    f_neutral   = max(0.0, 1.0 - f_stress * 2.0)  # 0 stress → 1.0; 0.5+ stress → 0
    out["scen_accumulation"] = _g3(vc, vol_dry, f_neutral)

    # ------------------------------------------------------------------
    # 7. Momentum Exhaustion
    # Сильный 5h импульс + ценовой ликвидационный спайк → перегрев, разворот.
    # ------------------------------------------------------------------
    liq60       = _s(signals, "deriv_liq_spike_60")
    strong_move = max(mom_up5, mom_down5)
    out["scen_momentum_exhaustion"] = _g2(strong_move, liq60)

    # ------------------------------------------------------------------
    # 8. Liq-Driven Reversal
    # Острый ценовой спайк (12h proxy) + зона mean-reversion
    # = вероятный отскок после ликвидационного провала.
    # ------------------------------------------------------------------
    liq12       = _s(signals, "deriv_liq_spike_12")
    mr_long     = _s(signals, "mean_reversion_long")
    mr_short    = _s(signals, "mean_reversion_short")
    mr_any      = max(mr_long, mr_short)
    out["scen_liq_reversal"] = _g2(liq12, mr_any)

    # ------------------------------------------------------------------
    # 9. Trend Flow (подтверждённый тренд)
    # Пробой или сильный импульс + растущий OI + volume spike
    # = деньги входят в тренд.
    # ------------------------------------------------------------------
    oi_up       = _s(signals, "oi_up_price_up")
    oi_stress   = _s(signals, "deriv_oi_stress_24")  # если OI data есть
    oi_conf     = max(oi_up, oi_stress * 0.6)
    trend_dir   = max(bo_up, mom_up5, mom_up1 * 0.5)
    out["scen_trend_flow"] = _g3(trend_dir, vol_conf, max(oi_conf, 0.1))

    return out

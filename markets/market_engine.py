"""
Market edge detection / gating.

Цель: включать предсказания (и FIE agents) только в edge zones,
а не "угадывать рынок" на каждом тике.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from data.collectors.derivatives import get_btc_derivatives
from data.collectors.market import get_btc_ohlcv
from data.features.context import compute_context
from data.features.derivatives_signals import multi_timeframe_derivatives_signals
from data.features.scenario_signals import compute_scenario_signals
from data.features.signals import MIN_SIGNAL_INDEX, generate_signals


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return None if math.isnan(v) else v
    return None


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return str(v).strip().lower() not in {"0", "false", "no", "off"}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(float(x), hi))


@dataclass(frozen=True)
class EdgeZone:
    allow_prediction: bool
    state: str
    market_state: dict[str, Any]
    regime: str
    volatility: str
    regime_key: str
    scen_accumulation: float
    scen_breakout_confirmed: float
    scen_breakout_suspicious: float
    scen_trend_flow: float
    breakout_up: float
    funding_stress: float
    liq_spike: float
    momentum_up: float
    oi_strength: float
    gate_score: float
    gate_threshold: float
    reason: str


_market_state: dict[str, Any] = {
    "phase": "idle",        # idle | accumulation | breakout | trend
    "confidence": 0.0,      # [0,1]
    "duration": 0,          # bars in current phase
}


def _update_market_state(phase: str, confidence: float) -> dict[str, Any]:
    prev_phase = str(_market_state.get("phase", "idle"))
    prev_duration = int(_market_state.get("duration", 0))
    if phase == prev_phase:
        duration = prev_duration + 1
    else:
        duration = 1
    _market_state["phase"] = phase
    _market_state["confidence"] = max(0.0, min(float(confidence), 1.0))
    _market_state["duration"] = duration
    return dict(_market_state)


def compute_edge_zone(
    *,
    t_accumulation: float | None = None,
    t_breakout: float | None = None,
    min_candles: int = 400,
    force_refresh: bool = False,
) -> EdgeZone:
    """
    Возвращает решение, можно ли сейчас делать предсказание.

    Правило (edge-only):
      allow_prediction = (regime == "trend" and volatility == "low")
                         and (scen_accumulation > t_acc or breakout_up > t_bo)
    """
    t_acc = float(t_accumulation if t_accumulation is not None else _env_float("FIE_T_ACCUMULATION", 0.35))
    t_bo = float(t_breakout if t_breakout is not None else _env_float("FIE_T_BREAKOUT", t_acc))

    raw = get_btc_ohlcv(force_refresh=force_refresh, min_candles=max(min_candles, MIN_SIGNAL_INDEX + 50))
    if len(raw) <= MIN_SIGNAL_INDEX:
        ms = dict(_market_state)
        return EdgeZone(
            allow_prediction=False,
            state=str(ms.get("phase", "idle")),
            market_state=ms,
            regime="unknown",
            volatility="unknown",
            regime_key="unknown",
            scen_accumulation=0.0,
            scen_breakout_confirmed=0.0,
            scen_breakout_suspicious=0.0,
            scen_trend_flow=0.0,
            breakout_up=0.0,
            funding_stress=0.0,
            liq_spike=0.0,
            momentum_up=0.0,
            oi_strength=0.0,
            gate_score=0.0,
            gate_threshold=0.0,
            reason="not_enough_candles",
        )

    deriv = get_btc_derivatives(force_refresh=force_refresh)
    deriv_by_ts = {int(r["timestamp"]): r for r in deriv if isinstance(r, dict) and "timestamp" in r}

    i = len(raw) - 1
    closes = [float(x["price"]) for x in raw[: i + 1]]
    highs = [float(x.get("high", x["price"])) for x in raw[: i + 1]]
    lows = [float(x.get("low", x["price"])) for x in raw[: i + 1]]
    volumes = [float(x["volume"]) for x in raw[: i + 1]]

    ctx_window = closes[-30:] if len(closes) >= 30 else closes
    context = compute_context(ctx_window)
    regime = str(context.get("regime", "unknown"))
    volatility = str(context.get("volatility", "unknown"))
    regime_key = f"{regime}_{volatility}" if regime in ("trend", "range") and volatility in ("low", "high") else "unknown"

    ts_i = int(raw[i].get("timestamp", 0))
    d = deriv_by_ts.get(ts_i, {})
    signals = generate_signals(
        closes,
        highs,
        lows,
        volumes,
        i,
        funding_rate=_to_float(d.get("funding_rate")) if isinstance(d, dict) else None,
        open_interest=_to_float(d.get("open_interest")) if isinstance(d, dict) else None,
        liq_long_usd=_to_float(d.get("liq_long_usd")) if isinstance(d, dict) else None,
        liq_short_usd=_to_float(d.get("liq_short_usd")) if isinstance(d, dict) else None,
    )
    funding_series = [
        (_to_float(deriv_by_ts.get(int(r["timestamp"]), {}).get("funding_rate")) if isinstance(deriv_by_ts.get(int(r["timestamp"]), {}), dict) else None)
        for r in raw
    ]
    oi_series = [
        (_to_float(deriv_by_ts.get(int(r["timestamp"]), {}).get("open_interest")) if isinstance(deriv_by_ts.get(int(r["timestamp"]), {}), dict) else None)
        for r in raw
    ]
    if _env_bool("FIE_FEATURE_AUDIT", False):
        nn_f = sum(1 for x in funding_series if x is not None)
        nn_oi = sum(1 for x in oi_series if x is not None)
        print(
            f"[feat-audit] deriv_rows={len(deriv_by_ts)} ts_i={ts_i} d_keys={sorted(d.keys()) if isinstance(d, dict) else []} "
            f"funding_nonnull={nn_f}/{len(funding_series)} oi_nonnull={nn_oi}/{len(oi_series)}",
            flush=True,
        )
    signals.update(multi_timeframe_derivatives_signals(
        funding_series,
        oi_series,
        closes,
        i,
    ))
    signals.update(compute_scenario_signals(signals))

    scen_acc = float(signals.get("scen_accumulation", 0.0))
    scen_bo_c = float(signals.get("scen_breakout_confirmed", 0.0))
    scen_bo_s = float(signals.get("scen_breakout_suspicious", 0.0))
    scen_trend_flow = float(signals.get("scen_trend_flow", 0.0))
    bo_up = float(signals.get("breakout_up", 0.0))
    funding_stress = float(signals.get("deriv_funding_stress_48", 0.0))
    liq_spike = max(float(signals.get("long_liquidations_spike", 0.0)), float(signals.get("deriv_liq_spike_12", 0.0)))
    momentum = max(float(signals.get("momentum_up_1", 0.0)), float(signals.get("momentum_up_5", 0.0)))
    oi_strength = max(float(signals.get("oi_up_price_up", 0.0)), float(signals.get("deriv_oi_stress_24", 0.0)))
    gate_w_acc = _env_float("FIE_GATE_W_ACC", 0.40)
    gate_w_bo = _env_float("FIE_GATE_W_BREAKOUT", 0.30)
    gate_w_oi = _env_float("FIE_GATE_W_OI", 0.20)
    gate_w_mom = _env_float("FIE_GATE_W_MOMENTUM", 0.10)
    gate_w_risk = _env_float("FIE_GATE_W_RISK", 0.25)
    gate_threshold = _env_float("FIE_GATE_THRESHOLD", 0.40)
    if volatility == "high":
        gate_threshold *= _env_float("FIE_GATE_HIGH_VOL_MULT", 1.5)
    # Risk combo: избегаем winner-takes-all (max), иначе gate_score легко схлопывается в несколько уровней.
    # Входы уже нормированы к [0,1]; мягкая агрегация сохраняет непрерывность.
    rw_f = _env_float("FIE_RISK_W_FUNDING", 0.50)
    rw_l = _env_float("FIE_RISK_W_LIQ", 0.30)
    rw_b = _env_float("FIE_RISK_W_BREAKOUT_SUS", 0.20)
    rw_sum = max(rw_f + rw_l + rw_b, 1e-9)
    risk_combo = (rw_f * funding_stress + rw_l * liq_spike + rw_b * scen_bo_s) / rw_sum
    risk_combo = _clamp(float(risk_combo), 0.0, 1.0)
    gate_score = (
        gate_w_acc * scen_acc
        + gate_w_bo * bo_up
        + gate_w_oi * oi_strength
        + gate_w_mom * momentum
        - gate_w_risk * risk_combo
    )

    allow_high_vol = _env_bool("FIE_ALLOW_HIGH_VOL", False)
    is_trend_allowed = (regime == "trend" and (volatility == "low" or allow_high_vol))
    allow = False
    phase = "idle"
    confidence = 0.0

    # Stage 1: regime gating
    if not is_trend_allowed:
        reason = f"regime_block:{regime_key}"
    else:
        prev_phase = str(_market_state.get("phase", "idle"))

        # Stage 2: detection (accumulation)
        if scen_acc > t_acc:
            phase = "accumulation"
            confidence = scen_acc

        # Stage 3: trigger (breakout only after accumulation memory)
        if prev_phase == "accumulation" and bo_up > t_bo:
            phase = "breakout"
            confidence = max(bo_up, scen_bo_c)
            reason = "triggered:accumulation_breakout"
        # Stage 4: trend continuation
        elif momentum > 0.55 and oi_strength > 0.35:
            phase = "trend"
            confidence = min(1.0, 0.5 * momentum + 0.5 * oi_strength)
            reason = f"trend_flow:momentum={momentum:.3f} oi={oi_strength:.3f}"
        elif phase == "accumulation":
            reason = f"accumulating:acc={scen_acc:.3f} t_acc={t_acc:.3f}"
        else:
            reason = f"idle_in_regime:acc={scen_acc:.3f} bo_up={bo_up:.3f}"

        # Soft gating: допускаем вход по интегральному score, а не только по hard правилам.
        if phase in ("accumulation", "breakout", "trend") and gate_score > gate_threshold:
            allow = True
            reason = f"{reason}|soft_gate:{gate_score:.3f}>{gate_threshold:.3f}"

    market_state = _update_market_state(phase, confidence)

    # Чистый эксперимент: не блокировать по Ferrari-gate (idle_in_regime / soft_gate).
    # Торговый цикл сам решает по prob/threshold; см. FIE_SIGNAL_EXPERIMENT в start_prod_loop.sh.
    if _env_bool("FIE_SIGNAL_EXPERIMENT", False) or _env_bool("FIE_BYPASS_ZONE_GATE", False):
        allow = True
        reason = f"{reason}|signal_experiment"

    return EdgeZone(
        allow_prediction=allow,
        state=str(market_state["phase"]),
        market_state=market_state,
        regime=regime,
        volatility=volatility,
        regime_key=regime_key,
        scen_accumulation=scen_acc,
        scen_breakout_confirmed=scen_bo_c,
        scen_breakout_suspicious=scen_bo_s,
        scen_trend_flow=scen_trend_flow,
        breakout_up=bo_up,
        funding_stress=funding_stress,
        liq_spike=liq_spike,
        momentum_up=momentum,
        oi_strength=oi_strength,
        gate_score=gate_score,
        gate_threshold=gate_threshold,
        reason=reason,
    )


def detect_edge(event: str, probability: float | None = None, zone: EdgeZone | None = None) -> dict[str, Any]:
    """
    Legacy API для автономного лупа.
    Теперь возвращает и gating-инфу (allow_prediction) для edge-only режима.
    """
    z = zone if zone is not None else compute_edge_zone()
    return {
        "event": event,
        "allow_prediction": z.allow_prediction,
        "state": z.state,
        "market_state": z.market_state,
        "regime_key": z.regime_key,
        "scen_accumulation": z.scen_accumulation,
        "scen_breakout_confirmed": z.scen_breakout_confirmed,
        "scen_breakout_suspicious": z.scen_breakout_suspicious,
        "scen_trend_flow": z.scen_trend_flow,
        "breakout_up": z.breakout_up,
        "funding_stress": z.funding_stress,
        "liq_spike": z.liq_spike,
        "momentum_up": z.momentum_up,
        "oi_strength": z.oi_strength,
        "gate_score": z.gate_score,
        "gate_threshold": z.gate_threshold,
        "prediction_probability": probability,
        "reason": z.reason,
    }


from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backtesting.metrics import roc_auc
from data.datasets.builder import build_dataset
from learning.role_weights import load_role_stats, load_role_weights


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _role_votes_from_case(case: dict, weights: dict[str, float]) -> list[dict]:
    s = case.get("signals", {})
    acc = float(s.get("scen_accumulation", 0.0))
    bo = float(s.get("breakout_up", 0.0))
    false_bo = float(s.get("scen_breakout_suspicious", 0.0))
    fstress = float(s.get("deriv_funding_stress_48", 0.0))
    liq = max(float(s.get("long_liquidations_spike", 0.0)), float(s.get("deriv_liq_spike_12", 0.0)))
    momentum = max(float(s.get("momentum_up_1", 0.0)), float(s.get("momentum_up_5", 0.0)))
    oi = max(float(s.get("oi_up_price_up", 0.0)), float(s.get("deriv_oi_stress_24", 0.0)))

    smart_money_prob = _clamp01(0.45 + 0.35 * acc + 0.25 * oi - 0.10 * fstress)
    breakout_prob = _clamp01(0.40 + 0.45 * bo + 0.20 * momentum - 0.10 * false_bo)
    risk_prob = _clamp01(0.50 - 0.30 * fstress - 0.25 * liq - 0.20 * false_bo + 0.10 * acc)
    contrarian_prob = _clamp01(0.55 - 0.45 * bo + 0.30 * false_bo + 0.20 * fstress)

    stats = load_role_stats()

    def _w(role: str, fallback: float) -> float:
        st = stats.get(role, {})
        if isinstance(st, dict) and "accuracy" in st:
            return max(0.1, min(1.0, float(st.get("accuracy", 0.5))))
        return max(0.1, float(fallback))

    return [
        {"role": "smart_money", "probability": smart_money_prob, "weight": _w("smart_money", float(weights.get("smart_money", 1.30)))},
        {"role": "breakout_trader", "probability": breakout_prob, "weight": _w("breakout_trader", float(weights.get("breakout_trader", 1.20)))},
        {"role": "risk_manager", "probability": risk_prob, "weight": _w("risk_manager", float(weights.get("risk_manager", 1.40)))},
        {"role": "contrarian", "probability": contrarian_prob, "weight": _w("contrarian", float(weights.get("contrarian", 1.10)))},
    ]


def _weighted_vote(votes: list[dict], disagreement_threshold: float = 0.30) -> tuple[float, float]:
    if not votes:
        return 0.5, 0.0
    ps = []
    ws = []
    for v in votes:
        ps.append(_clamp01(float(v.get("probability", 0.5))))
        ws.append(max(0.0, float(v.get("weight", 1.0))))
    total_w = sum(ws)
    final_p = sum(p * w for p, w in zip(ps, ws)) / total_w if total_w > 1e-12 else 0.5
    disagreement = max(ps) - min(ps)
    conf = _clamp01(1.0 - disagreement)
    if disagreement > disagreement_threshold:
        conf = _clamp01(conf * 0.75)
    return final_p, conf


def run_state_backtest(
    dataset: list[dict],
    t_acc: float = 0.35,
    t_bo: float = 0.35,
    *,
    gate_threshold: float | None = None,
    min_agents_confidence: float | None = None,
    accumulation_duration_n: int | None = None,
    position_size_multiplier: float | None = None,
    regime_mode: str = "strict",
    return_trades: bool = False,
    strategy_profile: str = "default",
) -> dict:
    phase = "idle"
    duration = 0
    weights = load_role_weights()
    min_conf = float(
        min_agents_confidence
        if min_agents_confidence is not None
        else (os.environ.get("FIE_MIN_AGENTS_CONFIDENCE", "0.6") or "0.6")
    )
    gw_acc = float(os.environ.get("FIE_GATE_W_ACC", "0.40") or "0.40")
    gw_bo = float(os.environ.get("FIE_GATE_W_BREAKOUT", "0.30") or "0.30")
    gw_oi = float(os.environ.get("FIE_GATE_W_OI", "0.20") or "0.20")
    gw_mom = float(os.environ.get("FIE_GATE_W_MOMENTUM", "0.10") or "0.10")
    gw_risk = float(os.environ.get("FIE_GATE_W_RISK", "0.25") or "0.25")
    gate_threshold = float(
        gate_threshold if gate_threshold is not None else (os.environ.get("FIE_GATE_THRESHOLD", "0.40") or "0.40")
    )
    acc_n = int(
        accumulation_duration_n
        if accumulation_duration_n is not None
        else int(os.environ.get("FIE_ACCUMULATION_DURATION_N", "3") or "3")
    )
    size_mul = float(
        position_size_multiplier
        if position_size_multiplier is not None
        else float(os.environ.get("FIE_POSITION_SIZE_MULTIPLIER", "1.0") or "1.0")
    )
    t_mom = float(os.environ.get("FIE_A_T_MOM", "0.35") or "0.35")
    t_flow = float(os.environ.get("FIE_A_T_FLOW", "0.35") or "0.35")
    t_pullback = float(os.environ.get("FIE_A_T_PULLBACK", "0.20") or "0.20")
    t_vc = float(os.environ.get("FIE_A_T_VC", "0.25") or "0.25")
    t_acc_strong = float(os.environ.get("FIE_A_T_ACC_STRONG", "0.45") or "0.45")
    t_early_conf = float(os.environ.get("FIE_A_T_EARLY_CONF", "0.70") or "0.70")
    a_reentry_flow = float(os.environ.get("FIE_A_REENTRY_T_FLOW", "0.45") or "0.45")
    a_reentry_cooldown_bars = int(os.environ.get("FIE_A_REENTRY_COOLDOWN", "3") or "3")
    a_hold_flow = float(os.environ.get("FIE_A_HOLD_T_FLOW", "0.50") or "0.50")
    a_event_acc = float(os.environ.get("FIE_A_EVENT_ACC", "0.4") or "0.4")
    a_event_bo = float(os.environ.get("FIE_A_EVENT_BO", "0.3") or "0.3")
    a_event_flow = float(os.environ.get("FIE_A_EVENT_FLOW", "0.4") or "0.4")
    a_hold_extend_flow = float(os.environ.get("FIE_A_HOLD_EXTEND_FLOW", "0.4") or "0.4")
    a_max_ttl = int(os.environ.get("FIE_A_MAX_TTL", "6") or "6")

    traded_probs: list[float] = []
    traded_outcomes: list[int] = []
    predicted_classes: list[int] = []
    trade_sizes: list[float] = []
    trade_logs: list[dict] = []
    a_reentry_cooldown = 0
    in_position_a = False
    a_position_ttl = 0

    for idx, case in enumerate(dataset):
        if a_reentry_cooldown > 0:
            a_reentry_cooldown -= 1
        if a_position_ttl > 0:
            a_position_ttl -= 1
        if a_position_ttl <= 0:
            in_position_a = False
        ctx = case.get("market_context", {})
        regime = str(ctx.get("regime", "unknown"))
        volatility = str(ctx.get("volatility", "unknown"))
        s = case.get("signals", {})

        acc = float(s.get("scen_accumulation", 0.0))
        bo = float(s.get("breakout_up", 0.0))
        bo_s = float(s.get("scen_breakout_suspicious", 0.0))
        fstress = float(s.get("deriv_funding_stress_48", 0.0))
        liq = max(float(s.get("long_liquidations_spike", 0.0)), float(s.get("deriv_liq_spike_12", 0.0)))
        momentum = max(float(s.get("momentum_up_1", 0.0)), float(s.get("momentum_up_5", 0.0)))
        momentum_strength = float(s.get("momentum_strength", 0.0))
        oi = max(float(s.get("oi_up_price_up", 0.0)), float(s.get("deriv_oi_stress_24", 0.0)))
        scen_trend_flow = float(s.get("scen_trend_flow", 0.0))
        mr_long = float(s.get("mean_reversion_long", 0.0))
        vol_comp = float(s.get("volatility_compression", 0.0))
        risk_combo = max(fstress, liq, bo_s)
        gate_score = gw_acc * acc + gw_bo * bo + gw_oi * oi + gw_mom * momentum - gw_risk * risk_combo

        if regime_mode == "strict":
            is_regime_ok = regime == "trend" and volatility == "low"
        elif regime_mode == "broad_low_vol":
            is_regime_ok = volatility == "low" and regime in ("trend", "range")
        else:
            is_regime_ok = regime == "trend" and volatility == "low"
        allow_prediction = False

        if not is_regime_ok:
            phase = "idle"
            duration = 1
            continue

        prev_phase = phase
        # Relaxed accumulation: lower threshold + required duration in regime
        t_acc_low = max(0.0, t_acc - 0.08)
        if acc > t_acc or (acc > t_acc_low and duration >= acc_n):
            phase = "accumulation"
        elif momentum > 0.55 and oi > 0.35:
            phase = "trend"
        elif phase != "accumulation":
            phase = "idle"

        if prev_phase == "accumulation" and bo > t_bo:
            phase = "breakout"
            allow_prediction = True

        if phase in ("accumulation", "breakout", "trend") and gate_score > gate_threshold:
            allow_prediction = True

        # Strategy A: event-driven high-quality scenarios + early entry.
        a_size_profile = 1.0
        if strategy_profile == "A":
            regime_trend = 1.0 if regime == "trend" else 0.0
            volatility_low = 1.0 if volatility == "low" else 0.0
            detect_a_event = (
                (regime_trend > 0.6)
                and (volatility_low > 0.5)
                and (
                    (acc > a_event_acc)
                    or (bo > a_event_bo)
                    or (scen_trend_flow > a_event_flow)
                )
            )
            breakout_continuation = bo > t_bo and momentum_strength > t_mom
            reaccumulation_in_trend = (regime == "trend") and (volatility == "low") and (mr_long > t_pullback) and (vol_comp > t_vc)
            strong_trend_flow = scen_trend_flow > t_flow
            accumulation_only = acc > t_acc

            high_quality_a = breakout_continuation or reaccumulation_in_trend or strong_trend_flow
            # Event-gated Strategy A: only active on detected event windows.
            if detect_a_event:
                allow_prediction = bool(high_quality_a and gate_score > (gate_threshold * 0.8))
            else:
                allow_prediction = False
            if breakout_continuation:
                a_size_profile = 1.0
            elif strong_trend_flow:
                a_size_profile = 0.7
            elif accumulation_only or reaccumulation_in_trend:
                a_size_profile = 0.5
            # Hold extension keeps A "alive" inside continuing trend flow.
            if in_position_a and scen_trend_flow > a_hold_extend_flow:
                allow_prediction = True
                a_position_ttl = min(a_max_ttl, a_position_ttl + 1)

        duration = duration + 1 if phase == prev_phase else 1

        if not allow_prediction:
            continue

        votes = _role_votes_from_case(case, weights)
        prob, conf = _weighted_vote(votes)
        if strategy_profile == "A":
            # Early entry: сильное accumulation + высокая согласованность агентов.
            if (acc > t_acc_strong) and (conf > t_early_conf):
                allow_prediction = True
                # small early entry
                a_size_profile = min(a_size_profile, 0.5)
        pred = 1 if prob >= 0.5 else 0
        actual = int(case.get("outcome", 0))
        # Confidence as position size (no hard skip): layered entries.
        base_size = _clamp01(conf) * max(0.0, size_mul)
        if strategy_profile == "A":
            base_size *= _clamp01(a_size_profile)
        if conf < min_conf:
            # При низком согласии не обнуляем, а уменьшаем размер.
            base_size *= 0.5

        entry_sizes: list[float] = []
        if strategy_profile == "A":
            # Event-based A entries with differentiated sizing.
            if bo > t_bo and momentum_strength > t_mom:
                entry_sizes.append(1.00 * base_size)  # breakout confirmed
            elif scen_trend_flow > t_flow:
                entry_sizes.append(0.70 * base_size)  # trend flow
            elif acc > t_acc:
                entry_sizes.append(0.50 * base_size)  # accumulation-only
            elif allow_prediction:
                entry_sizes.append(0.35 * base_size)  # fallback small
        else:
            if bo > t_bo:
                entry_sizes.append(0.50 * base_size)   # breakout_1: small entry
            if bo > (t_bo + 0.08):
                entry_sizes.append(0.75 * base_size)   # breakout_2: add position
            if momentum > 0.55 and oi > 0.35:
                entry_sizes.append(1.00 * base_size)   # trend confirmed: scale in
            if not entry_sizes and allow_prediction:
                entry_sizes.append(0.35 * base_size)   # soft-gate fallback micro entry

        for sz in entry_sizes:
            sz = _clamp01(sz)
            if sz <= 1e-9:
                continue
            traded_probs.append(prob)
            traded_outcomes.append(actual)
            predicted_classes.append(pred)
            trade_sizes.append(sz)
            if return_trades:
                entry_type = "default"
                if strategy_profile == "A":
                    if (acc > t_acc_strong) and (conf > t_early_conf):
                        entry_type = "early_entry"
                    elif bo > t_bo and momentum_strength > t_mom:
                        entry_type = "breakout_continuation"
                    elif scen_trend_flow > t_flow:
                        entry_type = "trend_flow"
                    elif acc > t_acc:
                        entry_type = "accumulation"
                trade_logs.append(
                    {
                        "entry_time": idx,
                        "regime": f"{regime}_{volatility}",
                        "scenario": {
                            "scen_accumulation": acc,
                            "breakout_up": bo,
                            "scen_trend_flow": scen_trend_flow,
                            "mean_reversion_long": mr_long,
                            "volatility_compression": vol_comp,
                            "scen_breakout_suspicious": bo_s,
                            "funding_stress": fstress,
                            "liq_spike": liq,
                            "momentum_up": momentum,
                            "momentum_strength": momentum_strength,
                            "oi_strength": oi,
                            "gate_score": gate_score,
                        },
                        "agents_votes": votes,
                        "confidence": conf,
                        "position_size": sz,
                        "predicted_probability": prob,
                        "predicted_class": pred,
                        "result": int(pred == actual),
                        "actual_outcome": actual,
                        "entry_type": entry_type,
                    }
                )
            if strategy_profile == "A":
                in_position_a = True
                a_position_ttl = a_max_ttl

        # A_reentry_mode: внутри тренда допускаем повторный вход по trend_flow.
        if strategy_profile == "A":
            allow_reentry_a = (phase == "trend") and (scen_trend_flow > a_reentry_flow) and (a_reentry_cooldown <= 0)
            if allow_reentry_a:
                reentry_size = _clamp01(_clamp01(conf) * max(0.0, size_mul) * 0.6)
                if reentry_size > 1e-9:
                    traded_probs.append(prob)
                    traded_outcomes.append(actual)
                    predicted_classes.append(pred)
                    trade_sizes.append(reentry_size)
                    a_reentry_cooldown = max(1, a_reentry_cooldown_bars)
                    if return_trades:
                        trade_logs.append(
                            {
                                "entry_time": idx,
                                "regime": f"{regime}_{volatility}",
                                "scenario": {
                                    "scen_accumulation": acc,
                                    "breakout_up": bo,
                                    "scen_trend_flow": scen_trend_flow,
                                    "gate_score": gate_score,
                                },
                                "agents_votes": votes,
                                "confidence": conf,
                                "position_size": reentry_size,
                                "predicted_probability": prob,
                                "predicted_class": pred,
                                "result": int(pred == actual),
                                "actual_outcome": actual,
                                "entry_type": "a_reentry",
                            }
                        )

            # Time-in-trade expansion: если тренд продолжается, увеличиваем эффективный size.
            if (phase == "trend") and (scen_trend_flow > a_hold_flow):
                hold_boost = 1.15
                if trade_sizes:
                    trade_sizes[-1] = _clamp01(trade_sizes[-1] * hold_boost)

    n = len(traded_probs)
    if n == 0:
        out = {
            "auc": float("nan"),
            "precision": 0.0,
            "trades_count": 0,
            "winrate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "ev": 0.0,
            "score": 0.0,
        }
        if return_trades:
            out["trades"] = []
        return out

    tp_w = sum(sz for p, a, sz in zip(predicted_classes, traded_outcomes, trade_sizes) if p == 1 and a == 1)
    fp_w = sum(sz for p, a, sz in zip(predicted_classes, traded_outcomes, trade_sizes) if p == 1 and a == 0)
    precision = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else 0.0
    win_weight = sum(sz for p, a, sz in zip(predicted_classes, traded_outcomes, trade_sizes) if p == a)
    total_weight = sum(trade_sizes) if trade_sizes else float(n)
    winrate = win_weight / total_weight if total_weight > 1e-12 else 0.0
    wins = sum(1 for p, a in zip(predicted_classes, traded_outcomes) if p == a)
    losses = n - wins
    avg_win = (win_weight / wins) if wins > 0 else 0.0
    loss_weight = total_weight - win_weight
    avg_loss = (loss_weight / losses) if losses > 0 else 0.0
    ev = (winrate * avg_win) - ((1.0 - winrate) * avg_loss)
    score = winrate * math.log(n) if n > 0 else 0.0

    auc = roc_auc(traded_probs, traded_outcomes)
    out = {
        "auc": auc if isinstance(auc, float) else float("nan"),
        "precision": precision,
        "trades_count": n,
        "winrate": winrate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "ev": ev,
        "score": score,
    }
    if return_trades:
        out["trades"] = trade_logs
    return out


def main() -> None:
    min_candles = int(os.environ.get("FIE_MIN_CANDLES", "4000") or "4000")
    horizon = int(os.environ.get("FIE_HORIZON", "1") or "1")
    t_acc = float(os.environ.get("FIE_T_ACCUMULATION", "0.35") or "0.35")
    t_bo = float(os.environ.get("FIE_T_BREAKOUT", str(t_acc)) or str(t_acc))

    dataset = build_dataset(min_candles=min_candles, horizon=horizon, horizons=[horizon])
    result = run_state_backtest(dataset, t_acc=t_acc, t_bo=t_bo)

    auc = result["auc"]
    auc_str = "n/a" if math.isnan(float(auc)) else f"{float(auc):.4f}"
    print("State-based results:\n")
    print(f"AUC: {auc_str}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Trades count: {result['trades_count']}")
    print(f"Winrate: {result['winrate']:.4f}")
    print(f"EV: {result['ev']:.4f}")
    print(f"Score: {result['score']:.4f}")


if __name__ == "__main__":
    main()

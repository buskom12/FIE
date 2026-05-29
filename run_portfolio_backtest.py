from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from data.datasets.builder import build_dataset
from run_state_backtest import run_state_backtest
from strategy_c_filter import strategy_c_filter, strategy_c_score


def _split_walk_forward(data: list[dict], train: float = 0.6, test: float = 0.2) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(data)
    i1 = int(n * train)
    i2 = int(n * (train + test))
    return data[:i1], data[i1:i2], data[i2:]


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _trade_pnl(trade: dict[str, Any]) -> float:
    size = float(trade.get("position_size", 0.0))
    return size if int(trade.get("result", 0)) == 1 else -size


def _equity_metrics(pnls: list[float]) -> tuple[float, float]:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)
    return equity, max_dd


def _sharpe(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    n = len(pnls)
    mean = sum(pnls) / n
    var = sum((x - mean) ** 2 for x in pnls) / n
    std = math.sqrt(var)
    if std <= 1e-12:
        return 0.0
    return (mean / std) * math.sqrt(n)


def _run_strategy(ds: list[dict], strategy: str) -> dict[str, Any]:
    if strategy == "A":
        return run_state_backtest(
            ds,
            t_acc=0.45,
            t_bo=0.25,
            gate_threshold=0.40,
            min_agents_confidence=0.40,
            position_size_multiplier=1.0,
            regime_mode="strict",
            accumulation_duration_n=3,
            return_trades=True,
        )
    return run_state_backtest(
        ds,
        t_acc=0.20,
        t_bo=0.25,
        gate_threshold=0.30,
        min_agents_confidence=0.40,
        position_size_multiplier=0.5,
        regime_mode="broad_low_vol",
        accumulation_duration_n=1,
        return_trades=True,
    )


def _optimize_strategy_a(train_ds: list[dict], test_ds: list[dict], forward_ds: list[dict]) -> dict[str, float]:
    best: dict[str, float] | None = None
    for t_acc in [0.35, 0.40, 0.45, 0.50]:
        for t_bo in [0.20, 0.25, 0.30]:
            for gate in [0.30, 0.40, 0.50]:
                tr = run_state_backtest(
                    train_ds,
                    t_acc=t_acc,
                    t_bo=t_bo,
                    gate_threshold=gate,
                    min_agents_confidence=0.40,
                    position_size_multiplier=1.0,
                    regime_mode="strict",
                    accumulation_duration_n=3,
                    return_trades=False,
                    strategy_profile="A",
                )
                te = run_state_backtest(
                    test_ds,
                    t_acc=t_acc,
                    t_bo=t_bo,
                    gate_threshold=gate,
                    min_agents_confidence=0.40,
                    position_size_multiplier=1.0,
                    regime_mode="strict",
                    accumulation_duration_n=3,
                    return_trades=False,
                    strategy_profile="A",
                )
                fw = run_state_backtest(
                    forward_ds,
                    t_acc=t_acc,
                    t_bo=t_bo,
                    gate_threshold=gate,
                    min_agents_confidence=0.40,
                    position_size_multiplier=1.0,
                    regime_mode="strict",
                    accumulation_duration_n=3,
                    return_trades=False,
                    strategy_profile="A",
                )
                # MUST: keep A alive in forward
                if int(fw.get("trades_count", 0)) < 5:
                    continue
                score = (
                    float(te.get("score", 0.0)) * 0.6
                    + float(fw.get("score", 0.0)) * 0.3
                    + float(tr.get("score", 0.0)) * 0.1
                )
                cand = {
                    "t_acc": float(t_acc),
                    "t_bo": float(t_bo),
                    "gate": float(gate),
                    "score": float(score),
                }
                if best is None or cand["score"] > best["score"]:
                    best = cand

    if best is None:
        # fallback if no config passes forward min_trades
        return {"t_acc": 0.40, "t_bo": 0.25, "gate": 0.40}
    return {"t_acc": best["t_acc"], "t_bo": best["t_bo"], "gate": best["gate"]}


def _run_strategy_a_with_params(ds: list[dict], params: dict[str, float]) -> dict[str, Any]:
    return run_state_backtest(
        ds,
        t_acc=float(params["t_acc"]),
        t_bo=float(params["t_bo"]),
        gate_threshold=float(params["gate"]),
        min_agents_confidence=0.40,
        position_size_multiplier=1.0,
        regime_mode="strict",
        accumulation_duration_n=3,
        return_trades=True,
        strategy_profile="A",
    )


def compute_d_score(trade: dict[str, Any]) -> int:
    scen = trade.get("scenario", {}) if isinstance(trade.get("scenario"), dict) else {}
    confidence = float(trade.get("confidence", 0.0))
    gate_score = float(scen.get("gate_score", 0.0))
    momentum_up = float(scen.get("momentum_up", 0.0))

    score = 0
    if confidence >= 0.47:
        score += 1
    if gate_score <= -0.08:
        score += 1
    if momentum_up <= 0.24:
        score += 1
    return score


def main() -> None:
    min_candles = int(os.environ.get("FIE_MIN_CANDLES", "4000") or "4000")
    horizon = int(os.environ.get("FIE_HORIZON", "1") or "1")
    c_mode = os.environ.get("FIE_C_MODE", "score").strip().lower()  # hard | score | off
    d_mode = os.environ.get("FIE_D_MODE", "info").strip().lower()   # off | info | tail
    d_tail_dd = float(os.environ.get("FIE_D_TAIL_DD", "0.35") or "0.35")
    d_tail_coeff = float(os.environ.get("FIE_D_TAIL_COEFF", "0.07") or "0.07")
    d_tail_floor = float(os.environ.get("FIE_D_TAIL_FLOOR", "0.86") or "0.86")

    dataset = build_dataset(min_candles=min_candles, horizon=horizon, horizons=[horizon])
    train_ds, test_ds, forward_ds = _split_walk_forward(dataset, train=0.6, test=0.2)

    a_params = _optimize_strategy_a(train_ds, test_ds, forward_ds)
    res_a = _run_strategy_a_with_params(forward_ds, a_params)
    res_b = _run_strategy(forward_ds, "B")

    tagged: list[dict[str, Any]] = []
    for t in res_a.get("trades", []):
        x = dict(t)
        x["strategy"] = "A"
        tagged.append(x)
    for t in res_b.get("trades", []):
        x = dict(t)
        x["strategy"] = "B"
        tagged.append(x)
    tagged.sort(key=lambda t: int(t.get("entry_time", 0)))
    a_entry_times = {int(t.get("entry_time", -1)) for t in res_a.get("trades", [])}

    pnl_series: list[float] = []
    wins = 0
    logs: list[dict[str, Any]] = []
    equity = 0.0
    peak = 0.0
    trades_a_count = 0
    trades_b_count = 0
    wins_a_count = 0
    pnl_a_total = 0.0
    score_c_dist: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    score_d_dist: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    alpha_score_dist: dict[float, int] = {}

    for t in tagged:
        scen = t.get("scenario", {}) if isinstance(t.get("scenario"), dict) else {}
        gate_score = float(scen.get("gate_score", 0.0))
        confidence = float(t.get("confidence", 0.0))
        edge_score = gate_score * confidence

        has_signal_a = int(t.get("entry_time", -1)) in a_entry_times
        if has_signal_a:
            capital_a = _clip(edge_score, 0.3, 0.8)
        else:
            capital_a = 0.0
        capital_b = 1.0 - capital_a
        current_dd = max(0.0, peak - equity)

        # Strategy C integration for B:
        # - hard: binary filter (skip weak B trades)
        # - score: smooth sizing via score_C (keeps more frequency)
        c_pass = True
        c_score = 0
        alpha_multiplier = 1.0
        base_c_mult = 1.0
        c_upper_cap = 1.0
        strong_multiplier = 1.0
        d_score = 0
        d_multiplier = 1.0
        dd_norm = 0.0
        if t.get("strategy") == "B":
            c_pass = strategy_c_filter(t)
            c_score = strategy_c_score(t)
            score_c_dist[int(c_score)] = score_c_dist.get(int(c_score), 0) + 1
            d_score = compute_d_score(t)
            score_d_dist[int(d_score)] = score_d_dist.get(int(d_score), 0) + 1
            if c_mode == "hard":
                if not c_pass:
                    continue
            elif c_mode == "score":
                # C is core alpha. D is used as risk-penalty signal.
                alpha_score = float(c_score)
                alpha_score_key = round(alpha_score, 1)
                alpha_score_dist[alpha_score_key] = alpha_score_dist.get(alpha_score_key, 0) + 1
                base_c_mult = _clip(0.6 + 0.12 * alpha_score, 0.6, 1.2)

                # Smooth adaptive upper-bound under drawdown pressure.
                c_upper_cap = _clip(1.2 - 1.0 * current_dd, 0.85, 1.2)
                alpha_multiplier = _clip(base_c_mult, 0.6, c_upper_cap)
                # Adaptive cap for strong B trades under drawdown stress.
                if int(c_score) >= 2:
                    if current_dd > 0.3:
                        strong_multiplier = 0.8
                    elif current_dd > 0.2:
                        strong_multiplier = 0.9

            # D mode:
            # - off: do not affect PnL (multiplier=1), still log score
            # - info: same as off (diagnostics only)
            # - tail: apply soft penalty only in extreme drawdown regime
            if d_mode == "tail" and current_dd > d_tail_dd:
                dd_norm = _clip(current_dd / 0.4, 0.0, 1.0)
                d_gap = max(0.0, 2.0 - float(d_score))  # 0 for d>=2, 1 for d=1, 2 for d=0
                d_multiplier = _clip(1.0 - d_tail_coeff * d_gap * dd_norm, d_tail_floor, 1.00)
            else:
                d_multiplier = 1.0
                dd_norm = _clip(current_dd / 0.4, 0.0, 1.0)

        # Portfolio-level protection: if DD is elevated, temporarily throttle Strategy B.
        if current_dd > 0.3 and t.get("strategy") == "B":
            capital_b *= 0.5
        alloc = capital_a if t.get("strategy") == "A" else capital_b

        raw_pnl = _trade_pnl(t)
        # Rolling risk-budget by current drawdown
        if current_dd > 0.4:
            risk_multiplier = 0.4
        elif current_dd > 0.2:
            risk_multiplier = 0.7
        else:
            risk_multiplier = 1.0

        contrib = raw_pnl * alloc * risk_multiplier * strong_multiplier * alpha_multiplier * d_multiplier
        equity += contrib
        peak = max(peak, equity)
        pnl_series.append(contrib)
        if contrib > 0:
            wins += 1
        if t.get("strategy") == "A":
            trades_a_count += 1
            pnl_a_total += contrib
            if contrib > 0:
                wins_a_count += 1
        else:
            trades_b_count += 1

        logs.append(
            {
                "entry_time": t.get("entry_time"),
                "strategy": t.get("strategy"),
                "regime": t.get("regime"),
                "scenario": t.get("scenario"),
                "agents_votes": t.get("agents_votes"),
                "confidence": confidence,
                "edge_score": round(edge_score, 6),
                "has_signal_A": has_signal_a,
                "capital_A": round(capital_a, 4),
                "capital_B": round(capital_b, 4),
                "c_mode": c_mode,
                "c_filter_pass": c_pass,
                "c_score": c_score,
                "alpha_multiplier": round(alpha_multiplier, 4),
                "base_c_multiplier": round(base_c_mult, 4),
                "c_upper_cap": round(c_upper_cap, 4),
                "strong_multiplier": round(strong_multiplier, 4),
                "d_score": int(d_score),
                "d_multiplier": round(d_multiplier, 4),
                "dd_norm": round(dd_norm, 4),
                "strategy_a_active": has_signal_a,
                "trade_source": t.get("strategy"),
                "risk_multiplier": risk_multiplier,
                "current_drawdown": round(current_dd, 6),
                "result": t.get("result"),
                "raw_pnl": round(raw_pnl, 6),
                "pnl_contribution": round(contrib, 6),
            }
        )

    trades = len(pnl_series)
    total_pnl, max_dd = _equity_metrics(pnl_series)
    winrate = (wins / trades) if trades > 0 else 0.0
    sharpe = _sharpe(pnl_series)
    winrate_a = (wins_a_count / trades_a_count) if trades_a_count > 0 else 0.0
    ev_a = (pnl_a_total / trades_a_count) if trades_a_count > 0 else 0.0

    log_path = Path(__file__).resolve().parent / "data" / "logs" / "portfolio_trades.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        for row in logs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Portfolio backtest results:\n")
    print(
        "Strategy A params: "
        f"T_acc={a_params['t_acc']:.2f}, "
        f"T_bo={a_params['t_bo']:.2f}, "
        f"gate={a_params['gate']:.2f}"
    )
    print(f"Total PnL: {total_pnl:.6f}")
    print(f"Max Drawdown: {max_dd:.6f}")
    print(f"Sharpe: {sharpe:.4f}")
    print(f"Winrate: {winrate:.4f}")
    print(f"Trades: {trades}")
    print(f"Trades_A: {trades_a_count}")
    print(f"Winrate_A: {winrate_a:.4f}")
    print(f"EV_A: {ev_a:.6f}")
    print(f"Trades_B: {trades_b_count}")
    print(
        "score_C distribution: "
        f"0={score_c_dist.get(0,0)} "
        f"1={score_c_dist.get(1,0)} "
        f"2={score_c_dist.get(2,0)} "
        f"3={score_c_dist.get(3,0)}"
    )
    print(
        "d_score distribution: "
        f"0={score_d_dist.get(0,0)} "
        f"1={score_d_dist.get(1,0)} "
        f"2={score_d_dist.get(2,0)} "
        f"3={score_d_dist.get(3,0)}"
    )
    if alpha_score_dist:
        keys = sorted(alpha_score_dist.keys())
        parts = [f"{k:.1f}={alpha_score_dist[k]}" for k in keys]
        print("alpha_score distribution: " + " ".join(parts))
    print(f"D mode: {d_mode}")
    print(f"\nTrade-level log: {log_path}")


if __name__ == "__main__":
    main()

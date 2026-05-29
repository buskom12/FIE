from __future__ import annotations

import math
import os
from typing import Any

from data.datasets.builder import build_dataset
from run_portfolio_backtest import _clip, compute_d_score
from run_state_backtest import run_state_backtest
from strategy_c_filter import strategy_c_score


def _trade_pnl(trade: dict[str, Any]) -> float:
    size = float(trade.get("position_size", 0.0))
    return size if int(trade.get("result", 0)) == 1 else -size


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


def _equity_and_dd(pnls: list[float]) -> tuple[float, float]:
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = peak - eq
        max_dd = max(max_dd, dd)
    return eq, max_dd


def _portfolio_on_trades(
    trades: list[dict[str, Any]],
    *,
    d_mode: str,
    d_tail_dd: float = 0.35,
    d_tail_coeff: float = 0.07,
    d_tail_floor: float = 0.86,
) -> dict[str, Any]:
    equity = 0.0
    peak = 0.0
    pnls: list[float] = []
    d_hits = 0

    for t in trades:
        # baseline portfolio for B-only (A is ignored in this stress test)
        current_dd = max(0.0, peak - equity)

        # C score scaling (as in portfolio baseline)
        c_score = strategy_c_score(t)
        alpha_score = float(c_score)
        base_c_mult = _clip(0.6 + 0.12 * alpha_score, 0.6, 1.2)
        c_upper_cap = _clip(1.2 - 1.0 * current_dd, 0.85, 1.2)
        alpha_multiplier = _clip(base_c_mult, 0.6, c_upper_cap)

        # strong multiplier (only for strong C under DD)
        strong_multiplier = 1.0
        if int(c_score) >= 2:
            if current_dd > 0.3:
                strong_multiplier = 0.8
            elif current_dd > 0.2:
                strong_multiplier = 0.9

        # risk-budget
        if current_dd > 0.4:
            risk_multiplier = 0.4
        elif current_dd > 0.2:
            risk_multiplier = 0.7
        else:
            risk_multiplier = 1.0

        # D tail mode
        d_multiplier = 1.0
        if d_mode == "tail" and current_dd > d_tail_dd:
            d_score = compute_d_score(t)
            dd_norm = _clip(current_dd / 0.4, 0.0, 1.0)
            d_gap = max(0.0, 2.0 - float(d_score))
            d_multiplier = _clip(1.0 - d_tail_coeff * d_gap * dd_norm, d_tail_floor, 1.00)
            if d_multiplier < 0.9999:
                d_hits += 1

        raw = _trade_pnl(t)
        contrib = raw * risk_multiplier * strong_multiplier * alpha_multiplier * d_multiplier
        equity += contrib
        peak = max(peak, equity)
        pnls.append(contrib)

    total_pnl, max_dd = _equity_and_dd(pnls)
    sharpe = _sharpe(pnls)
    wins = sum(1 for p in pnls if p > 0)
    winrate = wins / len(pnls) if pnls else 0.0
    return {
        "trades": len(pnls),
        "pnl": total_pnl,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "winrate": winrate,
        "d_tail_hits": d_hits,
    }


def _b_trades_for_window(window_ds: list[dict]) -> list[dict[str, Any]]:
    res = run_state_backtest(
        window_ds,
        t_acc=0.20,
        t_bo=0.25,
        gate_threshold=0.30,
        min_agents_confidence=0.40,
        position_size_multiplier=0.5,
        regime_mode="broad_low_vol",
        accumulation_duration_n=1,
        return_trades=True,
    )
    return list(res.get("trades", []))


def main() -> None:
    min_candles = int(os.environ.get("FIE_MIN_CANDLES", "12000") or "12000")
    horizon = int(os.environ.get("FIE_HORIZON", "1") or "1")

    window_len = int(os.environ.get("FIE_STRESS_WINDOW_LEN", "2500") or "2500")
    step = int(os.environ.get("FIE_STRESS_STEP", "500") or "500")
    target_dd = float(os.environ.get("FIE_STRESS_TARGET_DD", "0.35") or "0.35")
    top_n = int(os.environ.get("FIE_STRESS_TOP_N", "5") or "5")
    min_trades = int(os.environ.get("FIE_STRESS_MIN_TRADES", "8") or "8")

    ds = build_dataset(min_candles=min_candles, horizon=horizon, horizons=[horizon])
    n = len(ds)
    if n < window_len:
        print("Dataset too small for stress window.")
        return

    candidates: list[dict[str, Any]] = []
    for i in range(0, n - window_len + 1, step):
        w = ds[i : i + window_len]
        trades = _b_trades_for_window(w)
        stats = _portfolio_on_trades(trades, d_mode="off")
        if stats["trades"] < min_trades:
            continue
        if stats["max_dd"] < target_dd:
            continue
        candidates.append({"start": i, "off": stats})

    if not candidates:
        print("No stress windows found with DD >= target.")
        return

    candidates.sort(key=lambda x: float(x["off"]["max_dd"]), reverse=True)
    top = candidates[: max(1, top_n)]

    print("Tail-D multi-window stress test\n")
    print(f"window_len={window_len} step={step} target_dd>={target_dd} top_n={top_n} min_trades>={min_trades}\n")

    deltas_dd: list[float] = []
    deltas_sharpe: list[float] = []
    deltas_pnl: list[float] = []
    hit_rates: list[float] = []

    print(f"{'start':>6} {'tr':>3} {'DD_off':>7} {'DD_tail':>7} {'ΔDD':>7} "
          f"{'Sh_off':>7} {'Sh_tail':>7} {'ΔSh':>7} "
          f"{'PnL_off':>8} {'PnL_tail':>8} {'hits':>4} {'hit%':>6}")
    print("-" * 86)

    for c in top:
        start = int(c["start"])
        w = ds[start : start + window_len]
        trades = _b_trades_for_window(w)
        off = _portfolio_on_trades(trades, d_mode="off")
        tail = _portfolio_on_trades(trades, d_mode="tail")

        d_dd = float(off["max_dd"]) - float(tail["max_dd"])
        d_sh = float(tail["sharpe"]) - float(off["sharpe"])
        d_pnl = float(tail["pnl"]) - float(off["pnl"])
        hits = int(tail["d_tail_hits"])
        tr = int(off["trades"])
        hit_rate = (hits / tr) if tr > 0 else 0.0

        deltas_dd.append(d_dd)
        deltas_sharpe.append(d_sh)
        deltas_pnl.append(d_pnl)
        hit_rates.append(hit_rate)

        print(
            f"{start:>6} {tr:>3} "
            f"{off['max_dd']:>7.3f} {tail['max_dd']:>7.3f} {d_dd:>+7.3f} "
            f"{off['sharpe']:>7.3f} {tail['sharpe']:>7.3f} {d_sh:>+7.3f} "
            f"{off['pnl']:>8.3f} {tail['pnl']:>8.3f} "
            f"{hits:>4} {hit_rate:>6.1%}"
        )

    def _avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    print("\nSummary (mean over windows):")
    print(f"Avg ΔDD (off - tail): {_avg(deltas_dd):+.4f}  ( >0 = DD improved )")
    print(f"Avg ΔSharpe:         {_avg(deltas_sharpe):+.4f}")
    print(f"Avg ΔPnL:            {_avg(deltas_pnl):+.4f}")
    print(f"Avg tail hit-rate:   {_avg(hit_rates):.2%}")


if __name__ == "__main__":
    main()

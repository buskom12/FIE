from __future__ import annotations

import math
from typing import Any

from data.datasets.builder import build_dataset
from run_state_backtest import run_state_backtest


def _split_walk_forward(data: list[dict], train: float = 0.6, test: float = 0.2) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(data)
    i1 = int(n * train)
    i2 = int(n * (train + test))
    return data[:i1], data[i1:i2], data[i2:]


def _grid_values(start: float, stop: float, step: float) -> list[float]:
    out: list[float] = []
    x = start
    while x < stop - 1e-12:
        out.append(round(x, 2))
        x += step
    return out


def _pick_best_b(train_ds: list[dict], *, min_trades: int = 40) -> dict[str, float]:
    best: dict[str, float] | None = None
    for t_acc in _grid_values(0.20, 0.35, 0.05):
        for t_bo in _grid_values(0.20, 0.35, 0.05):
            for gate in _grid_values(0.20, 0.50, 0.10):
                for conf in _grid_values(0.40, 0.80, 0.10):
                    r = run_state_backtest(
                        train_ds,
                        t_acc=t_acc,
                        t_bo=t_bo,
                        gate_threshold=gate,
                        min_agents_confidence=conf,
                        position_size_multiplier=0.5,
                        regime_mode="broad_low_vol",
                        accumulation_duration_n=1,
                        return_trades=False,
                    )
                    if int(r["trades_count"]) < min_trades:
                        continue
                    cand = {
                        "t_acc": t_acc,
                        "t_bo": t_bo,
                        "gate": gate,
                        "conf": conf,
                        "score": float(r.get("score", 0.0)),
                        "ev": float(r.get("ev", 0.0)),
                        "winrate": float(r.get("winrate", 0.0)),
                    }
                    if best is None:
                        best = cand
                    else:
                        if (cand["score"], cand["ev"], cand["winrate"]) > (best["score"], best["ev"], best["winrate"]):
                            best = cand
    if best is None:
        return {"t_acc": 0.20, "t_bo": 0.25, "gate": 0.30, "conf": 0.40}
    return {k: best[k] for k in ("t_acc", "t_bo", "gate", "conf")}


def _strategy_c_filter(trade: dict[str, Any]) -> bool:
    scen = trade.get("scenario", {}) if isinstance(trade.get("scenario"), dict) else {}
    momentum_up = float(scen.get("momentum_up", 0.0))
    breakout_up = float(scen.get("breakout_up", 0.0))
    confidence = float(trade.get("confidence", 0.0))
    return (
        momentum_up <= 0.2985
        and breakout_up <= 0.4247
        and confidence >= 0.4425
    )


def _equity_stats(trades: list[dict[str, Any]]) -> dict[str, float]:
    if not trades:
        return {"trades": 0, "winrate": 0.0, "ev": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    pnls: list[float] = []
    wins = 0
    for t in trades:
        size = float(t.get("position_size", 0.0))
        pnl = size if int(t.get("result", 0)) == 1 else -size
        pnls.append(pnl)
        if pnl > 0:
            wins += 1

    n = len(pnls)
    ev = sum(pnls) / n
    winrate = wins / n

    mean = ev
    var = sum((x - mean) ** 2 for x in pnls) / n
    std = math.sqrt(var)
    sharpe = ((mean / std) * math.sqrt(n)) if std > 1e-12 else 0.0

    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = peak - eq
        max_dd = max(max_dd, dd)

    return {
        "trades": float(n),
        "winrate": winrate,
        "ev": ev,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


def main() -> None:
    ds = build_dataset(min_candles=4000, horizon=1, horizons=[1])
    train_ds, _, forward_ds = _split_walk_forward(ds, 0.6, 0.2)
    p = _pick_best_b(train_ds, min_trades=40)

    baseline = run_state_backtest(
        forward_ds,
        t_acc=float(p["t_acc"]),
        t_bo=float(p["t_bo"]),
        gate_threshold=float(p["gate"]),
        min_agents_confidence=float(p["conf"]),
        position_size_multiplier=0.5,
        regime_mode="broad_low_vol",
        accumulation_duration_n=1,
        return_trades=True,
    )
    trades_all = list(baseline.get("trades", []))
    trades_filtered = [t for t in trades_all if _strategy_c_filter(t)]

    b = _equity_stats(trades_all)
    f = _equity_stats(trades_filtered)

    print("Strategy B vs B+C filter (forward)\n")
    print(f"Params: T_acc={p['t_acc']:.2f}, T_bo={p['t_bo']:.2f}, gate={p['gate']:.2f}, conf={p['conf']:.2f}\n")
    print("Before (B baseline):")
    print(f"Sharpe: {b['sharpe']:.4f}")
    print(f"Max DD: {b['max_dd']:.6f}")
    print(f"Trades: {int(b['trades'])}")
    print(f"Winrate: {b['winrate']:.4f}")
    print(f"EV: {b['ev']:.6f}\n")
    print("After (B + C filter):")
    print(f"Sharpe: {f['sharpe']:.4f}")
    print(f"Max DD: {f['max_dd']:.6f}")
    print(f"Trades: {int(f['trades'])}")
    print(f"Winrate: {f['winrate']:.4f}")
    print(f"EV: {f['ev']:.6f}")


if __name__ == "__main__":
    main()

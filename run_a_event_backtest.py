from __future__ import annotations

import os
from typing import Any

from data.datasets.builder import build_dataset
from run_state_backtest import run_state_backtest


def detect_a_event(case: dict[str, Any]) -> bool:
    ctx = case.get("market_context", {})
    s = case.get("signals", {})

    regime_trend = 1.0 if str(ctx.get("regime", "")) == "trend" else 0.0
    volatility_low = 1.0 if str(ctx.get("volatility", "")) == "low" else 0.0
    scen_acc = float(s.get("scen_accumulation", 0.0))
    breakout_up = float(s.get("breakout_up", 0.0))
    trend_flow = float(s.get("scen_trend_flow", 0.0))

    return (
        regime_trend > 0.6
        and volatility_low > 0.5
        and (
            scen_acc > 0.4
            or breakout_up > 0.3
            or trend_flow > 0.4
        )
    )


def split_walk_forward(data: list[dict], train: float = 0.6, test: float = 0.2) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(data)
    i1 = int(n * train)
    i2 = int(n * (train + test))
    return data[:i1], data[i1:i2], data[i2:]


def verdict(*, events_detected: int, trades: int, winrate: float, ev: float, coverage: float) -> str:
    if events_detected < 20 or coverage < 0.02:
        return "A не подтверждена: недостаточное event coverage"
    if trades < 5:
        return "A не подтверждена: слишком мало сделок в event-окнах"
    if winrate >= 0.6 and ev > 0:
        return "A имеет edge (event-driven)"
    return "A не подтверждена (event-driven edge неустойчив)"


def main() -> None:
    min_candles = int(os.environ.get("FIE_MIN_CANDLES", "12000") or "12000")
    horizon = int(os.environ.get("FIE_HORIZON", "1") or "1")

    ds = build_dataset(min_candles=min_candles, horizon=horizon, horizons=[horizon])
    _, _, forward = split_walk_forward(ds, train=0.6, test=0.2)

    events = [c for c in forward if detect_a_event(c)]
    events_detected = len(events)
    coverage = (events_detected / len(forward)) if forward else 0.0

    # A-only event-driven execution profile.
    res_a = run_state_backtest(
        events,
        t_acc=0.40,
        t_bo=0.25,
        gate_threshold=0.40,
        min_agents_confidence=0.40,
        position_size_multiplier=1.0,
        regime_mode="strict",
        accumulation_duration_n=3,
        strategy_profile="A",
        return_trades=True,
    )

    trades = int(res_a.get("trades_count", 0))
    winrate = float(res_a.get("winrate", 0.0))
    ev = float(res_a.get("ev", 0.0))

    print("A event-based backtest\n")
    print(f"A events detected: {events_detected}")
    print(f"A trades from events: {trades}")
    print(f"Winrate_A (events): {winrate:.4f}")
    print(f"EV_A (events): {ev:.6f}")
    print(f"Coverage: {coverage:.4%}")
    print(f"\nVERDICT: {verdict(events_detected=events_detected, trades=trades, winrate=winrate, ev=ev, coverage=coverage)}")


if __name__ == "__main__":
    main()

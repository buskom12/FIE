from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from data.datasets.builder import build_dataset
from run_state_backtest import run_state_backtest


def _split_walk_forward(data: list[dict], train: float = 0.6, test: float = 0.2) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(data)
    i1 = int(n * train)
    i2 = int(n * (train + test))
    return data[:i1], data[i1:i2], data[i2:]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _trade_pnl(trade: dict[str, Any]) -> float:
    size = float(trade.get("position_size", 0.0))
    return size if int(trade.get("result", 0)) == 1 else -size


def _eval_strategy(ds: list[dict], *, strategy: str, return_trades: bool) -> dict[str, Any]:
    if strategy == "A":
        # Sniper mode
        return run_state_backtest(
            ds,
            t_acc=0.45,
            t_bo=0.25,
            gate_threshold=0.40,
            min_agents_confidence=0.40,
            position_size_multiplier=1.0,
            regime_mode="strict",
            accumulation_duration_n=3,
            return_trades=return_trades,
        )
    # Frequency mode
    return run_state_backtest(
        ds,
        t_acc=0.20,
        t_bo=0.25,
        gate_threshold=0.30,
        min_agents_confidence=0.40,
        position_size_multiplier=0.5,
        regime_mode="broad_low_vol",
        accumulation_duration_n=1,
        return_trades=return_trades,
    )


def _weights_v1(gate_score: float, confidence: float) -> tuple[float, float]:
    w_a = max(0.0, min(1.0, gate_score * confidence))
    return w_a, 1.0 - w_a


def _weights_v2(gate_score: float) -> tuple[float, float]:
    w_a = _sigmoid((gate_score - 0.4) * 10.0)
    return w_a, 1.0 - w_a


def _weights_v3(ev_a: float, ev_b: float) -> tuple[float, float]:
    a = max(1e-6, ev_a + 1.0)
    b = max(1e-6, ev_b + 1.0)
    s = a + b
    return a / s, b / s


def _portfolio_from_trades(
    trades_a: list[dict[str, Any]],
    trades_b: list[dict[str, Any]],
    *,
    allocation_mode: str,
    ev_a_init: float,
    ev_b_init: float,
) -> dict[str, Any]:
    tagged: list[dict[str, Any]] = []
    for t in trades_a:
        x = dict(t)
        x["strategy"] = "A"
        tagged.append(x)
    for t in trades_b:
        x = dict(t)
        x["strategy"] = "B"
        tagged.append(x)
    tagged.sort(key=lambda t: int(t.get("entry_time", 0)))

    ev_a_ema = ev_a_init
    ev_b_ema = ev_b_init
    alpha = 0.1
    pnl = 0.0
    wins = 0
    logs: list[dict[str, Any]] = []

    for t in tagged:
        sc = t.get("scenario", {}) if isinstance(t.get("scenario"), dict) else {}
        gate_score = float(sc.get("gate_score", 0.0))
        confidence = float(t.get("confidence", 0.0))

        if allocation_mode == "v1":
            w_a, w_b = _weights_v1(gate_score, confidence)
        elif allocation_mode == "v2":
            w_a, w_b = _weights_v2(gate_score)
        elif allocation_mode == "v3":
            w_a, w_b = _weights_v3(ev_a_ema, ev_b_ema)
        else:
            raise ValueError(f"Unknown allocation_mode: {allocation_mode}")

        raw = _trade_pnl(t)
        strat = str(t["strategy"])
        alloc = w_a if strat == "A" else w_b
        contrib = raw * alloc
        pnl += contrib
        if contrib > 0:
            wins += 1

        # Update EV trackers (for v3 adaptive reallocation).
        if strat == "A":
            ev_a_ema = (1.0 - alpha) * ev_a_ema + alpha * raw
        else:
            ev_b_ema = (1.0 - alpha) * ev_b_ema + alpha * raw

        logs.append(
            {
                "entry_time": t.get("entry_time"),
                "strategy": strat,
                "regime": t.get("regime"),
                "scenario": t.get("scenario"),
                "agents_votes": t.get("agents_votes"),
                "confidence": confidence,
                "raw_pnl": raw,
                "wA": round(w_a, 4),
                "wB": round(w_b, 4),
                "alloc_weight": round(alloc, 4),
                "pnl_contribution": round(contrib, 6),
                "result": t.get("result"),
            }
        )

    n = len(tagged)
    winrate = (wins / n) if n > 0 else 0.0
    ev = (pnl / n) if n > 0 else 0.0
    return {
        "trades": n,
        "winrate": winrate,
        "ev": ev,
        "pnl": pnl,
        "logs": logs,
    }


def main() -> None:
    min_candles = int(os.environ.get("FIE_MIN_CANDLES", "4000") or "4000")
    horizon = int(os.environ.get("FIE_HORIZON", "1") or "1")
    mode = os.environ.get("FIE_META_MODE", "v2").strip().lower()

    dataset = build_dataset(min_candles=min_candles, horizon=horizon, horizons=[horizon])
    _, test_ds, forward_ds = _split_walk_forward(dataset, train=0.6, test=0.2)

    # Unseen calibration for EV-based allocation (v3)
    test_a = _eval_strategy(test_ds, strategy="A", return_trades=False)
    test_b = _eval_strategy(test_ds, strategy="B", return_trades=False)
    ev_a_init = float(test_a.get("ev", 0.0))
    ev_b_init = float(test_b.get("ev", 0.0))

    forward_a = _eval_strategy(forward_ds, strategy="A", return_trades=True)
    forward_b = _eval_strategy(forward_ds, strategy="B", return_trades=True)

    trades_a = list(forward_a.get("trades", []))
    trades_b = list(forward_b.get("trades", []))
    portfolio = _portfolio_from_trades(
        trades_a,
        trades_b,
        allocation_mode=mode,
        ev_a_init=ev_a_init,
        ev_b_init=ev_b_init,
    )

    out_path = Path(__file__).resolve().parent / "data" / "logs" / "meta_portfolio_trades.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in portfolio["logs"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Meta-allocation complete:\n")
    print(f"Mode: {mode}")
    print(f"Forward Strategy A: trades={forward_a['trades_count']} winrate={forward_a['winrate']:.4f} EV={forward_a['ev']:.4f}")
    print(f"Forward Strategy B: trades={forward_b['trades_count']} winrate={forward_b['winrate']:.4f} EV={forward_b['ev']:.4f}")
    print(
        f"Forward Portfolio: trades={portfolio['trades']} "
        f"winrate={portfolio['winrate']:.4f} EV={portfolio['ev']:.4f} PnL={portfolio['pnl']:.4f}"
    )
    print(f"Trade-level log: {out_path}")


if __name__ == "__main__":
    main()

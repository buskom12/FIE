from __future__ import annotations

import json
import math
import os
from pathlib import Path

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


def _pick_best_on_train(train_ds: list[dict], *, min_trades: int = 40) -> dict | None:
    best: dict | None = None
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
                    row = {"t_acc": t_acc, "t_bo": t_bo, "gate": gate, "conf": conf, **r}
                    if best is None:
                        best = row
                    else:
                        cand = (float(row["score"]), float(row["ev"]), float(row["winrate"]))
                        cur = (float(best["score"]), float(best["ev"]), float(best["winrate"]))
                        if cand > cur:
                            best = row
    return best


def _run_with_params(ds: list[dict], params: dict, *, with_trades: bool) -> dict:
    return run_state_backtest(
        ds,
        t_acc=float(params["t_acc"]),
        t_bo=float(params["t_bo"]),
        gate_threshold=float(params["gate"]),
        min_agents_confidence=float(params["conf"]),
        position_size_multiplier=0.5,
        regime_mode="broad_low_vol",
        accumulation_duration_n=1,
        return_trades=with_trades,
    )


def main() -> None:
    min_candles = int(os.environ.get("FIE_MIN_CANDLES", "4000") or "4000")
    horizon = int(os.environ.get("FIE_HORIZON", "1") or "1")
    min_trades_train = int(os.environ.get("FIE_B_MIN_TRADES_TRAIN", "40") or "40")

    dataset = build_dataset(min_candles=min_candles, horizon=horizon, horizons=[horizon])
    train_ds, test_ds, forward_ds = _split_walk_forward(dataset, train=0.6, test=0.2)

    best = _pick_best_on_train(train_ds, min_trades=min_trades_train)
    if best is None:
        print("Strategy B: no valid config on train with required min trades.")
        return

    train_res = _run_with_params(train_ds, best, with_trades=False)
    test_res = _run_with_params(test_ds, best, with_trades=False)
    forward_res = _run_with_params(forward_ds, best, with_trades=True)

    logs_path = Path(__file__).resolve().parent / "data" / "logs" / "strategy_b_trades.jsonl"
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    with logs_path.open("w", encoding="utf-8") as f:
        for t in forward_res.get("trades", []):
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    def _fmt_auc(v: float) -> str:
        return "n/a" if math.isnan(float(v)) else f"{float(v):.4f}"

    print("Strategy B build complete:\n")
    print(f"Best params: T_acc={best['t_acc']:.2f}, T_bo={best['t_bo']:.2f}, gate={best['gate']:.2f}, conf={best['conf']:.2f}")
    print("\nWalk-forward:")
    print(f"Train   | trades={train_res['trades_count']} winrate={train_res['winrate']:.4f} EV={train_res['ev']:.4f} AUC={_fmt_auc(train_res['auc'])}")
    print(f"Test    | trades={test_res['trades_count']} winrate={test_res['winrate']:.4f} EV={test_res['ev']:.4f} AUC={_fmt_auc(test_res['auc'])}")
    print(f"Forward | trades={forward_res['trades_count']} winrate={forward_res['winrate']:.4f} EV={forward_res['ev']:.4f} AUC={_fmt_auc(forward_res['auc'])}")
    print(f"\nTrade-level log: {logs_path}")


if __name__ == "__main__":
    main()

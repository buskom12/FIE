from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.datasets.builder import build_dataset
from run_state_backtest import run_state_backtest


def _grid_values(start: float, stop: float, step: float) -> list[float]:
    out: list[float] = []
    x = start
    while x < stop - 1e-12:
        out.append(round(x, 2))
        x += step
    return out


def main() -> None:
    min_candles = int(os.environ.get("FIE_MIN_CANDLES", "4000") or "4000")
    horizon = int(os.environ.get("FIE_HORIZON", "1") or "1")
    min_trades = int(os.environ.get("FIE_MIN_TRADES", "30") or "30")

    dataset = build_dataset(min_candles=min_candles, horizon=horizon, horizons=[horizon])

    t_values = _grid_values(0.20, 0.50, 0.05)
    gate_values = _grid_values(0.30, 0.70, 0.10)
    conf_values = _grid_values(0.40, 0.80, 0.10)
    best: dict | None = None
    all_rows: list[dict] = []

    for t_acc in t_values:
        for t_bo in t_values:
            for gate in gate_values:
                for conf in conf_values:
                    result = run_state_backtest(
                        dataset,
                        t_acc=t_acc,
                        t_bo=t_bo,
                        gate_threshold=gate,
                        min_agents_confidence=conf,
                    )
                    trades = int(result["trades_count"])
                    if trades < min_trades:
                        continue
                    row = {
                        "t_acc": t_acc,
                        "t_bo": t_bo,
                        "gate": gate,
                        "conf": conf,
                        **result,
                    }
                    all_rows.append(row)
                    if best is None:
                        best = row
                    else:
                        # primary: score; secondary: EV; tertiary: winrate
                        cand = (float(row["score"]), float(row["ev"]), float(row["winrate"]))
                        cur = (float(best["score"]), float(best["ev"]), float(best["winrate"]))
                        if cand > cur:
                            best = row

    print("Grid search results (objective = winrate * log(trades)):\n")
    if not all_rows:
        print(f"No valid configs with trades >= {min_trades}.")
        return

    all_rows.sort(key=lambda r: float(r["score"]), reverse=True)
    print("Top-5 configs:")
    for r in all_rows[:5]:
        auc = r["auc"]
        auc_s = "n/a" if (not isinstance(auc, float) or math.isnan(auc)) else f"{auc:.4f}"
        print(
            f"T_acc={r['t_acc']:.2f} T_bo={r['t_bo']:.2f} | "
            f"gate={r['gate']:.2f} conf={r['conf']:.2f} | "
            f"score={r['score']:.4f} ev={r['ev']:.4f} wr={r['winrate']:.4f} "
            f"trades={r['trades_count']} precision={r['precision']:.4f} auc={auc_s}"
        )

    assert best is not None
    auc = best["auc"]
    auc_s = "n/a" if (not isinstance(auc, float) or math.isnan(auc)) else f"{auc:.4f}"
    print("\nBest config:")
    print(f"T_acc: {best['t_acc']:.2f}")
    print(f"T_bo: {best['t_bo']:.2f}")
    print(f"Gate threshold: {best['gate']:.2f}")
    print(f"Min agents confidence: {best['conf']:.2f}")
    print(f"Score: {best['score']:.4f}")
    print(f"EV: {best['ev']:.4f}")
    print(f"Winrate: {best['winrate']:.4f}")
    print(f"Trades count: {best['trades_count']}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"AUC: {auc_s}")


if __name__ == "__main__":
    main()

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


def _rank(results: list[dict], top_n: int = 3) -> list[dict]:
    return sorted(
        results,
        key=lambda r: (float(r["score"]), float(r["ev"]), float(r["winrate"])),
        reverse=True,
    )[:top_n]


def main() -> None:
    min_candles = int(os.environ.get("FIE_MIN_CANDLES", "4000") or "4000")
    horizon = int(os.environ.get("FIE_HORIZON", "1") or "1")
    dataset = build_dataset(min_candles=min_candles, horizon=horizon, horizons=[horizon])

    # Strategy A: sniper mode (precision-first)
    a_results: list[dict] = []
    for t_acc in _grid_values(0.30, 0.50, 0.05):
        for t_bo in _grid_values(0.25, 0.45, 0.05):
            for gate in _grid_values(0.30, 0.70, 0.10):
                for conf in _grid_values(0.40, 0.80, 0.10):
                    r = run_state_backtest(
                        dataset,
                        t_acc=t_acc,
                        t_bo=t_bo,
                        gate_threshold=gate,
                        min_agents_confidence=conf,
                        position_size_multiplier=1.0,
                        regime_mode="strict",
                    )
                    if 20 <= int(r["trades_count"]) <= 80:
                        a_results.append(
                            {
                                "strategy": "A",
                                "t_acc": t_acc,
                                "t_bo": t_bo,
                                "gate": gate,
                                "conf": conf,
                                "size_mul": 1.0,
                                **r,
                            }
                        )

    # Strategy B: frequency mode + stronger risk control via size scaling
    b_results: list[dict] = []
    for t_acc in _grid_values(0.20, 0.35, 0.05):
        for t_bo in _grid_values(0.20, 0.35, 0.05):
            for gate in _grid_values(0.20, 0.50, 0.10):
                for conf in _grid_values(0.40, 0.80, 0.10):
                    r = run_state_backtest(
                        dataset,
                        t_acc=t_acc,
                        t_bo=t_bo,
                        gate_threshold=gate,
                        min_agents_confidence=conf,
                        position_size_multiplier=0.5,  # risk control for Strategy B
                    )
                    if 80 <= int(r["trades_count"]) <= 200:
                        b_results.append(
                            {
                                "strategy": "B",
                                "t_acc": t_acc,
                                "t_bo": t_bo,
                                "gate": gate,
                                "conf": conf,
                                "size_mul": 0.5,
                                **r,
                            }
                        )
                    elif int(r["trades_count"]) < 80:
                        # Пробуем более широкий режим только для Strategy B.
                        rb = run_state_backtest(
                            dataset,
                            t_acc=t_acc,
                            t_bo=t_bo,
                            gate_threshold=gate,
                            min_agents_confidence=conf,
                            position_size_multiplier=0.5,
                            regime_mode="broad_low_vol",
                            accumulation_duration_n=1,
                        )
                        if 80 <= int(rb["trades_count"]) <= 200:
                            b_results.append(
                                {
                                    "strategy": "B",
                                    "t_acc": t_acc,
                                    "t_bo": t_bo,
                                    "gate": gate,
                                    "conf": conf,
                                    "size_mul": 0.5,
                                    "regime_mode": "broad_low_vol",
                                    **rb,
                                }
                            )

    top_a = _rank(a_results, top_n=3)
    top_b = _rank(b_results, top_n=3)

    print("Strategy A (sniper) top-3:\n")
    if not top_a:
        print("No configs matched target trades range (20-80).\n")
    else:
        for i, r in enumerate(top_a, 1):
            auc = r["auc"]
            auc_s = "n/a" if (not isinstance(auc, float) or math.isnan(auc)) else f"{auc:.4f}"
            print(
                f"{i}) trades={r['trades_count']} winrate={r['winrate']:.4f} ev={r['ev']:.4f} "
                f"score={r['score']:.4f} auc={auc_s} | "
                f"T_acc={r['t_acc']:.2f} T_bo={r['t_bo']:.2f} gate={r['gate']:.2f} conf={r['conf']:.2f}"
            )

    print("\nStrategy B (frequency) top-3:\n")
    if not top_b:
        print("No configs matched target trades range (80-200).")
    else:
        for i, r in enumerate(top_b, 1):
            auc = r["auc"]
            auc_s = "n/a" if (not isinstance(auc, float) or math.isnan(auc)) else f"{auc:.4f}"
            print(
                f"{i}) trades={r['trades_count']} winrate={r['winrate']:.4f} ev={r['ev']:.4f} "
                f"score={r['score']:.4f} auc={auc_s} | "
                f"T_acc={r['t_acc']:.2f} T_bo={r['t_bo']:.2f} gate={r['gate']:.2f} conf={r['conf']:.2f} "
                f"size_mul={r['size_mul']:.2f} regime_mode={r.get('regime_mode', 'strict')}"
            )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


LOG_PATH = Path(__file__).resolve().parent / "data" / "logs" / "strategy_b_trades.jsonl"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _pnl(row: dict[str, Any]) -> float:
    size = float(row.get("position_size", 0.0))
    return size if int(row.get("result", 0)) == 1 else -size


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = int(max(0, min(len(ys) - 1, round((len(ys) - 1) * q))))
    return float(ys[idx])


def _get_feature(row: dict[str, Any], key: str) -> float:
    if key in ("confidence", "predicted_probability", "position_size"):
        return float(row.get(key, 0.0))
    scen = row.get("scenario", {})
    if isinstance(scen, dict):
        return float(scen.get(key, 0.0))
    return 0.0


def _candidate_conditions(rows: list[dict[str, Any]], features: list[str]) -> list[tuple[str, str, float]]:
    conds: list[tuple[str, str, float]] = []
    for feat in features:
        vals = [_get_feature(r, feat) for r in rows]
        for q in (0.35, 0.50, 0.65):
            t = _quantile(vals, q)
            conds.append((feat, ">=", t))
            conds.append((feat, "<=", t))
    return conds


def _satisfy(row: dict[str, Any], cond: tuple[str, str, float]) -> bool:
    feat, op, t = cond
    v = _get_feature(row, feat)
    if op == ">=":
        return v >= t
    return v <= t


def _score_condition(rows: list[dict[str, Any]], cond: tuple[str, str, float], base_winrate: float) -> dict[str, Any]:
    matched = [r for r in rows if _satisfy(r, cond)]
    if not matched:
        return {"support": 0}
    wr = mean(int(r.get("result", 0)) for r in matched)
    support = len(matched) / len(rows)
    lift = wr - base_winrate
    score = lift * math.sqrt(max(len(matched), 1))
    return {
        "feature": cond[0],
        "op": cond[1],
        "threshold": cond[2],
        "support_count": len(matched),
        "support": support,
        "winrate": wr,
        "lift": lift,
        "score": score,
    }


def main() -> None:
    rows = _load_rows(LOG_PATH)
    if not rows:
        print("No strategy_b_trades.jsonl rows found.")
        return

    for r in rows:
        r["_pnl"] = _pnl(r)

    sorted_rows = sorted(rows, key=lambda x: float(x["_pnl"]), reverse=True)
    n = len(sorted_rows)
    k = max(1, int(n * 0.2))
    winners = sorted_rows[:k]
    losers = sorted_rows[-k:]

    base_wr = mean(int(r.get("result", 0)) for r in rows)
    winner_wr = mean(int(r.get("result", 0)) for r in winners)
    loser_wr = mean(int(r.get("result", 0)) for r in losers)

    features = [
        "confidence",
        "predicted_probability",
        "breakout_up",
        "scen_accumulation",
        "gate_score",
        "momentum_up",
        "oi_strength",
        "funding_stress",
        "liq_spike",
        "scen_breakout_suspicious",
    ]

    conds = _candidate_conditions(rows, features)
    scored = [_score_condition(rows, c, base_wr) for c in conds]
    scored = [s for s in scored if s.get("support_count", 0) >= max(3, int(0.1 * n))]
    scored.sort(key=lambda s: float(s["score"]), reverse=True)

    print("Strategy C discovery (data-driven)\n")
    print(f"Total trades: {n}")
    print(f"Base winrate: {base_wr:.4f}")
    print(f"Top20% winners winrate: {winner_wr:.4f}")
    print(f"Bottom20% losers winrate: {loser_wr:.4f}\n")

    print("Top patterns from winning trades:")
    for i, s in enumerate(scored[:8], 1):
        print(
            f"{i}) {s['feature']} {s['op']} {s['threshold']:.4f} | "
            f"wr={s['winrate']:.4f} lift={s['lift']:+.4f} support={s['support_count']}"
        )

    # Build compact Strategy C candidate from top non-conflicting rules.
    picked: list[dict[str, Any]] = []
    used_features: set[str] = set()
    for s in scored:
        feat = str(s["feature"])
        if feat in used_features:
            continue
        picked.append(s)
        used_features.add(feat)
        if len(picked) >= 3:
            break

    print("\nSuggested Strategy C rules (candidate):")
    for s in picked:
        print(f"- {s['feature']} {s['op']} {s['threshold']:.4f}")


if __name__ == "__main__":
    main()

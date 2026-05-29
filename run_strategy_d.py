from __future__ import annotations

import os
from dataclasses import dataclass
from statistics import mean
from typing import Any

from data.datasets.builder import build_dataset
from run_state_backtest import run_state_backtest


@dataclass
class Rule:
    feature: str
    op: str
    threshold: float


def _split_walk_forward(data: list[dict], train: float = 0.6, test: float = 0.2) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(data)
    i1 = int(n * train)
    i2 = int(n * (train + test))
    return data[:i1], data[i1:i2], data[i2:]


def _pnl(trade: dict[str, Any]) -> float:
    size = float(trade.get("position_size", 0.0))
    return size if int(trade.get("result", 0)) == 1 else -size


def _feat(trade: dict[str, Any], key: str) -> float:
    if key in ("confidence", "predicted_probability", "position_size"):
        return float(trade.get(key, 0.0))
    scen = trade.get("scenario", {})
    if isinstance(scen, dict):
        return float(scen.get(key, 0.0))
    return 0.0


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = int(round((len(ys) - 1) * q))
    idx = max(0, min(len(ys) - 1, idx))
    return float(ys[idx])


def _build_rules_from_winners(train_trades: list[dict[str, Any]]) -> list[Rule]:
    if len(train_trades) < 10:
        return [
            Rule("confidence", ">=", 0.44),
            Rule("breakout_up", "<=", 0.42),
            Rule("momentum_up", "<=", 0.29),
        ]

    # top 30% winners by PnL
    trades = [{**t, "_pnl": _pnl(t)} for t in train_trades]
    trades.sort(key=lambda t: float(t["_pnl"]), reverse=True)
    k = max(3, int(len(trades) * 0.3))
    winners = trades[:k]
    losers = trades[-k:]

    # fixed small feature set to avoid overfitting complexity
    feats = ["confidence", "breakout_up", "momentum_up", "gate_score", "funding_stress"]

    candidate_rules: list[Rule] = []
    for f in feats:
        wv = [_feat(t, f) for t in winners]
        lv = [_feat(t, f) for t in losers]
        mw = mean(wv) if wv else 0.0
        ml = mean(lv) if lv else 0.0
        if mw >= ml:
            thr = _quantile(wv, 0.35)
            candidate_rules.append(Rule(f, ">=", thr))
        else:
            thr = _quantile(wv, 0.65)
            candidate_rules.append(Rule(f, "<=", thr))

    # rank by lift between winners and losers hit-rate
    scored: list[tuple[float, Rule]] = []
    for r in candidate_rules:
        w_hit = sum(1 for t in winners if _match_rule(t, r)) / len(winners) if winners else 0.0
        l_hit = sum(1 for t in losers if _match_rule(t, r)) / len(losers) if losers else 0.0
        scored.append((w_hit - l_hit, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    # keep 3 simple rules max
    return [r for _, r in scored[:3]]


def _match_rule(trade: dict[str, Any], rule: Rule) -> bool:
    v = _feat(trade, rule.feature)
    if rule.op == ">=":
        return v >= rule.threshold
    return v <= rule.threshold


def _match_all(trade: dict[str, Any], rules: list[Rule]) -> bool:
    return all(_match_rule(trade, r) for r in rules)


def _metrics(trades: list[dict[str, Any]]) -> dict[str, float]:
    if not trades:
        return {"trades": 0.0, "winrate": 0.0, "ev": 0.0}
    pnls = [_pnl(t) for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    return {
        "trades": float(len(trades)),
        "winrate": wins / len(trades),
        "ev": sum(pnls) / len(pnls),
    }


def _run_b_trades(ds: list[dict]) -> list[dict[str, Any]]:
    res = run_state_backtest(
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
    return list(res.get("trades", []))


def main() -> None:
    min_candles = int(os.environ.get("FIE_MIN_CANDLES", "12000") or "12000")
    horizon = int(os.environ.get("FIE_HORIZON", "1") or "1")

    ds = build_dataset(min_candles=min_candles, horizon=horizon, horizons=[horizon])
    train, test, forward = _split_walk_forward(ds, train=0.6, test=0.2)

    train_b = _run_b_trades(train)
    rules = _build_rules_from_winners(train_b)

    test_b = _run_b_trades(test)
    forward_b = _run_b_trades(forward)

    d_test = [t for t in test_b if _match_all(t, rules)]
    d_forward = [t for t in forward_b if _match_all(t, rules)]

    m_test = _metrics(d_test)
    m_forward = _metrics(d_forward)

    print("Strategy D (data-driven)\n")
    print("Rules:")
    for r in rules:
        print(f"- {r.feature} {r.op} {r.threshold:.4f}")

    print("\nTest (unseen):")
    print(f"Trades: {int(m_test['trades'])}")
    print(f"Winrate: {m_test['winrate']:.4f}")
    print(f"EV: {m_test['ev']:.6f}")

    print("\nForward (unseen):")
    print(f"Trades: {int(m_forward['trades'])}")
    print(f"Winrate: {m_forward['winrate']:.4f}")
    print(f"EV: {m_forward['ev']:.6f}")


if __name__ == "__main__":
    main()

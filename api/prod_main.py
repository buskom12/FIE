from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi import FastAPI

from db.prod.crud import (
    get_current_signals,
    get_latest_snapshot,
    get_recent_timeseries,
    get_trades,
)
from db.prod.engine import get_session, init_db

app = FastAPI(title="FIE — Production API")


def _compute_equity_drop_1h(snaps: list) -> float:
    if len(snaps) < 2:
        return 0.0
    now_ts = snaps[-1].timestamp
    now_eq = float(snaps[-1].equity)
    cutoff = now_ts - timedelta(hours=1)
    anchor_eq = None
    for s in snaps:
        if s.timestamp >= cutoff:
            anchor_eq = float(s.equity)
            break
    if anchor_eq is None or anchor_eq <= 0:
        return 0.0
    return max(0.0, (anchor_eq - now_eq) / anchor_eq)


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/current_signal")
def current_signal():
    with get_session() as session:
        return get_current_signals(session)


@app.get("/current_position")
def current_position():
    with get_session() as session:
        snap = get_latest_snapshot(session)
        if not snap:
            return {"capital_A": 0.0, "capital_B": 0.0, "timestamp": None}
        return {"timestamp": snap.timestamp, "capital_A": snap.capital_A, "capital_B": snap.capital_B}


@app.get("/risk_status")
def risk_status():
    with get_session() as session:
        snap = get_latest_snapshot(session)
        if not snap:
            return {"current_dd": 0.0, "cap_hits": 0, "tail_hits": 0, "timestamp": None}
        return {
            "timestamp": snap.timestamp,
            "current_dd": snap.drawdown,
            "cap_hits": snap.cap_hits,
            "tail_hits": snap.tail_hits,
        }


@app.get("/metrics")
def metrics(window_days: int = 365, series_limit: int = 2000, trades_limit: int = 5000):
    since = datetime.now(timezone.utc) - timedelta(days=window_days)
    with get_session() as session:
        trades = get_trades(session, since=since, limit=max(10, int(trades_limit)))
        snaps = get_recent_timeseries(session, limit=max(10, int(series_limit)))

    pnls = [t.pnl for t in trades if t.pnl is not None]
    wins = [p for p in pnls if p > 0]

    if pnls:
        winrate = len(wins) / len(pnls)
    else:
        winrate = 0.0

    # Sharpe: берём rolling из снапшотов, иначе считаем грубо по дельтам equity
    sharpe = None
    rolling = [s.sharpe_rolling for s in snaps if s.sharpe_rolling is not None]
    if rolling:
        sharpe = float(rolling[-1])
    elif len(snaps) >= 3:
        eq = np.array([s.equity for s in snaps], dtype=float)
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        if rets.size > 1 and np.std(rets) > 0:
            sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(365))

    max_dd = float(max((s.drawdown for s in snaps), default=0.0))
    current_dd = float(snaps[-1].drawdown) if snaps else 0.0
    equity_drop_1h = _compute_equity_drop_1h(snaps)

    # Keep alert thresholds in one place for API+dashboard consumers.
    alert_thresholds = {
        "max_dd": 0.40,
        "min_size": 0.02,
        "max_size": 0.25,
        "max_kelly": 10.0,
        "equity_drop_1h": 0.05,
    }

    return {
        "window_days": window_days,
        "trades": len(pnls),
        "winrate": round(winrate, 4),
        "pnl": float(sum(pnls)) if pnls else 0.0,
        "sharpe": None if sharpe is None else round(float(sharpe), 4),
        "max_dd": round(max_dd, 6),
        "series": {
            "timestamps": [s.timestamp.isoformat() for s in snaps],
            "equity": [float(s.equity) for s in snaps],
            "drawdown": [float(s.drawdown) for s in snaps],
            "sharpe_rolling": [None if s.sharpe_rolling is None else float(s.sharpe_rolling) for s in snaps],
        },
        "trades_series": [
            {
                "timestamp": t.timestamp.isoformat(),
                "strategy": t.strategy,
                "entry_type": t.entry_type,
                "pnl": t.pnl,
                "confidence": t.confidence,
                "gate_score": t.gate_score,
                "score_C": t.score_C,
                "d_score": t.d_score,
                "alpha_multiplier": t.alpha_multiplier,
                "edge": t.edge,
                "edge_score": t.edge_score,
                "size": t.size,
                "variance": t.variance,
                "kelly_fraction": t.kelly_fraction,
                "tail_hit": t.tail_hit,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "side": t.side,
                "holding_steps": t.holding_steps,
                "hold_min": t.hold_min,
                "exit_reason": t.exit_reason,
                "exit_signal": t.exit_signal,
                "timeout_hit": bool(t.timeout_hit) if t.timeout_hit is not None else None,
                "reverse_hit": bool(t.reverse_hit) if t.reverse_hit is not None else None,
                "mfe": t.mfe,
                "mae": t.mae,
                "alert_dd": bool(current_dd > alert_thresholds["max_dd"]),
                "alert_size": bool(
                    t.size is not None
                    and (float(t.size) < alert_thresholds["min_size"] or float(t.size) > alert_thresholds["max_size"])
                ),
                "alert_kelly": bool(
                    t.kelly_fraction is not None and abs(float(t.kelly_fraction)) > alert_thresholds["max_kelly"]
                ),
                "alert_equity_drop": bool(equity_drop_1h > alert_thresholds["equity_drop_1h"]),
            }
            for t in trades
        ],
        "alert_thresholds": alert_thresholds,
        "diagnostics": {
            "score_C": [t.score_C for t in trades if t.score_C is not None],
            "d_score": [t.d_score for t in trades if t.d_score is not None],
            "tail_hit": [bool(t.tail_hit) for t in trades if t.tail_hit is not None],
            "edge": [float(t.edge) for t in trades if t.edge is not None],
            "edge_score": [float(t.edge_score) for t in trades if t.edge_score is not None],
            "size": [float(t.size) for t in trades if t.size is not None],
            "pnl": [float(t.pnl) for t in trades if t.pnl is not None],
            "variance": [float(t.variance) for t in trades if t.variance is not None],
            "kelly_fraction": [float(t.kelly_fraction) for t in trades if t.kelly_fraction is not None],
        },
    }


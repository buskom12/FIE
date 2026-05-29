from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlmodel import Session, desc, select

from db.prod.models import PortfolioSnapshot, RegimeStat, SignalLatest, Trade


def save_trade(session: Session, trade: Trade) -> Trade:
    session.add(trade)
    session.commit()
    session.refresh(trade)
    return trade


def save_snapshot(session: Session, snapshot: PortfolioSnapshot) -> PortfolioSnapshot:
    session.add(snapshot)
    session.commit()
    session.refresh(snapshot)
    return snapshot


def upsert_signal_latest(session: Session, signal: SignalLatest) -> SignalLatest:
    existing = session.exec(select(SignalLatest).where(SignalLatest.strategy == signal.strategy)).first()
    if existing:
        existing.timestamp = signal.timestamp
        existing.regime = signal.regime
        existing.volatility = signal.volatility
        existing.scen_accumulation = signal.scen_accumulation
        existing.breakout_up = signal.breakout_up
        existing.confidence = signal.confidence
        existing.gate_score = signal.gate_score
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return existing

    session.add(signal)
    session.commit()
    session.refresh(signal)
    return signal


def get_current_signals(session: Session) -> dict[str, dict[str, Any]]:
    rows = session.exec(select(SignalLatest)).all()
    return {
        r.strategy: {
            "timestamp": r.timestamp,
            "strategy": r.strategy,
            "regime": r.regime,
            "volatility": r.volatility,
            "scen_accumulation": r.scen_accumulation,
            "breakout_up": r.breakout_up,
            "confidence": r.confidence,
            "gate_score": r.gate_score,
        }
        for r in rows
    }


def get_latest_snapshot(session: Session) -> Optional[PortfolioSnapshot]:
    return session.exec(select(PortfolioSnapshot).order_by(desc(PortfolioSnapshot.timestamp))).first()


def get_trades(
    session: Session,
    since: Optional[datetime] = None,
    limit: int = 10_000,
) -> list[Trade]:
    q = select(Trade)
    if since is not None:
        q = q.where(Trade.timestamp >= since)
    q = q.order_by(desc(Trade.timestamp)).limit(limit)
    return list(session.exec(q).all())


def get_recent_pnls(session: Session, *, strategy: Optional[str] = None, limit: int = 500) -> list[float]:
    q = select(Trade.pnl).where(Trade.pnl.is_not(None))
    if strategy is not None:
        q = q.where(Trade.strategy == strategy)
    q = q.order_by(desc(Trade.timestamp)).limit(int(limit))
    rows = list(session.exec(q).all())
    pnls = [float(x) for x in rows if x is not None]
    return list(reversed(pnls))


def get_recent_size_pnl(
    session: Session,
    *,
    strategy: Optional[str] = None,
    limit: int = 100,
) -> list[tuple[float, float]]:
    """Пары (size, pnl) за последние limit сделок, по времени ascending — для corr(size, pnl)."""
    q = select(Trade.size, Trade.pnl).where(
        Trade.pnl.is_not(None),
        Trade.size.is_not(None),
    )
    if strategy is not None:
        q = q.where(Trade.strategy == strategy)
    q = q.order_by(desc(Trade.timestamp)).limit(int(limit))
    rows = list(session.exec(q).all())
    out: list[tuple[float, float]] = []
    for sz, pnl in rows:
        if sz is None or pnl is None:
            continue
        out.append((float(sz), float(pnl)))
    return list(reversed(out))


def _row_to_stats(r: RegimeStat) -> dict:
    lu = r.last_update_ts
    if lu is None and r.updated_at is not None:
        lu = r.updated_at.timestamp()
    return {
        "trades": float(r.trades),
        "pnl_sum": float(r.pnl_sum),
        "wins": float(r.wins),
        "ema_exp": r.ema_exp,
        "ema_var": float(r.ema_var or 0.0),
        "last_update_ts": lu,
    }


def _merge_regime_stats(a: dict, b: dict) -> dict:
    """Объединяет две записи одного логического bucket (кросс-дневной merge)."""
    ta, tb = float(a["trades"]), float(b["trades"])
    t = ta + tb
    out = {
        "trades": t,
        "pnl_sum": float(a["pnl_sum"]) + float(b["pnl_sum"]),
        "wins": float(a["wins"]) + float(b["wins"]),
    }
    ea, eb = a.get("ema_exp"), b.get("ema_exp")
    if ea is not None and eb is not None and t > 0:
        out["ema_exp"] = (float(ea) * ta + float(eb) * tb) / t
    elif ea is not None:
        out["ema_exp"] = ea
    else:
        out["ema_exp"] = eb
    eva = float(a.get("ema_var") or 0.0)
    evb = float(b.get("ema_var") or 0.0)
    out["ema_var"] = (eva * ta + evb * tb) / t if t > 0 else 0.0
    la = a.get("last_update_ts")
    lb = b.get("last_update_ts")
    if la is not None and lb is not None:
        out["last_update_ts"] = max(float(la), float(lb))
    elif la is not None:
        out["last_update_ts"] = float(la)
    elif lb is not None:
        out["last_update_ts"] = float(lb)
    else:
        out["last_update_ts"] = None
    return out


def _parse_regime_key_to_bucket(regime_key: str) -> Optional[tuple[int, str, str]]:
    """
    Канонический ключ: (hour_utc, var_bucket, scen_bucket).
    Legacy: session_id|hour|var|scen → те же три поля из хвоста.
    """
    parts = regime_key.split("|")
    if len(parts) == 3:
        try:
            return (int(parts[0]), parts[1], parts[2])
        except ValueError:
            return None
    if len(parts) == 4:
        try:
            return (int(parts[1]), parts[2], parts[3])
        except ValueError:
            return None
    return None


def load_regime_stats(session: Session) -> dict[tuple, dict]:
    """Кросс-дневной режим: ключ (hour, var_bucket, scen_bucket). Legacy 4-part строки мержатся по bucket."""
    rows = session.exec(select(RegimeStat)).all()
    result: dict[tuple, dict] = {}
    for r in rows:
        key = _parse_regime_key_to_bucket(r.regime_key)
        if key is None:
            continue
        row = _row_to_stats(r)
        if key in result:
            result[key] = _merge_regime_stats(result[key], row)
        else:
            result[key] = row
    return result


def upsert_regime_stat(session: Session, key: tuple, stats: dict) -> None:
    """key = (hour, var_bucket, scen_bucket). Строка в БД: hour|var_bucket|scen_bucket."""
    if len(key) != 3:
        raise ValueError("regime key must be (hour, var_bucket, scen_bucket)")
    regime_key = f"{key[0]}|{key[1]}|{key[2]}"
    # Убираем legacy-строки session|hour|var|scen для того же bucket — иначе при следующем load
    # счётчики суммируются дважды.
    for r in list(session.exec(select(RegimeStat)).all()):
        if r.regime_key == regime_key:
            continue
        if _parse_regime_key_to_bucket(r.regime_key) == key:
            session.delete(r)
    session.commit()
    existing = session.exec(select(RegimeStat).where(RegimeStat.regime_key == regime_key)).first()
    if existing:
        existing.trades = int(max(0, round(float(stats["trades"]))))
        existing.pnl_sum = float(stats["pnl_sum"])
        existing.wins = int(max(0, round(float(stats["wins"]))))
        existing.ema_exp = stats.get("ema_exp")
        existing.ema_var = float(stats.get("ema_var", 0.0))
        existing.last_update_ts = stats.get("last_update_ts")
        from datetime import datetime, timezone
        existing.updated_at = datetime.now(timezone.utc)
        session.add(existing)
    else:
        session.add(RegimeStat(
            regime_key=regime_key,
            trades=int(max(0, round(float(stats["trades"])))),
            pnl_sum=float(stats["pnl_sum"]),
            wins=int(max(0, round(float(stats["wins"])))),
            ema_exp=stats.get("ema_exp"),
            ema_var=float(stats.get("ema_var", 0.0)),
            last_update_ts=stats.get("last_update_ts"),
        ))
    session.commit()


def get_recent_timeseries(session: Session, limit: int = 10_000) -> list[PortfolioSnapshot]:
    q = select(PortfolioSnapshot).order_by(desc(PortfolioSnapshot.timestamp)).limit(limit)
    rows = list(session.exec(q).all())
    return list(reversed(rows))


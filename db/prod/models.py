from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Trade(SQLModel, table=True):
    __tablename__ = "trades"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=_utcnow, index=True)

    strategy: str = Field(index=True)  # "A" / "B"
    entry_type: str = Field(index=True)  # e.g. "enter", "exit", "reduce", ...
    # regime bucket at entry time (hour|var_bucket|scen_bucket)
    regime_key: Optional[str] = Field(default=None, index=True)

    confidence: Optional[float] = Field(default=None, index=True)
    gate_score: Optional[float] = Field(default=None, index=True)
    score_C: Optional[float] = Field(default=None, index=True)
    d_score: Optional[float] = Field(default=None, index=True)

    alpha_multiplier: Optional[float] = Field(default=None)
    c_size_multiplier: Optional[float] = Field(default=None)
    strong_multiplier: Optional[float] = Field(default=None)

    edge: Optional[float] = Field(default=None, index=True)
    edge_score: Optional[float] = Field(default=None, index=True)
    # Mispricing vs Polymarket (σ(edge_score) vs рыночная цена YES)
    p_model: Optional[float] = Field(default=None, index=True)
    p_market: Optional[float] = Field(default=None, index=True)
    edge_real: Optional[float] = Field(default=None, index=True)
    size: Optional[float] = Field(default=None, index=True)
    variance: Optional[float] = Field(default=None, index=True)
    kelly_fraction: Optional[float] = Field(default=None, index=True)

    tail_hit: Optional[bool] = Field(default=None, index=True)

    # ── Diagnostic features (to debug quantization / upstream discreteness) ──
    # Все в [0,1] (как в market_engine), но могут быть None для старых сделок.
    funding_stress_48: Optional[float] = Field(default=None, index=True)
    liq_spike: Optional[float] = Field(default=None, index=True)
    scen_breakout_suspicious: Optional[float] = Field(default=None, index=True)
    momentum_up: Optional[float] = Field(default=None, index=True)
    oi_strength: Optional[float] = Field(default=None, index=True)

    # ── Allocation context (to separate discovery vs exploitation) ─────────
    hour_utc: Optional[int] = Field(default=None, index=True)
    p_bucket: Optional[float] = Field(default=None, index=True)  # p_model bucket at entry (e.g. floor(p*100)/100)
    hour_weight: Optional[float] = Field(default=None, index=True)
    hour_n: Optional[int] = Field(default=None, index=True)
    is_prior: Optional[bool] = Field(default=None, index=True)
    market_active_min: Optional[float] = Field(default=None, index=True)  # active_min at entry time

    # Realized PnL fields
    entry_price: Optional[float] = Field(default=None)
    exit_price: Optional[float] = Field(default=None)
    side: Optional[str] = Field(default=None)         # "long" | "short"
    holding_steps: Optional[int] = Field(default=None)
    hold_min: Optional[float] = Field(default=None)   # holding time in minutes
    exit_reason: Optional[str] = Field(default=None)  # "tp" | "sl" | "timeout"
    exit_signal: Optional[str] = Field(default=None)  # original exit signal / event
    timeout_hit: Optional[bool] = Field(default=None)
    reverse_hit: Optional[bool] = Field(default=None)
    mfe: Optional[float] = Field(default=None)        # maximum favorable excursion
    mae: Optional[float] = Field(default=None)        # maximum adverse excursion

    pnl: Optional[float] = Field(default=None, index=True)


class PortfolioSnapshot(SQLModel, table=True):
    __tablename__ = "portfolio_snapshots"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=_utcnow, index=True)

    equity: float = Field(index=True)
    drawdown: float = Field(default=0.0, index=True)
    sharpe_rolling: Optional[float] = Field(default=None, index=True)
    capital_A: float = Field(default=0.0)
    capital_B: float = Field(default=0.0)

    # Для risk_status/дашборда (если risk-layer их считает)
    cap_hits: int = Field(default=0)
    tail_hits: int = Field(default=0)


class RegimeStat(SQLModel, table=True):
    """Персистентная статистика по режимам для auto-regime learning."""
    __tablename__ = "regime_stats"

    id: Optional[int] = Field(default=None, primary_key=True)
    # Ключ: "hour|var_bucket|scen_bucket" (кросс-дневной); legacy 4-part в БД мержится при load
    regime_key: str = Field(index=True, unique=True)
    trades: int = Field(default=0)
    pnl_sum: float = Field(default=0.0)
    wins: int = Field(default=0)
    ema_exp: Optional[float] = Field(default=None)
    ema_var: float = Field(default=0.0)
    # Unix timestamp последнего применения decay / обновления (для time decay)
    last_update_ts: Optional[float] = Field(default=None)
    updated_at: datetime = Field(default_factory=_utcnow)


class SignalLatest(SQLModel, table=True):
    __tablename__ = "signals_latest"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=_utcnow, index=True)

    strategy: str = Field(index=True, unique=True)  # "A" / "B"
    regime: Optional[str] = Field(default=None, index=True)
    volatility: Optional[str] = Field(default=None, index=True)  # "low" | "high" | numeric str
    scen_accumulation: Optional[float] = Field(default=None)
    breakout_up: Optional[float] = Field(default=None, index=True)
    confidence: Optional[float] = Field(default=None, index=True)
    gate_score: Optional[float] = Field(default=None, index=True)


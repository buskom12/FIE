from __future__ import annotations

import math
import os
import sqlite3
from pathlib import Path
from typing import Any, List

import altair as alt
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


API_BASE = os.environ.get("FIE_PROD_API", "http://localhost:8001").rstrip("/")
DB_PATH = os.environ.get(
    "FIE_PROD_DB_PATH",
    str(Path(__file__).resolve().parents[1] / "db" / "fie_prod.sqlite"),
)


@st.cache_data(ttl=10)
def _get(path: str, params: dict | None = None):
    r = requests.get(f"{API_BASE}{path}", params=params, timeout=5)
    r.raise_for_status()
    return r.json()


def _load_alert_events_from_db(
    *,
    limit: int = 2000,
    window_hours: float = 3.0,
    strategy_filter: str = "all",
) -> pd.DataFrame:
    """
    Load alert events from production SQLite (trades + portfolio_snapshots).
    Returns long-format: timestamp, alert_type, severity, strategy (optional).

    TailHit: severity = clip(alpha_multiplier * 50 + d_score * 15, 0..100) (градиент по α и d_score).
    Rolling window: only events with timestamp >= now - window_hours (UTC).
    strategy_filter: \"A\" | \"B\" | \"all\" — фильтр по trades.strategy; DD из снапшотов только при \"all\".
    """
    empty = pd.DataFrame(columns=["timestamp", "alert_type", "severity", "strategy", "d_score"])
    if not Path(DB_PATH).exists():
        return empty

    conn = sqlite3.connect(DB_PATH)
    try:
        trades_q = f"""
            SELECT timestamp, strategy, size, kelly_fraction, tail_hit, alpha_multiplier, d_score
            FROM trades
            ORDER BY timestamp DESC
            LIMIT {int(limit)}
        """
        snaps_q = f"""
            SELECT timestamp, drawdown
            FROM portfolio_snapshots
            ORDER BY timestamp DESC
            LIMIT {int(limit)}
        """
        trades_df = pd.read_sql_query(trades_q, conn)
        snaps_df = pd.read_sql_query(snaps_q, conn)
    finally:
        conn.close()

    events: list[pd.DataFrame] = []

    if not trades_df.empty:
        trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], errors="coerce", utc=True)
        trades_df["strategy"] = trades_df["strategy"].astype(str)
        trades_df["size"] = pd.to_numeric(trades_df["size"], errors="coerce")
        trades_df["kelly_fraction"] = pd.to_numeric(trades_df["kelly_fraction"], errors="coerce")
        trades_df["alpha_multiplier"] = pd.to_numeric(trades_df["alpha_multiplier"], errors="coerce")
        trades_df["d_score"] = pd.to_numeric(trades_df["d_score"], errors="coerce")

        sf = (strategy_filter or "all").strip().lower()
        if sf != "all":
            trades_df = trades_df[trades_df["strategy"] == strategy_filter.upper()]

        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=float(window_hours))
        trades_df = trades_df[trades_df["timestamp"] >= cutoff]
        if trades_df.empty:
            pass
        else:
            size_mask = (trades_df["size"] < 0.02) | (trades_df["size"] > 0.25)
            kelly_mask = trades_df["kelly_fraction"].abs() > 10.0
            tail_mask = trades_df["tail_hit"].fillna(0).astype(bool)

            if size_mask.any():
                df = trades_df.loc[size_mask, ["timestamp", "strategy"]].copy()
                df["alert_type"] = "Size"
                df["severity"] = 2.0
                df["d_score"] = math.nan
                events.append(df)
            if kelly_mask.any():
                df = trades_df.loc[kelly_mask, ["timestamp", "strategy"]].copy()
                df["alert_type"] = "Kelly"
                df["severity"] = 3.0
                df["d_score"] = math.nan
                events.append(df)
            if tail_mask.any():
                sub = trades_df.loc[tail_mask].copy()
                df = sub[["timestamp", "strategy", "d_score"]].copy()
                df["alert_type"] = "TailHit"
                am = sub["alpha_multiplier"].fillna(1.0)
                ds = sub["d_score"].fillna(0.0)
                df["severity"] = (am * 50.0 + ds * 15.0).clip(0.0, 100.0)
                df["d_score"] = ds.values
                events.append(df)

    if (strategy_filter or "all").strip().lower() == "all" and not snaps_df.empty:
        snaps_df["timestamp"] = pd.to_datetime(snaps_df["timestamp"], errors="coerce", utc=True)
        snaps_df["drawdown"] = pd.to_numeric(snaps_df["drawdown"], errors="coerce")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=float(window_hours))
        snaps_df = snaps_df[snaps_df["timestamp"] >= cutoff]
        dd_mask = snaps_df["drawdown"] > 0.40
        if dd_mask.any():
            df = snaps_df.loc[dd_mask, ["timestamp"]].copy()
            df["strategy"] = "—"
            df["alert_type"] = "DD"
            df["severity"] = 3.0
            df["d_score"] = math.nan
            events.append(df)

    if not events:
        return empty
    out = pd.concat(events, ignore_index=True).dropna(subset=["timestamp"])
    return out


def _load_strong_tail_markers_from_db(
    *,
    limit: int = 2000,
    window_hours: float = 3.0,
    strategy_filter: str = "all",
    d_score_min: float = 2.0,
) -> pd.DataFrame:
    """Строки trades: tail_hit и d_score >= порога — для синих маркеров на heatmap."""
    cols = ["timestamp", "strategy", "d_score", "alpha_multiplier", "tail_hit"]
    empty = pd.DataFrame(columns=cols)
    if not Path(DB_PATH).exists():
        return empty

    conn = sqlite3.connect(DB_PATH)
    try:
        q = """
            SELECT timestamp, strategy, d_score, alpha_multiplier, tail_hit
            FROM trades
            WHERE tail_hit = 1 AND d_score >= ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        df = pd.read_sql_query(q, conn, params=[float(d_score_min), int(limit)])
    finally:
        conn.close()

    if df.empty:
        return empty

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["strategy"] = df["strategy"].astype(str)
    df["d_score"] = pd.to_numeric(df["d_score"], errors="coerce")
    df["alpha_multiplier"] = pd.to_numeric(df["alpha_multiplier"], errors="coerce")

    sf = (strategy_filter or "all").strip().lower()
    if sf != "all":
        df = df[df["strategy"] == strategy_filter.upper()]

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=float(window_hours))
    df = df[df["timestamp"] >= cutoff]
    return df.dropna(subset=["timestamp"])


def _ema_tail_prob_series(hits: list[float], *, alpha: float = 0.3) -> list[float]:
    """EMA по бинарным хитам: prev = (1-alpha)*prev + alpha*hit."""
    prev = 0.0
    out: list[float] = []
    for h in hits:
        prev = (1.0 - alpha) * prev + alpha * float(h)
        out.append(prev)
    return out


def _tail_intensity_frame(
    trades_df: pd.DataFrame,
    *,
    metrics_payload: dict,
    risk_payload: dict,
) -> pd.DataFrame:
    """
    tail_intensity = tail_hit * dd_norm.
    dd_norm = clip(drawdown / 0.4, 0, 1) как в бэктесте: drawdown берётся из series
    с merge_asof по времени сделки; иначе — текущий current_dd из risk_status.
    """
    h = trades_df.copy()
    if h.empty:
        return h
    if "timestamp" not in h.columns:
        h["dd_norm"] = 0.0
        h["tail_intensity"] = 0.0
        return h
    if "tail_hit" in h.columns:
        h["tail_hit"] = pd.to_numeric(h["tail_hit"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    else:
        h["tail_hit"] = 0.0
    ser = metrics_payload.get("series") or {}
    ts = ser.get("timestamps") or []
    dd = ser.get("drawdown") or []
    if len(ts) >= 1 and len(ts) == len(dd):
        snaps = (
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(ts, utc=True, errors="coerce"),
                    "drawdown": pd.to_numeric(dd, errors="coerce"),
                }
            )
            .dropna(subset=["timestamp"])
            .sort_values("timestamp")
        )
        if snaps.empty:
            dd_now = float((risk_payload or {}).get("current_dd", 0.0))
            h["dd_norm"] = min(1.0, max(0.0, dd_now / 0.4))
        else:
            h["timestamp"] = pd.to_datetime(h["timestamp"], errors="coerce", utc=True)
            h = h.sort_values("timestamp")
            h = pd.merge_asof(h, snaps, on="timestamp", direction="backward")
            h["dd_norm"] = (h["drawdown"].fillna(0.0).clip(0.0, 1.0) / 0.4).clip(0.0, 1.0)
    else:
        dd_now = float((risk_payload or {}).get("current_dd", 0.0))
        h["dd_norm"] = min(1.0, max(0.0, dd_now / 0.4))
    h["tail_intensity"] = h["tail_hit"] * h["dd_norm"]
    return h


def _dyn_adaptive_tail_threshold_scalar(
    dd: float,
    *,
    dd_floor: float,
    dd_cap: float,
    threshold_min: float,
    threshold_max: float,
) -> float:
    """Порог tail_hit как в DynamicTailEventMonitor (live_alerts)."""
    span = dd_cap - dd_floor
    if span <= 1e-12:
        return threshold_min
    dd = float(dd or 0.0)
    dd_norm = (dd - dd_floor) / span
    dd_norm = min(max(dd_norm, 0.0), 1.0)
    return threshold_max - dd_norm * (threshold_max - threshold_min)


def _dyn_adaptive_tail_threshold_series(
    dd: pd.Series,
    *,
    dd_floor: float,
    dd_cap: float,
    threshold_min: float,
    threshold_max: float,
) -> pd.Series:
    span = dd_cap - dd_floor
    if span <= 1e-12:
        return pd.Series(threshold_min, index=dd.index)
    dn = ((dd.astype(float) - dd_floor) / span).clip(0.0, 1.0)
    return threshold_max - dn * (threshold_max - threshold_min)


def _merge_trades_with_series_drawdown(
    trades_df: pd.DataFrame,
    series: dict,
    *,
    fallback_dd: float,
) -> pd.DataFrame:
    """Drawdown на момент сделки (merge_asof) — для adaptive_threshold."""
    out = trades_df.copy()
    if "timestamp" not in out.columns:
        out["drawdown_at_trade"] = float(fallback_dd)
        return out
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = out.sort_values("timestamp")
    ts = series.get("timestamps") or []
    dd = series.get("drawdown") or []
    if len(ts) >= 1 and len(ts) == len(dd):
        snaps = (
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(ts, utc=True, errors="coerce"),
                    "drawdown": pd.to_numeric(dd, errors="coerce"),
                }
            )
            .dropna(subset=["timestamp"])
            .sort_values("timestamp")
        )
        if snaps.empty:
            out["drawdown_at_trade"] = float(fallback_dd)
        else:
            out = pd.merge_asof(out, snaps, on="timestamp", direction="backward")
            out["drawdown_at_trade"] = out["drawdown"].fillna(float(fallback_dd))
            out = out.drop(columns=["drawdown"], errors="ignore")
    else:
        out["drawdown_at_trade"] = float(fallback_dd)
    return out


def _high_risk_streak_series(high_risk: pd.Series) -> pd.Series:
    """Подряд идущие high-risk сделки (счётчик в каждой точке по времени)."""
    run = 0
    out: list[int] = []
    for v in high_risk.astype(bool).tolist():
        if v:
            run += 1
        else:
            run = 0
        out.append(run)
    return pd.Series(out, index=high_risk.index)


def compute_loss_streak(trades: List[dict[str, Any]], *, max_streak: int = 5) -> list[int]:
    """
    Подряд идущие сделки, где срабатывает условие: pnl < 0 или tail_hit.
    trades — в хронологическом порядке (как в trades_series).
    max_streak зарезервирован под будущий cap отображения.
    """
    _ = max_streak
    streaks: list[int] = []
    streak = 0
    for t in trades:
        pnl = t.get("pnl")
        try:
            pnl_neg = pnl is not None and float(pnl) < 0.0
        except (TypeError, ValueError):
            pnl_neg = False
        th = t.get("tail_hit", False)
        if th is True or th == 1:
            is_tail = True
        elif th is False or th == 0 or th is None:
            is_tail = False
        else:
            thn = float(pd.to_numeric(th, errors="coerce") or 0.0)
            is_tail = thn >= 1.0 or thn > 0.5
        if pnl_neg or is_tail:
            streak += 1
        else:
            streak = 0
        streaks.append(streak)
    return streaks


def _compute_custom_streak_rows(
    df_sorted: pd.DataFrame,
    *,
    mode: str,
) -> list[int]:
    """
    Подряд идущие «триггерные» сделки по хронологии (df_sorted по timestamp).
    mode: loss | tail_hit | loss_or_tail (через compute_loss_streak).
    """
    rows = df_sorted.to_dict("records")
    if mode == "loss_or_tail":
        return compute_loss_streak(rows, max_streak=5)

    streak = 0
    out: list[int] = []
    for _, row in df_sorted.iterrows():
        pnl = float(row["pnl"]) if pd.notna(row.get("pnl")) else 0.0
        th = row.get("tail_hit")
        if th is True or th == 1:
            is_tail = True
        elif th is False or th == 0 or th is None:
            is_tail = False
        else:
            thn = float(pd.to_numeric(th, errors="coerce") or 0.0)
            is_tail = thn >= 1.0 or thn > 0.5

        if mode == "loss":
            inc = pnl < 0.0
        elif mode == "tail_hit":
            inc = is_tail
        else:
            inc = (pnl < 0.0) or is_tail

        streak = streak + 1 if inc else 0
        out.append(streak)
    return out


def _load_tail_shadow_bins_from_db(
    *,
    window_hours: float,
    strategy_filter: str,
    time_bin: str,
    limit: int = 5000,
    ema_alpha: float = 0.3,
) -> pd.DataFrame:
    """
    По сделкам в окне: частота tail_hit в каждом time_bin, затем EMA по бинам (forward-looking shadow).
    Колонки: time_bin, ema_tail_prob (0..1).
    """
    empty = pd.DataFrame(columns=["time_bin", "ema_tail_prob"])
    if not Path(DB_PATH).exists():
        return empty

    conn = sqlite3.connect(DB_PATH)
    try:
        q = f"""
            SELECT timestamp, strategy, tail_hit
            FROM trades
            ORDER BY timestamp DESC
            LIMIT {int(limit)}
        """
        df = pd.read_sql_query(q, conn)
    finally:
        conn.close()

    if df.empty:
        return empty

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["strategy"] = df["strategy"].astype(str)
    df["tail_hit"] = df["tail_hit"].fillna(0).astype(float).clip(0, 1)

    sf = (strategy_filter or "all").strip().lower()
    if sf != "all":
        df = df[df["strategy"] == strategy_filter.upper()]

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=float(window_hours))
    df = df[df["timestamp"] >= cutoff]
    if df.empty:
        return empty

    df["time_bin"] = df["timestamp"].dt.floor(time_bin)
    rates = df.groupby("time_bin", as_index=False)["tail_hit"].mean().sort_values("time_bin")
    if rates.empty:
        return empty

    prev = 0.0
    emas: list[float] = []
    for rate in rates["tail_hit"].tolist():
        prev = (1.0 - ema_alpha) * prev + ema_alpha * float(rate)
        emas.append(prev)
    rates["ema_tail_prob"] = emas
    return rates[["time_bin", "ema_tail_prob"]]


st.set_page_config(layout="wide", page_title="AI Trading System — Live", page_icon="⚡")

st.markdown(
    """
<style>
.block-container {
    padding-top: 1rem;
}
div[data-testid="stMetric"] {
    background-color: #1A1F2B;
    padding: 10px;
    border-radius: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("## ⚡ AI Trading System — Live")
st.caption("Production control panel · metrics from `/metrics` & `/risk_status`")

risk = _get("/risk_status")
m = _get("/metrics")

_top1, _top2, _top3, _top4 = st.columns(4)
_sh = m.get("sharpe")
_top1.metric("Sharpe", "n/a" if _sh is None else f"{float(_sh):.2f}")
_top2.metric("Max DD", f"{float(m.get('max_dd', 0.0) or 0.0):.3f}")
_top3.metric("PnL", f"{float(m.get('pnl', 0.0) or 0.0):.3f}")
_top4.metric("Trades", int(m.get("trades", 0) or 0))

_dd_live = float(risk.get("current_dd", 0.0) or 0.0)
if _dd_live > 0.35:
    st.error(f"🔴 HIGH RISK — DD: {_dd_live:.3f}")
elif _dd_live > 0.25:
    st.warning(f"🟠 Elevated Risk — DD: {_dd_live:.3f}")
else:
    st.success(f"🟢 Normal — DD: {_dd_live:.3f}")

# --- Streak-баннер (только /metrics + env, API не меняется) ---
_trades_banner = m.get("trades_series") or []
_N_banner = int(os.environ.get("FIE_DASH_STREAK_ALERT_N", "4"))
_dd_banner = float(os.environ.get("FIE_DASH_STREAK_DD_THRESHOLD", "0"))
_streak_banner_active = False
_streaks_banner: list[int] = []
if _trades_banner:
    _tr_sorted = sorted(_trades_banner, key=lambda x: str(x.get("timestamp") or ""))
    _streaks_banner = compute_loss_streak(_tr_sorted, max_streak=5)
    _last_b = _streaks_banner[-1] if _streaks_banner else 0
    _dd_now_b = float(risk.get("current_dd", 0.0) or 0.0)
    _dd_gate_ok = _dd_banner <= 0.0 or _dd_now_b >= _dd_banner
    if _last_b >= _N_banner and _dd_gate_ok:
        _streak_banner_active = True
        st.warning(
            f"⚠️ High-risk streak detected! Streak = {_last_b} (порог N={_N_banner}). "
            f"Текущий DD = {_dd_now_b:.4f}"
            + (f" (условие DD ≥ {_dd_banner:.2f})" if _dd_banner > 0 else "")
        )

with st.expander("Streak-баннер: env и логика", expanded=False):
    st.markdown(
        f"""
- **`FIE_DASH_STREAK_ALERT_N`** — порог **N** подряд (сейчас **{_N_banner}**). Срабатывание: последний streak ≥ N.
- **`FIE_DASH_STREAK_DD_THRESHOLD`** — если **0** (сейчас **{_dd_banner}**), учитывается только streak; если **> 0**, нужно ещё **current DD ≥ порога**.
- Streak считается как **`compute_loss_streak`**: шаг увеличивается, если **pnl < 0** или **tail_hit**, иначе сброс в 0.
- Состояние баннера сейчас: **{'тревога' if _streak_banner_active else 'нет тревоги'}** (последний streak = **{_streaks_banner[-1] if _streaks_banner else 0}**).
        """.strip()
    )

colL, colR = st.columns([2, 1])

with colR:
    st.subheader("Risk Engine — detail")
    st.metric("Current DD", f"{risk.get('current_dd', 0.0):.4f}")
    st.metric("cap_hits", int(risk.get("cap_hits", 0)))
    st.metric("tail_hits", int(risk.get("tail_hits", 0)))
    st.metric("Winrate", f"{m.get('winrate', 0.0) * 100:.1f}%")

with colL:
    st.subheader("Risk Engine — curves")
    series = (m.get("series") or {})
    ts = series.get("timestamps") or []
    if ts:
        eq_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts),
                "equity": series.get("equity") or [],
                "drawdown": series.get("drawdown") or [],
            }
        )
        fig = px.line(eq_df, x="timestamp", y="equity", title="Equity curve")
        st.plotly_chart(fig, use_container_width=True)
        fig = px.line(eq_df, x="timestamp", y="drawdown", title="Drawdown curve")
        st.plotly_chart(fig, use_container_width=True)

        sr = series.get("sharpe_rolling") or []
        if any(v is not None for v in sr):
            sharpe_df = pd.DataFrame({"timestamp": pd.to_datetime(ts), "sharpe_rolling": sr})
            fig = px.line(sharpe_df, x="timestamp", y="sharpe_rolling", title="Rolling Sharpe")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Пока нет снапшотов в базе (portfolio_snapshots). Запустите `services/prod_loop.py`.")

st.divider()
st.subheader("Alpha Diagnostics — live signals")
signals = _get("/current_signal")
if not signals:
    st.warning("Нет signals_latest в базе.")
else:
    df = pd.DataFrame(list(signals.values()))
    st.dataframe(df, use_container_width=True)

    if "gate_score" in df.columns and df["gate_score"].notna().any():
        fig = px.histogram(df, x="gate_score", nbins=30, title="gate_score distribution")
        st.plotly_chart(fig, use_container_width=True)

    if "confidence" in df.columns and df["confidence"].notna().any():
        fig = px.histogram(df, x="confidence", nbins=30, title="confidence distribution")
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Alpha Diagnostics — trade histograms")
diag = (m.get("diagnostics") or {})
trades_series = m.get("trades_series") or []
score_c = diag.get("score_C") or []
d_score = diag.get("d_score") or []
tail_hit = diag.get("tail_hit") or []
edge = diag.get("edge") or []
edge_score = diag.get("edge_score") or []
size = diag.get("size") or []
pnl = diag.get("pnl") or []
variance = diag.get("variance") or []
kelly_fraction = diag.get("kelly_fraction") or []

cols = st.columns(3)
with cols[0]:
    if score_c:
        st.plotly_chart(px.histogram(pd.DataFrame({"score_C": score_c}), x="score_C", nbins=40, title="score_C distribution"), use_container_width=True)
    else:
        st.caption("Нет score_C в trades.")
with cols[1]:
    if d_score:
        st.plotly_chart(px.histogram(pd.DataFrame({"d_score": d_score}), x="d_score", nbins=40, title="d_score distribution"), use_container_width=True)
    else:
        st.caption("Нет d_score в trades.")
with cols[2]:
    if tail_hit:
        st.plotly_chart(px.histogram(pd.DataFrame({"tail_hit": tail_hit}), x="tail_hit", title="tail_hit"), use_container_width=True)
    else:
        st.caption("Нет tail_hit в trades.")

st.divider()
st.subheader("Alpha Diagnostics — execution")
cols2 = st.columns(3)
with cols2[0]:
    if edge:
        st.plotly_chart(px.histogram(pd.DataFrame({"edge": edge}), x="edge", nbins=40, title="edge distribution"), use_container_width=True)
    else:
        st.caption("Нет edge в trades.")
with cols2[1]:
    if edge_score:
        st.plotly_chart(px.histogram(pd.DataFrame({"edge_score": edge_score}), x="edge_score", nbins=40, title="edge_score distribution"), use_container_width=True)
    else:
        st.caption("Нет edge_score в trades.")
with cols2[2]:
    if size:
        st.plotly_chart(px.histogram(pd.DataFrame({"size": size}), x="size", nbins=40, title="size distribution"), use_container_width=True)
    else:
        st.caption("Нет size в trades.")

cols3 = st.columns(2)
with cols3[0]:
    if variance:
        st.plotly_chart(px.histogram(pd.DataFrame({"variance": variance}), x="variance", nbins=40, title="variance distribution"), use_container_width=True)
    else:
        st.caption("Нет variance в trades.")
with cols3[1]:
    if kelly_fraction:
        st.plotly_chart(px.histogram(pd.DataFrame({"kelly_fraction": kelly_fraction}), x="kelly_fraction", nbins=40, title="kelly_fraction distribution"), use_container_width=True)
    else:
        st.caption("Нет kelly_fraction в trades.")

if pnl and (edge or size):
    st.subheader("Execution Intelligence — PnL relations")
    scols = st.columns(2)
    with scols[0]:
        if edge and len(edge) == len(pnl):
            st.plotly_chart(px.scatter(pd.DataFrame({"edge": edge, "pnl": pnl}), x="edge", y="pnl", title="PnL vs edge"), use_container_width=True)
        else:
            st.caption("PnL vs edge: недостаточно данных/несовпадение длины.")
    with scols[1]:
        if size and len(size) == len(pnl):
            st.plotly_chart(px.scatter(pd.DataFrame({"size": size, "pnl": pnl}), x="size", y="pnl", title="PnL vs size"), use_container_width=True)
        else:
            st.caption("PnL vs size: недостаточно данных/несовпадение длины.")

st.divider()
st.subheader("Tail Detection System — dynamic tail")
st.caption(
    "Порог tail_hit (adaptive_threshold) как в DynamicTailEventMonitor: ниже при высоком DD. "
    "Drawdown у сделки — из equity-серии (merge_asof); high-risk совпадает с prod_loop (size/edge). "
    "Пороги по умолчанию из тех же env, что и в services/prod_loop.py."
)
_dd_f = float(os.environ.get("FIE_TAIL_DD_FLOOR", "0.3"))
_dd_c = float(os.environ.get("FIE_TAIL_DD_CAP", "0.6"))
_tmin = float(os.environ.get("FIE_TAIL_THRESHOLD_MIN", "0.2"))
_tmax = float(os.environ.get("FIE_TAIL_THRESHOLD_MAX", "0.7"))
_sz_th = float(os.environ.get("FIE_TAIL_SMART_SIZE_THRESHOLD", "0.05"))
_es_th = float(os.environ.get("FIE_TAIL_SMART_EDGE_SCORE_THRESHOLD", "0.02"))

series_dyn = m.get("series") or {}
if trades_series:
    _vdf = pd.DataFrame(trades_series)
    for _c in ("pnl", "tail_hit", "size", "edge", "edge_score"):
        if _c in _vdf.columns:
            _vdf[_c] = pd.to_numeric(_vdf[_c], errors="coerce")
    if "edge_score" not in _vdf.columns and "edge" in _vdf.columns:
        _vdf["edge_score"] = _vdf["edge"]
    elif "edge_score" not in _vdf.columns:
        _vdf["edge_score"] = 0.0

    _vdf = _merge_trades_with_series_drawdown(
        _vdf,
        series_dyn,
        fallback_dd=float(risk.get("current_dd", 0.0) or 0.0),
    )
    _vdf["adaptive_threshold"] = _dyn_adaptive_tail_threshold_series(
        _vdf["drawdown_at_trade"].fillna(0.0),
        dd_floor=_dd_f,
        dd_cap=_dd_c,
        threshold_min=_tmin,
        threshold_max=_tmax,
    )
    _th = _vdf["tail_hit"].fillna(0.0)
    _pnl = _vdf["pnl"]
    _sz = _vdf["size"].fillna(0.0)
    _es = _vdf["edge_score"].fillna(0.0)
    _vdf["high_risk_smart"] = (
        (_th > _vdf["adaptive_threshold"]) & (_pnl < 0.0) & (_sz > _sz_th) & (_es > _es_th)
    )
    _vdf = _vdf.sort_values("timestamp")
    _vdf["hr_streak"] = _high_risk_streak_series(_vdf["high_risk_smart"])

    st.markdown(
        f"**Параметры:** dd_floor={_dd_f}, dd_cap={_dd_c}, "
        f"threshold∈[{_tmin},{_tmax}], size>{_sz_th}, edge_score>{_es_th}"
    )
    fig_dyn = px.scatter(
        _vdf,
        x="timestamp",
        y="pnl",
        color="high_risk_smart",
        size="hr_streak",
        size_max=22,
        title="PnL vs time (high-risk подсветка, размер точки = подряд high-risk)",
        hover_data=[
            "strategy",
            "size",
            "edge_score",
            "tail_hit",
            "adaptive_threshold",
            "drawdown_at_trade",
            "hr_streak",
        ],
    )
    st.plotly_chart(fig_dyn, use_container_width=True)

    st.subheader("Tail Detection System — streaks & PnL / size")
    st.caption(
        "Цвет — high-risk (smart-фильтр как в prod_loop); размер пузырька — |edge_score| "
        "(чтобы нулевые edge не схлопывались в точку, задан нижний порог)."
    )
    _vdf["high_risk_streak"] = _vdf["hr_streak"]
    _v_sz = _vdf.dropna(subset=["pnl", "size"]).copy()
    if not _v_sz.empty:
        _v_sz["bubble_edge"] = _v_sz["edge_score"].fillna(0.0).abs().clip(lower=0.002)
        fig_sz = px.scatter(
            _v_sz,
            x="size",
            y="pnl",
            color="high_risk_smart",
            size="bubble_edge",
            size_max=24,
            title="PnL vs Size (high-risk, размер пузырька ∝ |edge_score|)",
            hover_data=[
                "timestamp",
                "strategy",
                "tail_hit",
                "adaptive_threshold",
                "high_risk_streak",
                "edge_score",
                "drawdown_at_trade",
            ],
        )
        st.plotly_chart(fig_sz, use_container_width=True)
        _m1, _m2 = st.columns(2)
        with _m1:
            st.metric(
                "Текущий streak (последняя сделка по времени)",
                int(_vdf["hr_streak"].iloc[-1]),
            )
        with _m2:
            st.metric("Макс. streak в окне", int(_vdf["hr_streak"].max()))
    else:
        st.caption("Нет пар size/pnl для scatter.")

    _ts_list = series_dyn.get("timestamps") or []
    _dd_list = series_dyn.get("drawdown") or []
    if len(_ts_list) == len(_dd_list) and _ts_list:
        _snap_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(_ts_list, utc=True, errors="coerce"),
                "drawdown": pd.to_numeric(_dd_list, errors="coerce"),
            }
        ).dropna(subset=["timestamp"])
        _snap_df["adaptive_threshold"] = _dyn_adaptive_tail_threshold_series(
            _snap_df["drawdown"].fillna(0.0),
            dd_floor=_dd_f,
            dd_cap=_dd_c,
            threshold_min=_tmin,
            threshold_max=_tmax,
        )
        fig_dd = px.line(
            _snap_df,
            x="timestamp",
            y=["drawdown", "adaptive_threshold"],
            title="Drawdown и adaptive tail_hit threshold (по снапшотам)",
        )
        st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.caption("Нет series.timestamps/drawdown — линию DD+threshold не строим.")

    fig_streak = px.line(
        _vdf,
        x="timestamp",
        y="hr_streak",
        title="Consecutive high-risk trades streak (интерактивный счётчик по времени)",
        markers=True,
    )
    st.plotly_chart(fig_streak, use_container_width=True)

    st.subheader("Tail Detection System — streak alerts")
    st.caption(
        "Триггер и порог N — без новых эндпоинтов, только trades_series из /metrics. "
        "Точки с streak ≥ N подсвечиваются; при выполнении условий — предупреждение сверху блока."
    )
    _sam = st.columns([2, 1, 1])
    with _sam[0]:
        _streak_mode = st.selectbox(
            "Триггер streak",
            options=["loss", "tail_hit", "loss_or_tail"],
            index=2,
            format_func=lambda x: {
                "loss": "Убыточные подряд (pnl < 0)",
                "tail_hit": "tail_hit подряд",
                "loss_or_tail": "pnl < 0 или tail_hit (compute_loss_streak)",
            }[x],
            key="dyn_streak_mode",
        )
    with _sam[1]:
        _streak_n = st.slider("Порог N подряд", min_value=2, max_value=10, value=4, key="dyn_streak_n")
    with _sam[2]:
        _dd_gate = st.checkbox("Только при DD ≥ порога", value=False, key="dyn_streak_dd_gate")
    _dd_alert_thr = 0.35
    if _dd_gate:
        _dd_alert_thr = st.number_input(
            "Мин. drawdown (текущий DD из Risk)",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
            format="%.2f",
            key="dyn_streak_dd_thr",
        )

    _sdf = _vdf.sort_values("timestamp").copy()
    _sdf["dyn_streak"] = _compute_custom_streak_rows(_sdf, mode=_streak_mode)
    _sdf["color"] = _sdf["dyn_streak"].apply(
        lambda x: "red" if int(x) >= int(_streak_n) else "blue"
    )
    _sdf["marker_size"] = _sdf["dyn_streak"].apply(
        lambda x: 18 if int(x) >= int(_streak_n) else 9
    )
    for _c in ("edge_score", "edge"):
        if _c in _sdf.columns:
            _sdf[_c] = pd.to_numeric(_sdf[_c], errors="coerce")
    if "edge_score" not in _sdf.columns:
        _sdf["edge_score"] = _sdf["edge"] if "edge" in _sdf.columns else 0.0

    _scatter_hs = _sdf.dropna(subset=["pnl", "edge_score"])
    if not _scatter_hs.empty:
        fig_hs = px.scatter(
            _scatter_hs,
            x="edge_score",
            y="pnl",
            color="color",
            color_discrete_map={"red": "#d32f2f", "blue": "#1565c0"},
            size="marker_size",
            size_max=22,
            title="PnL vs edge_score (red/blue: streak ≥ N — красный, крупнее маркер)",
            hover_data=[
                "timestamp",
                "strategy",
                "size",
                "variance",
                "kelly_fraction",
                "dyn_streak",
                "tail_hit",
            ],
            height=420,
        )
        fig_hs.update_traces(marker=dict(line=dict(width=0.8, color="white")))
        st.plotly_chart(fig_hs, use_container_width=True)
    else:
        st.caption("Нет данных для scatter edge_score vs pnl.")

    fig_st2 = px.line(
        _sdf,
        x="timestamp",
        y="dyn_streak",
        title="Dynamic streak по времени (выбранный триггер)",
        markers=True,
    )
    st.plotly_chart(fig_st2, use_container_width=True)

    _last_s = int(_sdf["dyn_streak"].iloc[-1]) if not _sdf.empty else 0
    _dd_now = float(risk.get("current_dd", 0.0) or 0.0)
    _dd_ok = (not _dd_gate) or (_dd_now >= float(_dd_alert_thr))
    if _last_s >= _streak_n and _dd_ok:
        st.warning(
            f"⚠️ High-risk streak: последний счётчик = **{_last_s}** (порог N = {_streak_n}, "
            f"режим `{_streak_mode}`). Текущий DD = {_dd_now:.4f}"
            + (f" (условие DD ≥ {_dd_alert_thr:.2f} выполнено)" if _dd_gate else "")
        )
    else:
        st.caption(
            f"Алерт не активен: streak={_last_s}, N={_streak_n}, DD={_dd_now:.4f}"
            + (f" (для тревоги нужен ещё DD ≥ {_dd_alert_thr:.2f})" if _dd_gate else "")
        )

else:
    st.caption("Нет trades_series — Dynamic Tail визуализация недоступна.")

st.divider()
st.subheader("Execution Intelligence — unified dashboard")
st.caption(
    "Один набор сделок: PnL vs Kelly, Size vs PnL с tail-shadow, Variance vs Size. "
    "Фильтры в боковой панели: strategy, только tail-hit, минимум d_score; tail-shadow на графиках пересчитывается по отфильтрованной хронологии. "
    "Hover: edge, edge_score, d_score, tail_hit, variance, kelly, size, pnl, strategy."
)
if trades_series:
    tdf = pd.DataFrame(trades_series)
    for c in ("pnl", "kelly_fraction", "size", "variance", "edge", "edge_score", "d_score"):
        if c in tdf.columns:
            tdf[c] = pd.to_numeric(tdf[c], errors="coerce")
    tdf["timestamp"] = pd.to_datetime(tdf["timestamp"], errors="coerce", utc=True)
    tdf = tdf.sort_values("timestamp")
    if "tail_hit" in tdf.columns:
        tdf["tail_hit_num"] = pd.to_numeric(tdf["tail_hit"], errors="coerce").fillna(0.0).clip(0, 1)
        tdf["tail_shadow"] = _ema_tail_prob_series(tdf["tail_hit_num"].tolist(), alpha=0.3)
    else:
        tdf["tail_shadow"] = 0.0
    tdf["tail_alpha"] = (0.2 + 0.6 * tdf["tail_shadow"].clip(0, 1)).astype(float)
    tdf["tail_shadow_trade"] = tdf["tail_shadow"]

    with st.sidebar:
        st.markdown("**Unified Execution**")
        strategy_sel: list[str] | None = None
        if "strategy" in tdf.columns:
            str_opts = sorted(tdf["strategy"].dropna().astype(str).unique().tolist())
            if str_opts:
                strategy_sel = st.multiselect(
                    "Strategy",
                    options=str_opts,
                    default=str_opts,
                    key="unified_strategy_filter",
                )
            else:
                st.caption("strategy: нет значений в данных")
                strategy_sel = None
        tail_only = st.checkbox("Только tail-hit сделки", value=False, key="unified_tail_only")
        dscore_floor: float | None = None
        if "d_score" in tdf.columns and tdf["d_score"].notna().any():
            ds = tdf["d_score"].dropna()
            d_lo = float(ds.min())
            d_hi = float(ds.max())
            if abs(d_hi - d_lo) < 1e-12:
                dscore_floor = d_lo
                st.caption(f"Мин. d_score: {d_lo:g} (одно значение в данных)")
            else:
                dscore_floor = st.slider(
                    "Минимум d_score",
                    min_value=d_lo,
                    max_value=d_hi,
                    value=d_lo,
                    key="unified_dscore_min",
                )

    fdf = tdf.copy()
    if strategy_sel is not None:
        if not strategy_sel:
            fdf = fdf.iloc[0:0]
        else:
            fdf = fdf[fdf["strategy"].astype(str).isin(strategy_sel)]
    if tail_only and "tail_hit" in fdf.columns:
        th = pd.to_numeric(fdf["tail_hit"], errors="coerce").fillna(0.0)
        fdf = fdf[th >= 1.0]
    if dscore_floor is not None and "d_score" in fdf.columns:
        fdf = fdf[pd.to_numeric(fdf["d_score"], errors="coerce") >= dscore_floor]

    fdf = fdf.sort_values("timestamp")
    if "tail_hit" in fdf.columns:
        fdf["tail_hit_num"] = pd.to_numeric(fdf["tail_hit"], errors="coerce").fillna(0.0).clip(0, 1)
        fdf["tail_shadow"] = _ema_tail_prob_series(fdf["tail_hit_num"].tolist(), alpha=0.3)
    else:
        fdf["tail_shadow"] = 0.0
    fdf["tail_alpha"] = (0.2 + 0.6 * pd.Series(fdf["tail_shadow"]).clip(0, 1)).astype(float)

    def _exec_tooltip(df: pd.DataFrame, cols: list[str]) -> list:
        tips: list = []
        if "timestamp" in df.columns:
            tips.append(alt.Tooltip("timestamp:T", title="timestamp"))
        for name in cols:
            if name not in df.columns:
                continue
            if name == "tail_hit":
                tips.append(alt.Tooltip("tail_hit:N", title="tail_hit"))
            elif name == "strategy":
                tips.append(alt.Tooltip("strategy:N", title="strategy"))
            else:
                tips.append(alt.Tooltip(f"{name}:Q", format=".4f", title=name))
        return tips

    full_metrics = [
        "pnl",
        "kelly_fraction",
        "size",
        "edge",
        "edge_score",
        "d_score",
        "variance",
        "tail_shadow",
        "tail_alpha",
        "tail_hit",
        "strategy",
    ]

    if fdf.empty:
        st.caption("Нет сделок после применения фильтров (или пустой выбор strategy).")

    uk = fdf.dropna(subset=["pnl", "kelly_fraction"])
    if not uk.empty:
        st.markdown("**PnL vs Kelly fraction**")
        c1 = (
            alt.Chart(uk)
            .mark_circle(size=55, stroke="#0d47a1", strokeWidth=1)
            .encode(
                x=alt.X("kelly_fraction:Q", title="kelly_fraction"),
                y=alt.Y("pnl:Q", title="pnl"),
                color=alt.value("#1565c0"),
                tooltip=_exec_tooltip(uk, full_metrics),
            )
            .interactive()
        )
        st.altair_chart(c1, use_container_width=True)
    else:
        st.caption("PnL vs Kelly: нет пар pnl/kelly_fraction.")

    us = fdf.dropna(subset=["pnl", "size"])
    if not us.empty:
        st.markdown("**Size vs PnL с Tail-shadow**")
        sh = (
            alt.Chart(us)
            .mark_circle(size=130, strokeWidth=0)
            .encode(
                x=alt.X("size:Q", title="size"),
                y=alt.Y("pnl:Q", title="pnl"),
                color=alt.Color(
                    "tail_shadow:Q",
                    scale=alt.Scale(scheme="reds", domain=[0, 1]),
                    legend=alt.Legend(title="Tail risk prob"),
                ),
                opacity=alt.Opacity("tail_alpha:Q", legend=alt.Legend(title="Shadow alpha")),
                tooltip=[
                    alt.Tooltip("tail_shadow:Q", format=".3f"),
                    alt.Tooltip("tail_alpha:Q", format=".3f"),
                ],
            )
        )
        bs = (
            alt.Chart(us)
            .mark_circle(size=48, stroke="#37474f", strokeWidth=1)
            .encode(
                x=alt.X("size:Q", title="size"),
                y=alt.Y("pnl:Q", title="pnl"),
                color=alt.value("#1565c0"),
                tooltip=_exec_tooltip(us, full_metrics),
            )
        )
        st.altair_chart((sh + bs).interactive(), use_container_width=True)
    else:
        st.caption("Size vs PnL: нет пар pnl/size.")

    uv = fdf.dropna(subset=["variance", "size"])
    if not uv.empty:
        st.markdown("**Variance vs Size**")
        c3 = (
            alt.Chart(uv)
            .mark_circle(size=55, stroke="#1b5e20", strokeWidth=1)
            .encode(
                x=alt.X("variance:Q", title="variance"),
                y=alt.Y("size:Q", title="size"),
                color=alt.value("#2e7d32"),
                tooltip=_exec_tooltip(uv, full_metrics),
            )
            .interactive()
        )
        st.altair_chart(c3, use_container_width=True)
    else:
        st.caption("Variance vs Size: нет пар variance/size.")

    st.divider()
    st.subheader("Execution Intelligence — correlations")
    st.caption(
        "Корреляции Пирсона по отфильтрованным сделкам (те же фильтры, что у Unified Execution). "
        "Матрица и scatter обновляются при смене strategy / tail-hit / d_score."
    )
    _corr_order = ["pnl", "kelly_fraction", "size", "variance", "edge_score"]
    corr_metrics = [m for m in _corr_order if m in fdf.columns]
    if "edge_score" not in fdf.columns and "edge" in fdf.columns:
        corr_metrics.append("edge")

    if fdf.empty or len(corr_metrics) < 2:
        st.caption("Недостаточно данных для корреляций (нужна непустая выборка и ≥2 метрик из списка).")
    else:
        corr_matrix = fdf[corr_metrics].corr(numeric_only=True)
        st.markdown("**Correlation matrix**")
        st.dataframe(
            corr_matrix.style.format("{:.3f}").background_gradient(
                cmap="coolwarm", axis=None, vmin=-1.0, vmax=1.0
            ),
            use_container_width=True,
        )

        _corr_tooltip: list = []
        for col in ("timestamp", "pnl", "size", "kelly_fraction", "variance", "edge_score", "edge", "d_score"):
            if col not in fdf.columns:
                continue
            if col == "timestamp":
                _corr_tooltip.append("timestamp:T")
            else:
                _corr_tooltip.append(f"{col}:Q")

        for i, x_metric in enumerate(corr_metrics):
            for y_metric in corr_metrics[i + 1 :]:
                pair_df = fdf.dropna(subset=[x_metric, y_metric])
                if pair_df.empty:
                    continue
                try:
                    corr_value = float(corr_matrix.loc[x_metric, y_metric])
                except (KeyError, TypeError, ValueError):
                    corr_value = float("nan")
                c_corr = (
                    alt.Chart(pair_df)
                    .mark_circle(size=52, stroke="#455a64", strokeWidth=1)
                    .encode(
                        x=alt.X(f"{x_metric}:Q", title=x_metric),
                        y=alt.Y(f"{y_metric}:Q", title=y_metric),
                        color=alt.value("#1565c0"),
                        tooltip=_corr_tooltip or [f"{x_metric}:Q", f"{y_metric}:Q"],
                    )
                    .interactive()
                )
                label = (
                    f"**{y_metric} vs {x_metric} | Corr=n/a**"
                    if math.isnan(corr_value)
                    else f"**{y_metric} vs {x_metric} | Corr={corr_value:.2f}**"
                )
                st.markdown(label)
                st.altair_chart(c_corr, use_container_width=True)

    st.divider()
    st.subheader("Tail Detection System — risk heatmap")
    st.caption(
        "PnL vs Size (те же фильтры, что у Unified Execution). tail_intensity = tail_hit × dd_norm. "
        "Аномалии: tail_intensity > 0.5 и pnl < 0 — красные точки. "
        "dd_norm — clip(dd/0.4, 0..1) по времени сделки из equity-серии; без снапшотов — current_dd из Risk block."
    )
    heat_df = _tail_intensity_frame(fdf, metrics_payload=m, risk_payload=risk)
    if not heat_df.empty:
        heat_df["is_anomaly"] = (
            (heat_df["tail_intensity"] > 0.5) & (heat_df["pnl"].astype(float) < 0.0)
        ).astype(int)
    heat_plot = heat_df.dropna(subset=["pnl", "size"])
    if heat_plot.empty:
        st.caption("Нет сделок с pnl/size после фильтров, или heatmap недоступен.")
    else:
        heat_tip: list = [
            "timestamp:T",
            alt.Tooltip("pnl:Q", format=".4f"),
            alt.Tooltip("size:Q", format=".4f"),
            alt.Tooltip("tail_intensity:Q", format=".4f", title="tail_intensity"),
            alt.Tooltip("dd_norm:Q", format=".4f", title="dd_norm"),
            alt.Tooltip("tail_hit:Q", format=".3f"),
            alt.Tooltip("is_anomaly:Q", format=".0f", title="anomaly"),
        ]
        if "kelly_fraction" in heat_plot.columns:
            heat_tip.append(alt.Tooltip("kelly_fraction:Q", format=".4f"))
        if "variance" in heat_plot.columns:
            heat_tip.append(alt.Tooltip("variance:Q", format=".4f"))
        if "edge_score" in heat_plot.columns:
            heat_tip.append(alt.Tooltip("edge_score:Q", format=".4f"))
        elif "edge" in heat_plot.columns:
            heat_tip.append(alt.Tooltip("edge:Q", format=".4f"))
        if "drawdown" in heat_plot.columns:
            heat_tip.append(alt.Tooltip("drawdown:Q", format=".4f", title="dd @ trade"))
        heatmap = (
            alt.Chart(heat_plot)
            .mark_circle(size=70, stroke="#37474f", strokeWidth=1)
            .encode(
                x=alt.X("size:Q", title="size"),
                y=alt.Y("pnl:Q", title="pnl"),
                color=alt.condition(
                    "datum.is_anomaly == 1",
                    alt.value("#d32f2f"),
                    alt.Color(
                        "tail_intensity:Q",
                        scale=alt.Scale(
                            scheme="redyellowgreen",
                            domain=[0.0, 1.0],
                            reverse=True,
                        ),
                        legend=alt.Legend(title="Tail intensity"),
                    ),
                ),
                tooltip=heat_tip,
            )
            .interactive()
        )
        st.altair_chart(heatmap, use_container_width=True)

        anomalies = heat_df[heat_df["is_anomaly"] == 1]
        if not anomalies.empty:
            st.markdown(
                f"**⚠️ Аномалии (tail_intensity > 0.5 и PnL < 0): {len(anomalies)} сделок**"
            )
            _anom_cols = [
                c
                for c in (
                    "timestamp",
                    "pnl",
                    "size",
                    "tail_intensity",
                    "dd_norm",
                    "kelly_fraction",
                )
                if c in anomalies.columns
            ]
            st.dataframe(anomalies[_anom_cols], use_container_width=True)

    st.divider()
    st.subheader("Tail Detection System — predictive shadow")
    st.caption(
        "Скользящая оценка вероятности tail-event: EMA по tail_hit на сделках и по частоте хитов в 5‑мин бинах "
        "(совпадает с бином heatmap при выборе Time bin = 5min). Текущий DD из Risk block — контекст."
    )
    tdf_ts = tdf.copy()
    tdf_ts["timestamp"] = pd.to_datetime(tdf_ts["timestamp"], errors="coerce", utc=True)
    tdf_ts = tdf_ts.sort_values("timestamp")
    if "tail_shadow_trade" not in tdf_ts.columns:
        if "tail_hit" in tdf_ts.columns:
            tdf_ts["tail_hit_num"] = pd.to_numeric(tdf_ts["tail_hit"], errors="coerce").fillna(0.0).clip(0, 1)
            tdf_ts["tail_shadow_trade"] = _ema_tail_prob_series(tdf_ts["tail_hit_num"].tolist(), alpha=0.3)
        else:
            tdf_ts["tail_shadow_trade"] = 0.0
    elif "tail_hit_num" not in tdf_ts.columns and "tail_hit" in tdf_ts.columns:
        tdf_ts["tail_hit_num"] = pd.to_numeric(tdf_ts["tail_hit"], errors="coerce").fillna(0.0).clip(0, 1)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Per-trade EMA tail probability**")
        if not tdf_ts.empty and "tail_shadow_trade" in tdf_ts.columns:
            tip: list = ["timestamp:T", alt.Tooltip("tail_shadow_trade:Q", format=".3f")]
            if "strategy" in tdf_ts.columns:
                tip.append("strategy:N")
            if "pnl" in tdf_ts.columns:
                tip.append("pnl:Q")
            line = (
                alt.Chart(tdf_ts)
                .mark_line(color="#c62828")
                .encode(
                    x=alt.X("timestamp:T", title="Time"),
                    y=alt.Y("tail_shadow_trade:Q", title="EMA tail prob", scale=alt.Scale(domain=[0, 1])),
                    tooltip=tip,
                )
                .properties(height=220)
            )
            st.altair_chart(line, use_container_width=True)
        else:
            st.caption("Нет данных для EMA по сделкам.")

    with c2:
        st.markdown("**Binned tail-hit rate → EMA (shadow по окнам)**")
        if "tail_hit_num" in tdf_ts.columns and not tdf_ts.empty:
            bin_choice = st.selectbox("Shadow time bin", ["5min", "15min"], index=0, key="shadow_bin_sel")
            tdf_ts["shadow_bin"] = tdf_ts["timestamp"].dt.floor(bin_choice)
            br = (
                tdf_ts.groupby("shadow_bin", as_index=False)["tail_hit_num"]
                .mean()
                .sort_values("shadow_bin")
            )
            prev = 0.0
            emas: list[float] = []
            for rate in br["tail_hit_num"].tolist():
                prev = 0.7 * prev + 0.3 * float(rate)
                emas.append(prev)
            br["ema_bin"] = emas
            area = (
                alt.Chart(br)
                .mark_area(opacity=0.35, color="#b71c1c")
                .encode(
                    x=alt.X("shadow_bin:T", title="Time bin"),
                    y=alt.Y("ema_bin:Q", title="EMA(bin rate)", scale=alt.Scale(domain=[0, 1])),
                    tooltip=[alt.Tooltip("ema_bin:Q", format=".3f")],
                )
                .properties(height=220)
            )
            st.altair_chart(area, use_container_width=True)
        else:
            st.caption("Нет tail_hit в trades_series.")

    st.markdown("**Interactive Tail-Risk Shadow (time × edge_score)**")
    df_plot = tdf_ts.dropna(subset=["timestamp"]).copy()
    if "edge_score" not in df_plot.columns and "edge" in df_plot.columns:
        df_plot["edge_score"] = df_plot["edge"]
    if "edge_score" in df_plot.columns:
        df_plot["edge_score"] = pd.to_numeric(df_plot["edge_score"], errors="coerce")
    if "tail_shadow" not in df_plot.columns and "tail_shadow_trade" in df_plot.columns:
        df_plot["tail_shadow"] = df_plot["tail_shadow_trade"]
    elif "tail_shadow" not in df_plot.columns:
        df_plot["tail_shadow"] = 0.0
    df_plot["tail_alpha"] = (0.2 + 0.6 * df_plot["tail_shadow"].clip(0, 1)).astype(float)
    df_plot = df_plot.dropna(subset=["edge_score"])
    if not df_plot.empty:
        tip_base: list = [
            "timestamp:T",
            alt.Tooltip("edge_score:Q", format=".4f"),
            alt.Tooltip("tail_shadow:Q", format=".3f", title="tail_shadow"),
        ]
        if "pnl" in df_plot.columns:
            tip_base.append("pnl:Q")
        if "size" in df_plot.columns:
            tip_base.append("size:Q")
        if "d_score" in df_plot.columns:
            tip_base.append("d_score:Q")
        if "tail_hit" in df_plot.columns:
            tip_base.append("tail_hit:N")
        shadow_layer = (
            alt.Chart(df_plot)
            .mark_circle(size=180, strokeWidth=0)
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y("edge_score:Q", title="edge_score"),
                color=alt.Color(
                    "tail_shadow:Q",
                    scale=alt.Scale(scheme="reds", domain=[0, 1]),
                    legend=alt.Legend(title="Tail risk prob"),
                ),
                opacity=alt.Opacity(
                    "tail_alpha:Q",
                    legend=alt.Legend(title="Shadow alpha"),
                ),
                tooltip=[
                    alt.Tooltip("tail_shadow:Q", format=".3f", title="tail_shadow"),
                    alt.Tooltip("tail_alpha:Q", format=".3f", title="tail_alpha"),
                ],
            )
        )
        base_layer = (
            alt.Chart(df_plot)
            .mark_circle(size=55, stroke="#0d47a1", strokeWidth=1)
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y("edge_score:Q", title="edge_score"),
                color=alt.value("#1565c0"),
                tooltip=tip_base,
            )
        )
        final_interactive = (shadow_layer + base_layer).interactive()
        st.altair_chart(final_interactive, use_container_width=True)
        st.caption(
            "Подложка: красный градиент + прозрачность (tail_alpha = 0.2 + 0.6×tail_shadow); сверху — сделки."
        )
    else:
        st.caption("Нет edge_score/edge для интерактивного shadow.")

    st.markdown("**Size vs PnL с Tail-Risk подсветкой (tail_shadow)**")
    df_sp = tdf_ts.copy()
    df_sp["size"] = pd.to_numeric(df_sp["size"], errors="coerce")
    df_sp["pnl"] = pd.to_numeric(df_sp["pnl"], errors="coerce")
    if "tail_shadow_trade" in df_sp.columns:
        df_sp["tail_shadow"] = df_sp["tail_shadow_trade"].clip(0, 1)
    elif "tail_shadow" not in df_sp.columns:
        df_sp["tail_shadow"] = 0.0
    else:
        df_sp["tail_shadow"] = df_sp["tail_shadow"].clip(0, 1)
    df_sp["tail_alpha"] = (0.2 + 0.6 * df_sp["tail_shadow"]).astype(float)
    df_sp = df_sp.dropna(subset=["size", "pnl"])
    if not df_sp.empty:
        tip_sp: list = [
            "timestamp:T",
            alt.Tooltip("size:Q", format=".4f"),
            alt.Tooltip("pnl:Q", format=".4f"),
            alt.Tooltip("tail_shadow:Q", format=".3f"),
        ]
        if "edge_score" in df_sp.columns:
            tip_sp.append("edge_score:Q")
        if "d_score" in df_sp.columns:
            tip_sp.append("d_score:Q")
        if "tail_hit" in df_sp.columns:
            tip_sp.append("tail_hit:N")
        shadow_sp = (
            alt.Chart(df_sp)
            .mark_circle(size=140, strokeWidth=0)
            .encode(
                x=alt.X("size:Q", title="size"),
                y=alt.Y("pnl:Q", title="pnl"),
                color=alt.Color(
                    "tail_shadow:Q",
                    scale=alt.Scale(scheme="reds", domain=[0, 1]),
                    legend=alt.Legend(title="Tail risk prob"),
                ),
                opacity=alt.Opacity("tail_alpha:Q", legend=alt.Legend(title="Shadow alpha")),
                tooltip=[
                    alt.Tooltip("tail_shadow:Q", format=".3f"),
                    alt.Tooltip("tail_alpha:Q", format=".3f"),
                ],
            )
        )
        base_sp = (
            alt.Chart(df_sp)
            .mark_circle(size=48, stroke="#37474f", strokeWidth=1)
            .encode(
                x=alt.X("size:Q", title="size"),
                y=alt.Y("pnl:Q", title="pnl"),
                color=alt.value("#1565c0"),
                tooltip=tip_sp,
            )
        )
        st.altair_chart((shadow_sp + base_sp).interactive(), use_container_width=True)
        st.caption(
            "Красная подложка: выше tail_shadow — сильнее красный; tail_alpha задаёт прозрачность. Сверху — те же точки (сделки)."
        )
    else:
        st.caption("Нет size/pnl для Size vs PnL shadow.")

    dd_now = float((risk or {}).get("current_dd", 0.0))
    st.metric("current_dd (context)", f"{dd_now:.4f}")
else:
    st.caption("Нет trades_series в ответе /metrics (или нет трейдов). Перезапусти API и дай системе набрать трейды.")

st.divider()
st.header("⚠️ Trade Alerts Scatter View")
if trades_series:
    adf = pd.DataFrame(trades_series)
    thresholds = m.get("alert_thresholds") or {}

    for c in ("pnl", "size", "kelly_fraction", "variance", "edge", "edge_score"):
        if c in adf.columns:
            adf[c] = pd.to_numeric(adf[c], errors="coerce")

    # Backward compatible: if API has no explicit alert flags, derive on dashboard side.
    max_dd = float(thresholds.get("max_dd", 0.40))
    min_size = float(thresholds.get("min_size", 0.02))
    max_size = float(thresholds.get("max_size", 0.25))
    max_kelly = float(thresholds.get("max_kelly", 10.0))
    eq_drop_thr = float(thresholds.get("equity_drop_1h", 0.05))
    current_dd = float((risk or {}).get("current_dd", 0.0))

    if "alert_dd" not in adf.columns:
        adf["alert_dd"] = current_dd > max_dd
    if "alert_size" not in adf.columns:
        adf["alert_size"] = adf["size"].apply(
            lambda x: bool(pd.notna(x) and (float(x) < min_size or float(x) > max_size))
        )
    if "alert_kelly" not in adf.columns:
        adf["alert_kelly"] = adf["kelly_fraction"].apply(
            lambda x: bool(pd.notna(x) and abs(float(x)) > max_kelly)
        )
    if "alert_equity_drop" not in adf.columns:
        # We don't have per-trade equity drop in legacy payload -> use global metrics/snapshot proxy.
        drawdowns = (m.get("series") or {}).get("drawdown") or []
        latest_drop = float(drawdowns[-1]) if drawdowns else 0.0
        adf["alert_equity_drop"] = latest_drop > eq_drop_thr

    adf["any_alert"] = adf[["alert_dd", "alert_size", "alert_kelly", "alert_equity_drop"]].any(axis=1)

    st.sidebar.subheader("Filter Alerts")
    show_dd = st.sidebar.checkbox("DD alert", True)
    show_size = st.sidebar.checkbox("Size alert", True)
    show_kelly = st.sidebar.checkbox("Kelly alert", True)
    show_equity = st.sidebar.checkbox("Equity drop alert", True)

    alert_mask = (
        ((adf["alert_dd"]) & show_dd)
        | ((adf["alert_size"]) & show_size)
        | ((adf["alert_kelly"]) & show_kelly)
        | ((adf["alert_equity_drop"]) & show_equity)
    )
    adf["alert_visible"] = alert_mask
    adf["alert_class"] = adf["alert_visible"].map({True: "Alert", False: "Normal"})

    st.subheader("Risk Engine — alert summary")
    alert_types = ["alert_dd", "alert_size", "alert_kelly", "alert_equity_drop"]
    summary_data = []
    total_trades = len(adf)
    for atype in alert_types:
        count = int(adf[atype].sum()) if atype in adf.columns else 0
        pct = 100.0 * count / total_trades if total_trades > 0 else 0.0
        summary_data.append({"Alert Type": atype, "Count": count, "Percent (%)": round(pct, 1)})
    st.table(pd.DataFrame(summary_data))

    st.subheader("Risk Engine — live alerts")

    def _cell_color(val, alert_type: str) -> str:
        is_alert = bool(val)
        if not is_alert:
            return ""
        if alert_type in {"alert_dd", "alert_kelly"}:
            return "background-color: #FF6961"  # high severity
        if alert_type == "alert_size":
            return "background-color: #FFD966"  # medium severity
        if alert_type == "alert_equity_drop":
            return "background-color: #FFFF99"  # warning
        return ""

    alerts_only_df = adf[
        adf["alert_dd"] | adf["alert_size"] | adf["alert_kelly"] | adf["alert_equity_drop"]
    ].copy()

    display_cols = [
        "timestamp",
        "strategy",
        "pnl",
        "size",
        "edge",
        "edge_score",
        "variance",
        "kelly_fraction",
    ] + alert_types
    display_cols = [c for c in display_cols if c in alerts_only_df.columns]

    if not alerts_only_df.empty and display_cols:
        alerts_display_df = alerts_only_df[display_cols].copy()
        min_sev = st.slider("Minimum Severity", min_value=1, max_value=3, value=1, step=1)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Recent Alerts (last 10)**")
            df_filtered = alerts_only_df.copy()
            mask = pd.Series(False, index=df_filtered.index)
            for col in alert_types:
                if col not in df_filtered.columns:
                    continue
                sev = int(
                    {
                        "alert_dd": 3,
                        "alert_size": 2,
                        "alert_kelly": 3,
                        "alert_equity_drop": 2,
                    }.get(col, 1)
                )
                mask |= (df_filtered[col].astype(bool) & (sev >= min_sev))
            df_filtered = df_filtered[mask]

            styler = alerts_display_df.style
            for col in alert_types:
                if col in alerts_display_df.columns:
                    styler = styler.map(lambda v, c=col: _cell_color(v, c), subset=[col])
            if not df_filtered.empty:
                styler_filtered = df_filtered[display_cols].style
                for col in alert_types:
                    if col in df_filtered.columns:
                        styler_filtered = styler_filtered.map(
                            lambda v, c=col: _cell_color(v, c), subset=[col]
                        )
                st.dataframe(styler_filtered.tail(10), use_container_width=True)
            else:
                st.caption("Нет алертов для выбранного minimum severity.")

        with col2:
            st.markdown("**Alert Type Distribution**")
            severity_map = {
                "alert_dd": 3,
                "alert_size": 2,
                "alert_kelly": 3,
                "alert_equity_drop": 2,
            }
            alert_counts = []
            for col in alert_types:
                if col not in alerts_only_df.columns:
                    continue
                sev = int(severity_map.get(col, 1))
                alert_counts.append(
                    {
                        "Alert Type": col,
                        "Count": int(
                            (
                                alerts_only_df[col].astype(bool)
                                & (sev >= min_sev)
                            ).sum()
                        ),
                        "Severity": sev,
                    }
                )
            alert_dist_df = pd.DataFrame(alert_counts)
            if not alert_dist_df.empty:
                chart = (
                    alt.Chart(alert_dist_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Alert Type:N", sort=alert_types),
                        y=alt.Y("Count:Q"),
                        color=alt.Color(
                            "Severity:Q",
                            scale=alt.Scale(
                                domain=[1, 2, 3], range=["#8BC34A", "#FFD966", "#FF6961"]
                            ),
                            legend=alt.Legend(title="Severity"),
                        ),
                        tooltip=["Alert Type", "Count", "Severity"],
                    )
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("Нет данных для распределения alert-типов.")
    else:
        st.caption("Нет активных alert-трейдов для таблицы severity.")

    st.subheader("Tail Detection System — live heatmap")
    hcol1, hcol2, hcol3 = st.columns(3)
    with hcol1:
        heatmap_strategy = st.selectbox(
            "Strategy filter",
            options=["all", "A", "B"],
            index=0,
            help="Только сделки выбранной стратегии; DD из снапшотов — при «all».",
        )
    with hcol2:
        heatmap_window_h = st.select_slider(
            "Rolling window (hours)",
            options=[2, 3],
            value=3,
            help="Только события за последние N часов (UTC).",
        )
    with hcol3:
        heatmap_time_bin = st.selectbox(
            "Time bin",
            options=["1min", "5min", "15min"],
            index=1,
            help="Бин по времени для heatmap.",
        )
    auto_refresh_heatmap = st.toggle("Auto-refresh heatmap", value=True)
    refresh_interval_sec = st.slider(
        "Refresh interval (sec)", min_value=10, max_value=120, value=30, step=5
    )

    def _render_alerts_heatmap() -> None:
        df_alerts = _load_alert_events_from_db(
            limit=2000,
            window_hours=float(heatmap_window_h),
            strategy_filter=heatmap_strategy,
        )
        markers_raw = _load_strong_tail_markers_from_db(
            limit=2000,
            window_hours=float(heatmap_window_h),
            strategy_filter=heatmap_strategy,
            d_score_min=2.0,
        )
        if df_alerts.empty:
            st.info("No alerts in the selected window / strategy")
            return

        df_alerts["time_bin"] = pd.to_datetime(
            df_alerts["timestamp"], errors="coerce", utc=True
        ).dt.floor(heatmap_time_bin)
        df_alerts = df_alerts.dropna(subset=["time_bin"])
        if df_alerts.empty:
            st.info("No alerts yet for heatmap")
            return

        heatmap_df = (
            df_alerts.groupby(["time_bin", "alert_type"])
            .agg(
                alert_count=("severity", "count"),
                avg_severity=("severity", "mean"),
                max_d_score=("d_score", "max"),
            )
            .reset_index()
        )
        shadow_bins = _load_tail_shadow_bins_from_db(
            window_hours=float(heatmap_window_h),
            strategy_filter=heatmap_strategy,
            time_bin=heatmap_time_bin,
            ema_alpha=0.3,
        )
        if not shadow_bins.empty:
            shadow_rows = []
            for _, r in shadow_bins.iterrows():
                shadow_rows.append(
                    {
                        "time_bin": r["time_bin"],
                        "alert_type": "TailShadow",
                        "alert_count": 0,
                        "avg_severity": float(r["ema_tail_prob"]) * 100.0,
                        "max_d_score": float("nan"),
                    }
                )
            heatmap_df = pd.concat([heatmap_df, pd.DataFrame(shadow_rows)], ignore_index=True)

        heatmap = (
            alt.Chart(heatmap_df)
            .mark_rect(opacity=0.85)
            .encode(
                x=alt.X("time_bin:T", title="Time"),
                y=alt.Y("alert_type:N", title="Alert Type"),
                color=alt.Color(
                    "avg_severity:Q",
                    scale=alt.Scale(domain=[0, 100], scheme="reds"),
                    title="Intensity + TailShadow (0–100)",
                ),
                tooltip=[
                    alt.Tooltip("time_bin:T", title="time_bin"),
                    alt.Tooltip("alert_type:N", title="alert_type"),
                    alt.Tooltip("alert_count:Q", title="alert_count"),
                    alt.Tooltip("avg_severity:Q", title="avg_severity", format=".2f"),
                    alt.Tooltip("max_d_score:Q", title="max_d_score", format=".3f"),
                ],
            )
            .properties(height=350)
        )

        if not markers_raw.empty:
            md = markers_raw.copy()
            md["time_bin"] = md["timestamp"].dt.floor(heatmap_time_bin)
            md["alert_type"] = "TailHit"
            markers = (
                alt.Chart(md)
                .mark_circle(size=90, color="blue", opacity=0.65)
                .encode(
                    x=alt.X("time_bin:T", title="Time"),
                    y=alt.Y("alert_type:N", title="Alert Type"),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="timestamp"),
                        alt.Tooltip("strategy:N", title="strategy"),
                        alt.Tooltip("d_score:Q", title="d_score", format=".3f"),
                        alt.Tooltip("alpha_multiplier:Q", title="alpha_multiplier", format=".3f"),
                    ],
                )
            )
            chart = heatmap + markers
        else:
            chart = heatmap

        st.altair_chart(chart, use_container_width=True)

    if hasattr(st, "fragment"):
        run_every = f"{refresh_interval_sec}s" if auto_refresh_heatmap else None

        @st.fragment(run_every=run_every)
        def _heatmap_fragment() -> None:
            _render_alerts_heatmap()

        _heatmap_fragment()
    else:
        _render_alerts_heatmap()

    fig1 = px.scatter(
        adf.dropna(subset=["size", "pnl"]),
        x="size",
        y="pnl",
        color="alert_class",
        color_discrete_map={"Normal": "blue", "Alert": "red"},
        hover_data=[
            "edge",
            "edge_score",
            "kelly_fraction",
            "variance",
            "pnl",
            "size",
            "alert_dd",
            "alert_size",
            "alert_kelly",
            "alert_equity_drop",
        ],
        title="PnL vs Size (Alerts Highlighted)",
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        adf.dropna(subset=["kelly_fraction", "pnl"]),
        x="kelly_fraction",
        y="pnl",
        color="alert_class",
        color_discrete_map={"Normal": "blue", "Alert": "red"},
        hover_data=[
            "edge",
            "edge_score",
            "size",
            "variance",
            "pnl",
            "kelly_fraction",
            "alert_dd",
            "alert_size",
            "alert_kelly",
            "alert_equity_drop",
        ],
        title="PnL vs Kelly Fraction (Alerts Highlighted)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        adf.dropna(subset=["variance", "size"]),
        x="variance",
        y="size",
        color="alert_class",
        color_discrete_map={"Normal": "blue", "Alert": "red"},
        hover_data=[
            "edge",
            "edge_score",
            "kelly_fraction",
            "pnl",
            "variance",
            "alert_dd",
            "alert_size",
            "alert_kelly",
            "alert_equity_drop",
        ],
        title="Variance vs Size (Alerts Highlighted)",
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.caption("Нет trades_series для Alert Scatter View.")


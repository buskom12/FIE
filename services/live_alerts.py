from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


def check_live_alerts(
    trade: Dict,
    snapshot: Dict,
    *,
    max_dd_thresh: float = 0.40,
    min_size: float = 0.02,
    max_size: float = 0.25,
    max_kelly: float = 10.0,
    equity_drop_thresh: float = 0.05,
) -> List[str]:
    """
    Validate trade/snapshot against runtime thresholds.
    Returns alert messages.
    """
    alerts: List[str] = []

    drawdown = float(snapshot.get("drawdown", 0.0) or 0.0)
    if drawdown > max_dd_thresh:
        alerts.append(f"High DD: {drawdown:.3f} > {max_dd_thresh:.3f}")

    if "size" in trade:
        size = float(trade.get("size", 0.0) or 0.0)
        if size < min_size or size > max_size:
            alerts.append(f"Out-of-bounds size: {size:.3f} not in [{min_size:.3f}, {max_size:.3f}]")

    if "kelly_fraction" in trade:
        kf = float(trade.get("kelly_fraction", 0.0) or 0.0)
        if abs(kf) > max_kelly:
            alerts.append(f"High Kelly fraction: {kf:.3f} (abs > {max_kelly:.3f})")

    equity_drop_1h = float(snapshot.get("equity_drop_1h", 0.0) or 0.0)
    if equity_drop_1h > equity_drop_thresh:
        alerts.append(
            f"Equity drop > {equity_drop_thresh*100:.0f}% in 1h: {equity_drop_1h:.3f}"
        )

    return alerts


def log_alerts(
    alerts: List[str],
    *,
    send_to_slack: Optional[Callable[[str], None]] = None,
    send_to_telegram: Optional[Callable[[str], None]] = None,
    last_alert_at: Optional[Dict[str, datetime]] = None,
    cooldown_sec: int = 300,
) -> None:
    """
    Log alerts and optionally dispatch to Slack/Telegram.
    Applies per-message cooldown when last_alert_at is provided.
    """
    now = datetime.now(timezone.utc)
    state = last_alert_at if last_alert_at is not None else {}

    for a in alerts:
        prev = state.get(a)
        if prev is not None and (now - prev).total_seconds() < cooldown_sec:
            continue
        state[a] = now

        line = f"[ALERT] {now.isoformat()} {a}"
        print(line, flush=True)
        if send_to_slack:
            send_to_slack(a)
        if send_to_telegram:
            send_to_telegram(a)


def slack_webhook_sender(webhook_url: str) -> Callable[[str], None]:
    """Incoming Slack webhook: POST JSON {\"text\": message}."""

    import requests

    url = webhook_url.strip()

    def _send(message: str) -> None:
        requests.post(url, json={"text": message}, timeout=15)

    return _send


def _high_risk_row_mask(
    trades_df: pd.DataFrame,
    *,
    size_threshold: float,
    edge_score_threshold: float,
    tail_hit_threshold: float,
) -> pd.Series:
    """
    Булева маска «smart» high-risk: tail_hit, pnl<0, size, edge_score (или edge).
    """
    need = ("pnl", "tail_hit", "size")
    if any(c not in trades_df.columns for c in need):
        return pd.Series(False, index=trades_df.index)

    df = trades_df.copy()
    if "edge_score" not in df.columns:
        if "edge" in df.columns:
            df["edge_score"] = pd.to_numeric(df["edge"], errors="coerce").fillna(0.0)
        else:
            df["edge_score"] = 0.0
    else:
        df["edge_score"] = pd.to_numeric(df["edge_score"], errors="coerce").fillna(0.0)

    th = pd.to_numeric(df["tail_hit"], errors="coerce").fillna(0.0)
    pnl = pd.to_numeric(df["pnl"], errors="coerce")
    sz = pd.to_numeric(df["size"], errors="coerce").fillna(0.0)
    es = df["edge_score"]

    return (
        (th > float(tail_hit_threshold))
        & (pnl < 0.0)
        & (sz > float(size_threshold))
        & (es > float(edge_score_threshold))
    )


class DynamicTailEventMonitor:
    """
    Адаптивный tail event alert: серия из N подряд high-risk сделок (по времени),
    порог tail_hit зависит от текущего drawdown — при росте DD порог ниже (чувствительнее).
    """

    def __init__(
        self,
        window: int = 3,
        cooldown_sec: float = 300.0,
        base_threshold: float = 0.5,
        dd_floor: float = 0.3,
        dd_cap: float = 0.6,
        threshold_min: float = 0.2,
        threshold_max: float = 0.7,
    ) -> None:
        self.window = max(1, int(window))
        self.cooldown_sec = float(cooldown_sec)
        self.base_threshold = float(base_threshold)
        self.dd_floor = float(dd_floor)
        self.dd_cap = float(dd_cap)
        self.threshold_min = float(threshold_min)
        self.threshold_max = float(threshold_max)
        self.last_alert_at: Optional[float] = None
        self.tail_queue: deque[bool] = deque(maxlen=self.window)

    def _adaptive_threshold(self, current_dd: float) -> float:
        """
        DD <= dd_floor → порог threshold_max (реже срабатывает).
        DD >= dd_cap → порог threshold_min (ловим tail раньше).
        Между — линейная интерполяция.
        """
        span = self.dd_cap - self.dd_floor
        if span <= 1e-12:
            return self.threshold_min
        dd = float(current_dd or 0.0)
        dd_norm = (dd - self.dd_floor) / span
        dd_norm = min(max(dd_norm, 0.0), 1.0)
        return self.threshold_max - dd_norm * (self.threshold_max - self.threshold_min)

    def check_trades(
        self,
        trades_df: pd.DataFrame,
        snapshot: Dict[str, Any],
        *,
        slack_client: Optional[Callable[[str], None]] = None,
        telegram_client: Optional[Callable[[str], None]] = None,
        size_threshold: float = 0.05,
        edge_score_threshold: float = 0.02,
    ) -> None:
        now = time.time()
        if self.last_alert_at is not None and (now - self.last_alert_at) < self.cooldown_sec:
            return

        if trades_df is None or trades_df.empty or "timestamp" not in trades_df.columns:
            return

        dd = float(snapshot.get("drawdown", 0.0) or 0.0)
        adaptive_threshold = self._adaptive_threshold(dd)

        df = trades_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp").dropna(subset=["timestamp"])
        if len(df) < self.window:
            return

        last_block = df.tail(self.window)
        mask = _high_risk_row_mask(
            last_block,
            size_threshold=size_threshold,
            edge_score_threshold=edge_score_threshold,
            tail_hit_threshold=adaptive_threshold,
        )
        if len(mask) != self.window or not bool(mask.all()):
            return

        self.tail_queue.clear()
        for v in mask.tolist():
            self.tail_queue.append(bool(v))

        equity = float(snapshot.get("equity", 0.0) or 0.0)
        msg_lines = [
            f"⚠️ Dynamic Tail Event Alert: {self.window} consecutive high-risk trades",
            f"adaptive tail_hit threshold: {adaptive_threshold:.4f} (DD={dd:.4f}, "
            f"dd_floor={self.dd_floor}, dd_cap={self.dd_cap}, thr∈[{self.threshold_min},{self.threshold_max}])",
            f"filters: pnl<0, size>{size_threshold}, edge_score>{edge_score_threshold}",
            f"Equity: {equity:.4f}, DD: {dd:.4f}",
        ]
        for _, row in last_block.iterrows():
            ts = row.get("timestamp", "")
            strat = row.get("strategy", "")
            pnl_v = row.get("pnl", float("nan"))
            sz_v = row.get("size", float("nan"))
            es_v = row.get("edge_score", row.get("edge", float("nan")))
            th_v = row.get("tail_hit", float("nan"))
            msg_lines.append(
                f"{ts} | {strat} | PnL: {pnl_v} | Size: {sz_v} | "
                f"edge_score: {es_v} | tail_hit: {th_v}"
            )
        msg = "\n".join(msg_lines)
        if len(msg) > 3500:
            msg = msg[:3490] + "\n…(truncated)"

        ts_iso = datetime.now(timezone.utc).isoformat()
        print(f"[TAIL_EVENT_DYN] {ts_iso}\n{msg}", flush=True)

        if slack_client:
            slack_client(msg)
        if telegram_client:
            telegram_client(msg)

        self.last_alert_at = now
        self.tail_queue.clear()


# Обратная совместимость имён
TailEventMonitor = DynamicTailEventMonitor


def alert_tail_anomalies(
    trades_df: pd.DataFrame,
    snapshot: Dict[str, Any],
    *,
    slack_client: Optional[Callable[[str], None]] = None,
    telegram_client: Optional[Callable[[str], None]] = None,
    cooldown_sec: float = 300.0,
    last_alert_at: Optional[float] = None,
    tail_intensity_threshold: float = 0.5,
) -> Optional[float]:
    """
    Аномалии tail risk: tail_intensity = tail_hit × dd_norm, dd_norm = clip(drawdown/0.4, 0..1)
    по текущему snapshot (как live-контекст на дашборде без merge_asof по истории).

    Условие: tail_intensity > tail_intensity_threshold и pnl < 0.
    Возвращает time.time() при отправке нового алерта, иначе прежний last_alert_at.
    """
    now = time.time()
    if last_alert_at is not None and (now - last_alert_at) < float(cooldown_sec):
        return last_alert_at

    if trades_df is None or trades_df.empty:
        return last_alert_at

    if "pnl" not in trades_df.columns or "tail_hit" not in trades_df.columns:
        return last_alert_at

    dd = float(snapshot.get("drawdown", 0.0) or 0.0)
    dd_norm = min(1.0, max(0.0, dd / 0.4))

    th = pd.to_numeric(trades_df["tail_hit"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    tail_intensity = th * dd_norm
    pnl = pd.to_numeric(trades_df["pnl"], errors="coerce")
    mask = (tail_intensity > float(tail_intensity_threshold)) & (pnl < 0.0)
    anomalies = trades_df.loc[mask].copy()
    if anomalies.empty:
        return last_alert_at

    anomalies["_tail_intensity"] = tail_intensity.loc[mask].astype(float).values

    equity = float(snapshot.get("equity", 0.0) or 0.0)
    msg_lines = [
        f"⚠️ Tail risk alert: {len(anomalies)} anomaly trade(s) "
        f"(tail_intensity > {tail_intensity_threshold} & PnL < 0)",
        f"Equity: {equity:.4f}, DD: {dd:.4f}, dd_norm: {dd_norm:.4f}",
    ]
    for _, row in anomalies.iterrows():
        ts = row.get("timestamp", "")
        pnl_v = row.get("pnl", float("nan"))
        sz = row.get("size", float("nan"))
        strat = row.get("strategy", "")
        ti = float(row["_tail_intensity"])
        msg_lines.append(
            f"{ts} | {strat} | PnL: {pnl_v} | Size: {sz} | tail_intensity: {ti:.4f}"
        )
    msg = "\n".join(msg_lines)
    if len(msg) > 3500:
        msg = msg[:3490] + "\n…(truncated)"

    ts_iso = datetime.now(timezone.utc).isoformat()
    print(f"[TAIL_ANOMALY] {ts_iso}\n{msg}", flush=True)

    if slack_client:
        slack_client(msg)
    if telegram_client:
        telegram_client(msg)

    return now


def alert_tail_anomalies_smart(
    trades_df: pd.DataFrame,
    snapshot: Dict[str, Any],
    *,
    slack_client: Optional[Callable[[str], None]] = None,
    telegram_client: Optional[Callable[[str], None]] = None,
    cooldown_sec: float = 300.0,
    last_alert_at: Optional[float] = None,
    size_threshold: float = 0.05,
    edge_score_threshold: float = 0.02,
    tail_hit_threshold: float = 0.5,
) -> Optional[float]:
    """
    «Умные» tail-risk алерты: только заметные по размеру и edge, с tail_hit и убытком.

    Условия (все одновременно):
        tail_hit > tail_hit_threshold, pnl < 0, size > size_threshold, edge_score > edge_score_threshold.
    Если нет колонки edge_score — используется edge (если есть), иначе edge_score = 0.
    """
    now = time.time()
    if last_alert_at is not None and (now - last_alert_at) < float(cooldown_sec):
        return last_alert_at

    if trades_df is None or trades_df.empty:
        return last_alert_at

    need = ("pnl", "tail_hit", "size")
    if any(c not in trades_df.columns for c in need):
        return last_alert_at

    df = trades_df.copy()
    if "edge_score" not in df.columns:
        if "edge" in df.columns:
            df["edge_score"] = pd.to_numeric(df["edge"], errors="coerce").fillna(0.0)
        else:
            df["edge_score"] = 0.0
    else:
        df["edge_score"] = pd.to_numeric(df["edge_score"], errors="coerce").fillna(0.0)

    mask = _high_risk_row_mask(
        df,
        size_threshold=size_threshold,
        edge_score_threshold=edge_score_threshold,
        tail_hit_threshold=tail_hit_threshold,
    )
    anomalies = df.loc[mask].copy()
    if anomalies.empty:
        return last_alert_at

    equity = float(snapshot.get("equity", 0.0) or 0.0)
    dd = float(snapshot.get("drawdown", 0.0) or 0.0)
    msg_lines = [
        f"⚠️ Smart Tail Risk Alert: {len(anomalies)} high-risk trade(s)",
        f"filters: tail_hit>{tail_hit_threshold}, pnl<0, size>{size_threshold}, edge_score>{edge_score_threshold}",
        f"Equity: {equity:.4f}, DD: {dd:.4f}",
    ]
    for _, row in anomalies.iterrows():
        ts = row.get("timestamp", "")
        strat = row.get("strategy", "")
        pnl_v = row.get("pnl", float("nan"))
        sz_v = row.get("size", float("nan"))
        edge_v = float(row.get("edge_score", 0.0) or 0.0)
        th_v = float(row.get("tail_hit", 0.0) or 0.0)
        msg_lines.append(
            f"{ts} | {strat} | PnL: {pnl_v} | Size: {sz_v:.4f} | "
            f"edge_score: {edge_v:.4f} | tail_hit: {th_v:.4f}"
        )
    msg = "\n".join(msg_lines)
    if len(msg) > 3500:
        msg = msg[:3490] + "\n…(truncated)"

    ts_iso = datetime.now(timezone.utc).isoformat()
    print(f"[TAIL_ANOMALY_SMART] {ts_iso}\n{msg}", flush=True)

    if slack_client:
        slack_client(msg)
    if telegram_client:
        telegram_client(msg)

    return now


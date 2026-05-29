#!/usr/bin/env python3
"""
Gate / signal report for production logs and trades.

Usage:
  export FIE_TRADE_REPORT_MIN_ID=$(sqlite3 db/fie_prod.sqlite "SELECT MAX(id) FROM trades;")
  FIE_TRADE_REPORT_MIN_ID=$cut_id python3 scripts/report_gate_metrics.py /tmp/fie_loop.log

Описание:
  Формирует 5 блоков:
    1) PnL summary (до/после gate threshold)
    2) skips_by_reason
    3) active_min distribution на входах
    4) conversion (signals → entries)
    5) (l,f) вклад после gate (LF skip)
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "db" / "fie_prod.sqlite"
DEFAULT_LOG = Path("/tmp/fie_loop.log")


def _report_min_id() -> int | None:
    raw = os.environ.get("FIE_TRADE_REPORT_MIN_ID", "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
        return v if v > 0 else None
    except ValueError:
        return None


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_signals(lines: Iterable[str]) -> list[dict]:
    parsed = []
    pattern = re.compile(
        r"\[signals\] zone refreshed: .*?gate=(?P<gate>[-+]?[0-9]*\.?[0-9]+) .*?allow=(?P<allow>[^ ]+) .*?reason=(?P<reason>.*)$"
    )
    for ln in lines:
        m = pattern.search(ln)
        if m:
            parsed.append(
                {
                    "gate": _safe_float(m.group("gate")),
                    "allow": m.group("allow").strip(),
                    "reason": m.group("reason").strip(),
                    "raw": ln,
                }
            )
        elif ln.startswith("[signals]") and "zone refreshed" in ln:
            parsed.append({"gate": None, "allow": None, "reason": None, "raw": ln})
    return parsed


def _parse_entries(lines: Iterable[str]) -> list[dict]:
    parsed = []
    pattern = re.compile(r"\[enter\].*?active_min=(?P<active_min>[-+]?[0-9]*\.?[0-9]+)")
    for ln in lines:
        if "[enter]" not in ln:
            continue
        m = pattern.search(ln)
        active_min = _safe_float(m.group("active_min")) if m else None
        parsed.append({"active_min": active_min, "raw": ln})
    return parsed


@dataclass
class SkipEvent:
    line: str
    reason: str
    kind: str
    liq: float | None = None
    fund: float | None = None


def _parse_skips(lines: Iterable[str]) -> list[SkipEvent]:
    events: list[SkipEvent] = []
    leaf_pattern = re.compile(r"skip:\s*(?P<reason>.+)")
    edge_gate_pattern = re.compile(r"\[edge-gate\] skip: .*?reason=(?P<reason>[^ ]+)")
    lf_pattern = re.compile(
        r"\[lf-skip\].*?liq=(?P<liq>[-+]?[0-9]*\.?[0-9]+|None).*?fund=(?P<fund>[-+]?[0-9]*\.?[0-9]+|None).*"
    )
    for ln in lines:
        if "skip" not in ln:
            continue
        if "[lf-skip]" in ln:
            m = lf_pattern.search(ln)
            liq = _safe_float(m.group("liq")) if m else None
            fund = _safe_float(m.group("fund")) if m else None
            events.append(SkipEvent(line=ln, reason="lf-skip", kind="lf-skip", liq=liq, fund=fund))
            continue
        if "[edge-gate] skip" in ln:
            m = edge_gate_pattern.search(ln)
            reason = m.group("reason").strip() if m else "edge-gate"
            events.append(SkipEvent(line=ln, reason=reason, kind="edge-gate"))
            continue
        m = leaf_pattern.search(ln)
        if m:
            reason = m.group("reason").strip()
            kind = "generic"
            if "entropy-guard" in ln:
                kind = "entropy-guard"
            elif "kill-switch" in ln:
                kind = "kill-switch"
            elif "[skip-rk]" in ln or "skip rk" in ln:
                kind = "skip-rk"
            events.append(SkipEvent(line=ln, reason=reason, kind=kind))
    return events


def _bucket_active_min(values: list[float]) -> list[tuple[str, int, float]]:
    buckets = [("<1", lambda x: x < 1), ("1-3", lambda x: 1 <= x < 3), ("3-5", lambda x: 3 <= x < 5), ("5-10", lambda x: 5 <= x < 10), ("10+", lambda x: x >= 10)]
    result = []
    total = len(values)
    for label, fn in buckets:
        subset = [x for x in values if fn(x)]
        percent = len(subset) / max(total, 1) * 100
        result.append((label, len(subset), percent))
    return result


def _summarize_pnls(rows: list[tuple[float, float]]) -> dict[str, float]:
    pnls = [float(p) for _, p in rows]
    n = len(pnls)
    total = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    avg = total / n if n else 0.0
    avg_win = statistics.mean([p for p in pnls if p > 0]) if wins else 0.0
    avg_loss = statistics.mean([p for p in pnls if p < 0]) if losses else 0.0
    median = statistics.median(pnls) if n else 0.0
    return {
        "n": n,
        "total": total,
        "avg": avg,
        "winrate": wins / n if n else 0.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "median": median,
    }


def _format_summary(name: str, summary: dict[str, float]) -> str:
    return (
        f"{name}: count={summary['n']}  total={summary['total']:+.6f}  "
        f"avg={summary['avg']:+.6f}  winrate={summary['winrate']:.3f}  "
        f"avg_win={summary['avg_win']:+.6f}  avg_loss={summary['avg_loss']:+.6f}  "
        f"median={summary['median']:+.6f}"
    )


def _query_pnl_rows(db_path: Path, min_id: int | None) -> list[tuple[float | None, float]]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    if min_id is not None:
        cur.execute(
            "SELECT gate_score, pnl FROM trades WHERE pnl IS NOT NULL AND id > ? ORDER BY timestamp",
            (min_id,),
        )
    else:
        cur.execute("SELECT gate_score, pnl FROM trades WHERE pnl IS NOT NULL ORDER BY timestamp")
    rows = cur.fetchall()
    conn.close()
    return rows


def _print_pnl_summary(rows: list[tuple[float | None, float]], gate_threshold: float) -> None:
    all_rows = rows
    above = [(g, p) for g, p in rows if g is not None and g >= gate_threshold]
    below = [(g, p) for g, p in rows if g is not None and g < gate_threshold]
    missing = [(g, p) for g, p in rows if g is None]
    print("\n--- PnL summary (до/после gate threshold) ---")
    print(_format_summary("All closed trades", _summarize_pnls(all_rows)))
    print(_format_summary(f"Gate >= {gate_threshold:.2f}", _summarize_pnls(above)))
    print(_format_summary(f"Gate < {gate_threshold:.2f}", _summarize_pnls(below)))
    if missing:
        print(_format_summary("Gate missing", _summarize_pnls(missing)))


def _where_edge_closed(min_id: int | None) -> tuple[str, list]:
    parts = ["edge_real IS NOT NULL", "pnl IS NOT NULL"]
    params: list = []
    if min_id is not None and min_id > 0:
        parts.insert(0, "id > ?")
        params.append(min_id)
    return " AND ".join(parts), params


def _print_gate_buckets(db_path: Path, min_id: int | None) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    wh, params = _where_edge_closed(min_id)
    sql = f"""
SELECT
  CASE
    WHEN gate_score < 0 THEN '<0'
    WHEN gate_score < 0.1 THEN '0-0.1'
    WHEN gate_score < 0.3 THEN '0.1-0.3'
    ELSE '0.3+'
  END AS bucket,
  COUNT(*) AS n,
  AVG(pnl) AS avg_pnl,
  SUM(pnl) AS total_pnl,
  AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) AS wr
FROM trades
WHERE {wh}
GROUP BY 1
ORDER BY MIN(gate_score)
"""
    print("\n--- gate buckets ---")
    try:
        cur.execute(sql, params)
        rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        print(f"    (SQL error: {e})")
        conn.close()
        return
    if not rows:
        print("    (нет строк)")
        conn.close()
        return
    print(f"    {'bucket':<8} {'n':>6}  {'avg_pnl':>12}  {'total_pnl':>12}  {'wr':>6}")
    for b, n, avg_pnl, total_pnl, wr in rows:
        ap = avg_pnl if avg_pnl is not None else 0.0
        tp = total_pnl if total_pnl is not None else 0.0
        w = wr if wr is not None else 0.0
        print(f"    {str(b):<8} {int(n):6d}  {float(ap):+12.8f}  {float(tp):+12.8f}  {float(w):6.4f}")
    conn.close()


def _print_payoff_decomposition(db_path: Path, min_id: int | None) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    wh, params = _where_edge_closed(min_id)
    sql = f"""
SELECT
  CASE
    WHEN gate_score < 0 THEN '<0'
    WHEN gate_score < 0.1 THEN '0-0.1'
    WHEN gate_score < 0.3 THEN '0.1-0.3'
    ELSE '0.3+'
  END AS gate_bucket,

  COUNT(*) AS n,

  AVG(pnl) AS avg_pnl,
  AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) AS wr,

  AVG(CASE WHEN pnl > 0 THEN pnl END) AS avg_win,
  AVG(CASE WHEN pnl <= 0 THEN pnl END) AS avg_loss,

  ABS(
    AVG(CASE WHEN pnl > 0 THEN pnl END)
    /
    NULLIF(AVG(CASE WHEN pnl <= 0 THEN pnl END),0)
  ) AS payoff_ratio

FROM trades
WHERE {wh}
GROUP BY 1
ORDER BY 1
"""
    print("\n--- payoff decomposition ---")
    try:
        cur.execute(sql, params)
        rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        print(f"    (SQL error: {e})")
        conn.close()
        return
    if not rows:
        print("    (нет строк)")
        conn.close()
        return
    print(f"    {'bucket':<8} {'n':>6}  {'avg_pnl':>12}  {'wr':>6}  {'avg_win':>12}  {'avg_loss':>12}  {'payoff':>8}")
    for b, n, avg_pnl, wr, avg_win, avg_loss, payoff_ratio in rows:
        ap = avg_pnl if avg_pnl is not None else 0.0
        w = wr if wr is not None else 0.0
        aw = avg_win if avg_win is not None else 0.0
        al = avg_loss if avg_loss is not None else 0.0
        pr = payoff_ratio if payoff_ratio is not None else 0.0
        print(f"    {str(b):<8} {int(n):6d}  {float(ap):+12.8f}  {float(w):6.4f}  {float(aw):+12.8f}  {float(al):+12.8f}  {float(pr):8.4f}")
    conn.close()


def _print_exit_decomposition(db_path: Path, min_id: int | None) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    wh, params = _where_edge_closed(min_id)
    sql = f"""
SELECT
  CASE
    WHEN gate_score < 0 THEN '<0'
    WHEN gate_score < 0.1 THEN '0-0.1'
    WHEN gate_score < 0.3 THEN '0.1-0.3'
    ELSE '0.3+'
  END AS gate_bucket,

  exit_reason,
  COUNT(*) AS n,

  AVG(CAST(reverse_hit AS FLOAT)) AS reverse_rate

FROM trades
WHERE {wh} AND exit_reason IS NOT NULL
GROUP BY 1,2
ORDER BY 1,3 DESC
"""
    print("\n--- exit decomposition ---")
    try:
        cur.execute(sql, params)
        rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        print(f"    (SQL error: {e})")
        conn.close()
        return
    if not rows:
        print("    (нет строк с exit_reason)")
        conn.close()
        return
    print(f"    {'bucket':<8} {'exit_reason':<12} {'n':>6}  {'reverse':>8}")
    for b, er, n, reverse_rate in rows:
        rr = reverse_rate if reverse_rate is not None else 0.0
        print(f"    {str(b):<8} {str(er or 'None'):<12} {int(n):6d}  {float(rr):8.4f}")
    conn.close()


def _print_cross_buckets(db_path: Path, min_id: int | None) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    wh, params = _where_edge_closed(min_id)
    sql = f"""
SELECT
  CASE
    WHEN market_active_min < 5 THEN '3-5'
    WHEN market_active_min < 10 THEN '5-10'
    ELSE '10+'
  END AS active_bucket,

  CASE
    WHEN gate_score < 0 THEN '<0'
    WHEN gate_score < 0.1 THEN '0-0.1'
    WHEN gate_score < 0.3 THEN '0.1-0.3'
    ELSE '0.3+'
  END AS gate_bucket,

  COUNT(*) AS n,
  AVG(pnl) AS avg_pnl,
  SUM(pnl) AS total_pnl,
  AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) AS wr

FROM trades
WHERE {wh} AND market_active_min IS NOT NULL
GROUP BY 1,2
ORDER BY 1,2
"""
    print("\n--- cross (active_min × gate) buckets ---")
    try:
        cur.execute(sql, params)
        rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        print(f"    (SQL error: {e})")
        conn.close()
        return
    if not rows:
        print("    (нет строк с market_active_min)")
        conn.close()
        return
    print(f"    {'active':<6} {'gate':<8} {'n':>6}  {'avg_pnl':>12}  {'total_pnl':>12}  {'wr':>6}")
    for ab, gb, n, avg_pnl, total_pnl, wr in rows:
        ap = avg_pnl if avg_pnl is not None else 0.0
        tp = total_pnl if total_pnl is not None else 0.0
        w = wr if wr is not None else 0.0
        print(f"    {str(ab):<6} {str(gb):<8} {int(n):6d}  {float(ap):+12.8f}  {float(tp):+12.8f}  {float(w):6.4f}")
    conn.close()


def _print_skips(skips: list[SkipEvent]) -> None:
    print("\n--- skips_by_reason ---")
    if not skips:
        print("  (skip events not found in log)")
        return
    reason_counts = Counter(sk.reason for sk in skips)
    for reason, count in reason_counts.most_common():
        print(f"  {reason:<20} {count:>4d}")
    print("  --- skip kinds ---")
    kind_counts = Counter(sk.kind for sk in skips)
    for kind, count in kind_counts.most_common():
        print(f"  {kind:<20} {count:>4d}")


def _print_active_min(entries: list[dict]) -> None:
    print("\n--- active_min distribution на входах ---")
    active_mins = [e["active_min"] for e in entries if e["active_min"] is not None]
    if not active_mins:
        print("  (нет входов c active_min в логе)")
        return
    for label, count, pct in _bucket_active_min(active_mins):
        print(f"  {label:<6} {count:>4d} ({pct:5.1f}%)")
    print(f"  avg active_min = {statistics.mean(active_mins):.2f}")
    print(f"  median active_min = {statistics.median(active_mins):.2f}")


def _print_conversion(signals: list[dict], entries: list[dict]) -> None:
    print("\n--- conversion (signals → entries) ---")
    n_signals = len(signals)
    n_entries = len(entries)
    print(f"  signals = {n_signals}, entries = {n_entries}")
    print(f"  entry rate = {n_entries / max(n_signals, 1):.3f} entries / signal")
    gate_allow = [s for s in signals if s.get("allow") is not None]
    if gate_allow:
        allow_yes = sum(1 for s in gate_allow if s["allow"] in {"True", "1", "true", "yes"})
        print(f"  allow signals = {len(gate_allow)} ({allow_yes}/{len(gate_allow)} allowed)")
    if any(s.get("gate") is not None for s in signals):
        gates = [s["gate"] for s in signals if s.get("gate") is not None]
        print(f"  gate mean = {statistics.mean(gates):.3f}, min = {min(gates):.3f}, max = {max(gates):.3f}")


def _print_lf_contribution(skips: list[SkipEvent], signals_count: int | None = None) -> None:
    print("\n--- (l,f) вклад после gate (LF skip) ---")
    lf_events = [sk for sk in skips if sk.kind == "lf-skip"]
    if not lf_events:
        print("  (нет LF skip events в логе)")
        return
    liqs = [sk.liq for sk in lf_events if sk.liq is not None]
    funds = [sk.fund for sk in lf_events if sk.fund is not None]
    print(f"  LF-skip count = {len(lf_events)}")
    if signals_count is not None:
        print(f"  share of signals = {len(lf_events) / max(signals_count, 1):.3f}")
    if liqs:
        print(f"  liq: mean={statistics.mean(liqs):.3f} min={min(liqs):.3f} max={max(liqs):.3f}")
    if funds:
        print(f"  fund: mean={statistics.mean(funds):.3f} min={min(funds):.3f} max={max(funds):.3f}")
    if liqs and funds:
        combined = [abs(l) + abs(f) for l, f in zip(liqs, funds)]
        print(f"  avg |liq|+|fund| = {statistics.mean(combined):.3f}")


def _print_realized_vs_potential(db_path: Path, min_id: int | None) -> None:
    """Realized vs Potential: regime, avg_pnl, avg_mfe, avg_mae, p50_mfe, p90_mfe."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    wh, params = _where_edge_closed(min_id)
    
    # Fetch all data for percentile calculation
    sql_all = f"""
SELECT
  regime_key,
  pnl,
  mfe,
  mae
FROM trades
WHERE {wh} AND pnl IS NOT NULL AND mfe IS NOT NULL AND mae IS NOT NULL
ORDER BY regime_key, mfe
"""
    print("\n--- Realized vs Potential (regime breakdown) ---")
    try:
        cur.execute(sql_all, params)
        all_rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        print(f"    (SQL error: {e})")
        conn.close()
        return
    
    if not all_rows:
        print("    (нет данных)")
        conn.close()
        return
    
    # Group by regime and calculate stats
    regimes_data: dict[str | None, dict] = {}
    for regime, pnl, mfe, mae in all_rows:
        if regime not in regimes_data:
            regimes_data[regime] = {'pnls': [], 'mfes': [], 'maes': []}
        regimes_data[regime]['pnls'].append(float(pnl) if pnl is not None else 0.0)
        regimes_data[regime]['mfes'].append(float(mfe) if mfe is not None else 0.0)
        regimes_data[regime]['maes'].append(float(mae) if mae is not None else 0.0)
    
    def percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100.0)
        return sorted_data[min(idx, len(sorted_data) - 1)]
    
    print(f"    {'regime':<20} {'avg_pnl':>12}  {'avg_mfe':>12}  {'avg_mae':>12}  {'p50_mfe':>12}  {'p90_mfe':>12}")
    for regime in sorted(regimes_data.keys(), key=lambda r: len(regimes_data[r]['pnls']), reverse=True):
        data = regimes_data[regime]
        avg_pnl = statistics.mean(data['pnls']) if data['pnls'] else 0.0
        avg_mfe = statistics.mean(data['mfes']) if data['mfes'] else 0.0
        avg_mae = statistics.mean(data['maes']) if data['maes'] else 0.0
        p50_mfe = percentile(data['mfes'], 50)
        p90_mfe = percentile(data['mfes'], 90)
        print(f"    {str(regime or 'None'):<20} {float(avg_pnl):+12.8f}  {float(avg_mfe):+12.8f}  {float(avg_mae):+12.8f}  {float(p50_mfe):+12.8f}  {float(p90_mfe):+12.8f}")
    conn.close()


def _print_timeout_autopsy(db_path: Path, min_id: int | None) -> None:
    """Timeout autopsy (ГЛАВНЫЙ): только timeout trades. regime, n, avg_pnl, avg_mfe, avg_mae, capture_ratio."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    wh, params = _where_edge_closed(min_id)
    
    sql = f"""
SELECT
  regime_key,
  COUNT(*) AS n,
  AVG(pnl) AS avg_pnl,
  AVG(mfe) AS avg_mfe,
  AVG(mae) AS avg_mae,
  AVG(CASE WHEN mfe > 0 THEN pnl / mfe ELSE 0 END) AS capture_ratio
FROM trades
WHERE {wh} AND timeout_hit = 1 AND pnl IS NOT NULL AND mfe IS NOT NULL AND mae IS NOT NULL
GROUP BY regime_key
ORDER BY n DESC
"""
    print("\n--- TIMEOUT AUTOPSY (timeout-only trades; if avg_mfe >> avg_pnl then alpha leakage!) ---")
    try:
        cur.execute(sql, params)
        rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        print(f"    (SQL error: {e})")
        conn.close()
        return
    
    if not rows:
        print("    (нет timeout trades)")
        conn.close()
        return
    
    print(f"    {'regime':<20} {'n':>6}  {'avg_pnl':>12}  {'avg_mfe':>12}  {'avg_mae':>12}  {'capture_ratio':>12}")
    for regime, n, apnl, amfe, amae, cr in rows:
        apnl = apnl if apnl is not None else 0.0
        amfe = amfe if amfe is not None else 0.0
        amae = amae if amae is not None else 0.0
        cr = cr if cr is not None else 0.0
        print(f"    {str(regime or 'None'):<20} {int(n):6d}  {float(apnl):+12.8f}  {float(amfe):+12.8f}  {float(amae):+12.8f}  {float(cr):12.4f}")
    conn.close()


def _print_hold_decomposition(db_path: Path, min_id: int | None) -> None:
    """Hold decomposition: regime, avg_hold, win_hold, loss_hold."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    wh, params = _where_edge_closed(min_id)
    
    sql = f"""
SELECT
  regime_key,
  AVG(hold_min) AS avg_hold,
  AVG(CASE WHEN pnl > 0 THEN hold_min END) AS win_hold,
  AVG(CASE WHEN pnl <= 0 THEN hold_min END) AS loss_hold
FROM trades
WHERE {wh} AND hold_min IS NOT NULL
GROUP BY regime_key
ORDER BY avg_hold DESC
"""
    print("\n--- Hold Decomposition (winners need more time? losers die quickly? timeout truncates drift?) ---")
    try:
        cur.execute(sql, params)
        rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        print(f"    (SQL error: {e})")
        conn.close()
        return
    
    if not rows:
        print("    (нет данных с hold_min)")
        conn.close()
        return
    
    print(f"    {'regime':<20} {'avg_hold':>12}  {'win_hold':>12}  {'loss_hold':>12}")
    for regime, avg_h, win_h, loss_h in rows:
        avg_h = avg_h if avg_h is not None else 0.0
        win_h = win_h if win_h is not None else 0.0
        loss_h = loss_h if loss_h is not None else 0.0
        print(f"    {str(regime or 'None'):<20} {float(avg_h):12.4f}  {float(win_h):12.4f}  {float(loss_h):12.4f}")
    conn.close()


def _print_exit_decomposition_v2(db_path: Path, min_id: int | None) -> None:
    """Exit decomposition v2: по exit_reason (timeout, tp, sl, reverse): avg_mfe, avg_mae, avg_hold."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    wh, params = _where_edge_closed(min_id)
    
    sql = f"""
SELECT
  exit_reason,
  COUNT(*) AS n,
  AVG(mfe) AS avg_mfe,
  AVG(mae) AS avg_mae,
  AVG(hold_min) AS avg_hold
FROM trades
WHERE {wh} AND exit_reason IS NOT NULL AND mfe IS NOT NULL AND mae IS NOT NULL AND hold_min IS NOT NULL
GROUP BY exit_reason
ORDER BY n DESC
"""
    print("\n--- Exit Decomposition v2 (by exit_reason: avg_mfe, avg_mae, avg_hold) ---")
    try:
        cur.execute(sql, params)
        rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        print(f"    (SQL error: {e})")
        conn.close()
        return
    
    if not rows:
        print("    (нет данных)")
        conn.close()
        return
    
    print(f"    {'exit_reason':<12} {'n':>6}  {'avg_mfe':>12}  {'avg_mae':>12}  {'avg_hold':>12}")
    for er, n, amfe, amae, ah in rows:
        amfe = amfe if amfe is not None else 0.0
        amae = amae if amae is not None else 0.0
        ah = ah if ah is not None else 0.0
        print(f"    {str(er or 'None'):<12} {int(n):6d}  {float(amfe):+12.8f}  {float(amae):+12.8f}  {float(ah):12.4f}")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate & signal report from production logs and trade DB.")
    parser.add_argument("log", nargs="?", default=str(DEFAULT_LOG), help="Path to production log file")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to sqlite trades DB")
    parser.add_argument("--gate-threshold", type=float, default=None, help="Gate threshold for split summary")
    args = parser.parse_args()

    log_path = Path(args.log)
    db_path = Path(args.db)
    min_id = _report_min_id()
    gate_threshold = args.gate_threshold
    if gate_threshold is None:
        gate_threshold = _safe_float(os.environ.get("FIE_GATE_THRESHOLD")) or 0.30

    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    if not log_path.exists():
        print(f"Log not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    with log_path.open("r", errors="ignore") as f:
        log_lines = [ln.strip() for ln in f if ln.strip()]

    signals = _parse_signals(log_lines)
    entries = _parse_entries(log_lines)
    skips = _parse_skips(log_lines)
    pnl_rows = _query_pnl_rows(db_path, min_id)

    print("\n" + "=" * 70)
    print("GATE / SIGNAL REPORT")
    print("=" * 70)
    if min_id:
        print(f"clean window: trades id > {min_id}")
    print(f"db={db_path}")
    print(f"log={log_path}")
    print(f"gate threshold = {gate_threshold:.2f}")

    _print_pnl_summary(pnl_rows, gate_threshold)
    _print_gate_buckets(db_path, min_id)
    _print_payoff_decomposition(db_path, min_id)
    _print_exit_decomposition(db_path, min_id)
    _print_cross_buckets(db_path, min_id)
    _print_skips(skips)
    _print_active_min(entries)
    _print_conversion(signals, entries)
    _print_lf_contribution(skips, signals_count=len(signals))
    
    # === DECISIVE ANALYSIS ===
    _print_timeout_autopsy(db_path, min_id)
    _print_realized_vs_potential(db_path, min_id)
    _print_hold_decomposition(db_path, min_id)
    _print_exit_decomposition_v2(db_path, min_id)
    print()


if __name__ == "__main__":
    main()

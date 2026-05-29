#!/usr/bin/env python3
"""
Отчёт после edge-aware патча: size distribution, PnL, Kelly r, Sharpe, [edge-pre]/[edge-size] из лога.

Чистое окно по сделкам:
  export FIE_TRADE_REPORT_MIN_ID=$(sqlite3 db/fie_prod.sqlite "SELECT MAX(id) FROM trades;")
  # после накопления 30–50+ новых закрытий с edge_real:
  FIE_TRADE_REPORT_MIN_ID=$cut_id python3 scripts/report_edge_metrics.py

Шаг 10: SQL-бакеты по |edge_real| + ROUND(ABS(edge_real),2) для градиента edge → avg_pnl.

Запуск: python3 scripts/report_edge_metrics.py [/tmp/fie_loop.log]
"""
from __future__ import annotations

import os
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "db" / "fie_prod.sqlite"
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/fie_loop.log")


def _report_min_id() -> int | None:
    raw = os.environ.get("FIE_TRADE_REPORT_MIN_ID", "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
        return v if v > 0 else None
    except ValueError:
        return None


def _where_edge_closed(min_id: int | None) -> tuple[str, list]:
    parts = ["edge_real IS NOT NULL", "pnl IS NOT NULL"]
    params: list = []
    if min_id is not None and min_id > 0:
        parts.insert(0, "id > ?")
        params.append(min_id)
    return " AND ".join(parts), params


def _report_edge_buckets_sql(cur: sqlite3.Cursor, min_id: int | None) -> None:
    """CASE-бакеты 0.05–0.10 / 0.10–0.20 / 0.20+"""
    wh, params = _where_edge_closed(min_id)
    sql = f"""
SELECT
  CASE
    WHEN ABS(edge_real) < 0.10 THEN '0.05-0.10'
    WHEN ABS(edge_real) < 0.20 THEN '0.10-0.20'
    ELSE '0.20+'
  END AS bucket,
  COUNT(*) AS n,
  AVG(pnl) AS avg_pnl,
  AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) AS winrate
FROM trades
WHERE {wh}
GROUP BY 1
ORDER BY bucket
"""
    print("    --- SQL buckets (CASE |edge_real|) ---")
    try:
        cur.execute(sql, params)
        sql_rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        print(f"    (SQL error: {e})")
        return
    if not sql_rows:
        print("    (нет строк)")
        return
    print(f"    {'bucket':<14} {'n':>6}  {'avg_pnl':>12}  {'winrate':>8}")
    for b, n, avg_pnl, wr in sql_rows:
        ap = avg_pnl if avg_pnl is not None else 0.0
        w = wr if wr is not None else 0.0
        print(f"    {str(b):<14} {int(n):6d}  {float(ap):+12.8f}  {float(w):8.4f}")


def _report_edge_round_bucket_sql(cur: sqlite3.Cursor, min_id: int | None) -> None:
    """Ключевой тест: ROUND(ABS(edge_real),2) → n, avg_pnl (градиент edge ↑ → pnl ↑)."""
    wh, params = _where_edge_closed(min_id)
    sql = f"""
SELECT
  ROUND(ABS(edge_real), 2) AS edge_bucket,
  COUNT(*) AS n,
  AVG(pnl) AS avg_pnl
FROM trades
WHERE {wh}
GROUP BY 1
ORDER BY 1
"""
    print("    --- SQL ROUND(|edge_real|, 2) → avg_pnl (ключевой тест) ---")
    try:
        cur.execute(sql, params)
        rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        print(f"    (SQL error: {e})")
        return
    if not rows:
        print("    (нет строк)")
        return
    print(f"    {'edge_bucket':>12} {'n':>6}  {'avg_pnl':>12}")
    for bucket, n, avg_pnl in rows:
        ap = avg_pnl if avg_pnl is not None else 0.0
        print(f"    {float(bucket):12.2f} {int(n):6d}  {float(ap):+12.8f}")


def _report_edge_real_mispricing(cur: sqlite3.Cursor, min_id: int | None) -> None:
    """Срез по edge_real; при min_id — только новые строки после cut."""
    cur.execute("PRAGMA table_info(trades)")
    col_names = {r[1] for r in cur.fetchall()}
    print("\n  --- edge_real (mispricing vs market) ---")
    if "edge_real" not in col_names:
        print(
            "    (в trades нет колонок p_model/p_market/edge_real — "
            "после обновления кода запустите prod_loop один раз или init_db)"
        )
        return

    wh, params = _where_edge_closed(min_id)
    cur.execute(f"SELECT COUNT(*) FROM trades WHERE {wh}", params)
    cnt = int(cur.fetchone()[0])
    if min_id:
        print(f"    [clean window] id > {min_id}  →  закрытых с edge_real: {cnt}")
    else:
        print(
            "    [clean window] FIE_TRADE_REPORT_MIN_ID не задан — вся история. "
            "Мягкий cut: export FIE_TRADE_REPORT_MIN_ID=$(sqlite3 db/fie_prod.sqlite \"SELECT MAX(id) FROM trades;\")"
        )
        print(f"    закрытых с edge_real (все id): {cnt}")

    cur.execute(
        f"""
        SELECT edge_real, pnl, kelly_fraction, regime_key
        FROM trades
        WHERE {wh}
        ORDER BY timestamp
        """,
        params,
    )
    rows = cur.fetchall()
    if not rows:
        print("    (нет закрытых сделок с edge_real в этом окне)")
        _report_edge_buckets_sql(cur, min_id)
        _report_edge_round_bucket_sql(cur, min_id)
        return

    edges = [float(r[0]) for r in rows]
    pnls = [float(r[1]) for r in rows]
    abs_edges = [abs(e) for e in edges]
    n = len(rows)
    wins = sum(1 for p in pnls if p > 0)
    strong = [(e, p) for e, p in zip(edges, pnls) if abs(e) > 0.10]
    print(f"    n={n}  WR(all)={wins/n:.3f}  avg_pnl={sum(pnls)/n:+.8f}")
    if strong:
        ws = sum(1 for _, p in strong if p > 0)
        print(
            f"    |edge_real|>0.10: n={len(strong)}  WR={ws/len(strong):.3f}  "
            f"avg_pnl={sum(p for _, p in strong)/len(strong):+.8f}"
        )
    else:
        print("    (пока нет сделок с |edge_real|>0.10 в выборке)")

    bands = [
        ("0.05–0.10", lambda a: 0.05 <= a < 0.10),
        ("0.10–0.20", lambda a: 0.10 <= a < 0.20),
        ("0.20+", lambda a: a >= 0.20),
    ]
    print("    --- |edge_real| buckets (Python) ---")
    for label, fn in bands:
        idx = [i for i, a in enumerate(abs_edges) if fn(a)]
        if not idx:
            print(f"      {label:<12} n=0")
            continue
        ap = statistics.mean([pnls[i] for i in idx])
        wr = sum(1 for i in idx if pnls[i] > 0) / len(idx)
        print(f"      {label:<12} n={len(idx):4d}  WR={wr:.3f}  avg_pnl={ap:+.8f}")

    pos = [(e, p) for e, p in zip(edges, pnls) if e > 0]
    neg = [(e, p) for e, p in zip(edges, pnls) if e < 0]
    if pos:
        ap = sum(p for _, p in pos) / len(pos)
        wp = sum(1 for _, p in pos if p > 0) / len(pos)
        print(f"    edge_real>0: n={len(pos)}  WR={wp:.3f}  avg_pnl={ap:+.8f}")
    if neg:
        an = sum(p for _, p in neg) / len(neg)
        wn = sum(1 for _, p in neg if p > 0) / len(neg)
        print(f"    edge_real<0: n={len(neg)}  WR={wn:.3f}  avg_pnl={an:+.8f}")

    paired = [(float(r[0]), float(r[2])) for r in rows if r[2] is not None]
    if len(paired) > 1:
        e2 = [a for a, _ in paired]
        k2 = [b for _, b in paired]
        mx = statistics.mean(k2)
        my = statistics.mean(e2)
        cov = sum((x - mx) * (y - my) for x, y in zip(k2, e2)) / len(k2)
        sx = (sum((x - mx) ** 2 for x in k2) / len(k2)) ** 0.5
        sy = (sum((y - my) ** 2 for y in e2) / len(k2)) ** 0.5
        rk = cov / (sx * sy) if sx * sy > 0 else 0.0
        print(f"    Kelly r (kelly_fraction↔edge_real): {rk:+.4f}")

    _report_edge_buckets_sql(cur, min_id)
    _report_edge_round_bucket_sql(cur, min_id)


def main() -> None:
    min_id = _report_min_id()
    conn = sqlite3.connect(str(DB))
    cur = conn.cursor()
    cur.execute(
        "SELECT pnl, size FROM trades WHERE pnl IS NOT NULL ORDER BY timestamp"
    )
    rows = cur.fetchall()
    cur.execute(
        "SELECT equity, drawdown, sharpe_rolling FROM portfolio_snapshots "
        "ORDER BY timestamp DESC LIMIT 1"
    )
    snap = cur.fetchone()

    pnls = [r[0] for r in rows]
    sizes = [r[1] for r in rows if r[1] is not None]
    n = len(pnls)

    print("\n" + "=" * 60)
    print("  EDGE RUN REPORT")
    print("=" * 60)
    if min_id:
        print(f"  (clean window: edge_real-секция только id > {min_id})")
    print(f"\n  trades={n}")
    print(f"  PnL total     : {sum(pnls):+.6f}")
    print(f"  Expectancy    : {sum(pnls)/max(n,1):+.8f} / trade")
    if snap:
        sh = snap[2]
        print(f"  Equity        : {snap[0]:.6f}")
        print(f"  Sharpe (roll) : {sh:.4f}" if sh is not None else "  Sharpe: n/a")
        print(f"  DD            : {snap[1]*100:.4f}%")

    if len(sizes) > 1:
        mx = statistics.mean(sizes)
        my = statistics.mean(pnls)
        cov = sum((x - mx) * (y - my) for x, y in zip(sizes, pnls)) / len(sizes)
        sx = (sum((x - mx) ** 2 for x in sizes) / len(sizes)) ** 0.5
        sy = (sum((y - my) ** 2 for y in pnls) / len(sizes)) ** 0.5
        r = cov / (sx * sy) if sx * sy > 0 else 0.0
        print(f"\n  Kelly r (size↔PnL): {r:+.4f}")

    print("\n  --- Size distribution ---")
    bands = [
        ("tiny   <0.025", lambda s: s < 0.025),
        ("mid    0.025–0.15", lambda s: 0.025 <= s < 0.15),
        ("big    ≥0.15", lambda s: s >= 0.15),
    ]
    for label, fn in bands:
        idx = [i for i, s in enumerate(sizes) if fn(s)]
        if not idx:
            continue
        ap = statistics.mean([pnls[i] for i in idx])
        share = len(idx) / len(sizes) * 100
        print(f"    {label:<22} n={len(idx):4d} ({share:5.1f}%)  avg_pnl={ap:+.6f}")

    if not LOG.exists():
        print("\n  --- [edge-pre] last 15 lines ---")
        print(f"    (log not found: {LOG})")
        print("\n  --- [edge-size] last 15 lines ---")
        print(f"    (log not found: {LOG})")
    else:
        log_text = LOG.read_text(errors="ignore").splitlines()
        pre_lines = [ln.strip() for ln in log_text if "[edge-pre]" in ln]
        edge_lines = [ln.strip() for ln in log_text if "[edge-size]" in ln]
        print("\n  --- [edge-pre] last 15 lines ---")
        for ln in pre_lines[-15:]:
            print(f"    {ln}")
        if not pre_lines:
            print("    (no [edge-pre] yet)")
        print("\n  --- [edge-size] last 15 lines ---")
        for ln in edge_lines[-15:]:
            print(f"    {ln}")
        if not edge_lines:
            print("    (no [edge-size] yet — or only SKIP paths)")

    _report_edge_real_mispricing(cur, min_id)
    conn.close()
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Один прогон по SQLite: энтропия p_model (квантование / два состояния).

  python3 scripts/audit_trades_p_model.py
  python3 scripts/audit_trades_p_model.py /path/to/fie_prod.sqlite
"""
from __future__ import annotations

import math
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "db" / "fie_prod.sqlite"


def main() -> None:
    if not DB.exists():
        print(f"DB not found: {DB}")
        sys.exit(1)
    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    cur.execute("PRAGMA table_info(trades)")
    cols = {r[1] for r in cur.fetchall()}
    if "p_model" not in cols:
        print("trades.p_model отсутствует — обновите схему/init_db.")
        sys.exit(1)
    cur.execute(
        """
        SELECT p_model FROM trades
        WHERE pnl IS NOT NULL AND p_model IS NOT NULL
        ORDER BY id DESC
        LIMIT 5000
        """
    )
    rows = [float(r[0]) for r in cur.fetchall()]
    n = len(rows)
    if n == 0:
        print("Нет закрытых сделок с p_model.")
        return
    arr = rows
    u = sorted(set(round(x, 8) for x in arr))
    print(f"n={n}  distinct_p (8dp)={len(u)}")
    print(f"min/mean/max={min(arr):.8f} {sum(arr)/n:.8f} {max(arr):.8f}")
    mu = sum(arr) / n
    var = sum((x - mu) ** 2 for x in arr) / max(n - 1, 1)
    print(f"std={math.sqrt(var):.8f}")
    pos = sum(1 for x in arr if x > 0.5)
    print(f"share_pos(p>0.5)={pos / n:.4f}")
    if len(u) <= 30:
        from collections import Counter

        c = Counter(round(x, 4) for x in arr)
        top = c.most_common(15)
        print("top buckets (4dp):", top)
    print("sample repr:", [repr(x) for x in arr[:5]])


if __name__ == "__main__":
    main()

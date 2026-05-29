#!/usr/bin/env python3
from pathlib import Path
import sqlite3
import time
import argparse
from datetime import datetime

TEMPLATE = """
{ts} | n={n} | avg_pnl={avg_pnl:+.8e} | capture_ratio={capture_ratio:+.6f} | avg_mfe={avg_mfe:+.8e} | avg_mae={avg_mae:+.8e} | avg_hold={avg_hold:+.3f} | tp={tp_rate:.3f} sl={sl_rate:.3f} timeout={timeout_rate:.3f}
"""

Q_METRICS = '''
SELECT
  COUNT(*) as n,
  AVG(pnl) as avg_pnl,
  AVG(mfe) as avg_mfe,
  AVG(mae) as avg_mae,
  AVG(CASE WHEN mfe > 0 THEN pnl/mfe END) as capture_ratio,
  AVG(hold_min) as avg_hold,
  SUM(CASE WHEN exit_reason='timeout' THEN 1 ELSE 0 END)*1.0/COUNT(*) as timeout_rate,
  SUM(CASE WHEN exit_reason='tp' THEN 1 ELSE 0 END)*1.0/COUNT(*) as tp_rate,
  SUM(CASE WHEN exit_reason='sl' THEN 1 ELSE 0 END)*1.0/COUNT(*) as sl_rate
FROM trades
WHERE gate_score >= ? AND gate_score < ? AND mfe IS NOT NULL AND pnl IS NOT NULL AND hold_min IS NOT NULL
AND id >= ?
'''

Q_COUNT = '''
SELECT COUNT(*) FROM trades WHERE gate_score >= ? AND gate_score < ? AND mfe IS NOT NULL AND pnl IS NOT NULL AND hold_min IS NOT NULL AND id >= ?
'''


def compute_metrics(conn, gate_min, gate_max, cut):
    cur = conn.cursor()
    cur.execute(Q_METRICS, (gate_min, gate_max, cut))
    row = cur.fetchone()
    if not row:
        return None
    n, avg_pnl, avg_mfe, avg_mae, capture_ratio, avg_hold, timeout_rate, tp_rate, sl_rate = row
    return {
        'n': int(n) if n is not None else 0,
        'avg_pnl': avg_pnl or 0.0,
        'avg_mfe': avg_mfe or 0.0,
        'avg_mae': avg_mae or 0.0,
        'capture_ratio': capture_ratio or 0.0,
        'avg_hold': avg_hold or 0.0,
        'timeout_rate': timeout_rate or 0.0,
        'tp_rate': tp_rate or 0.0,
        'sl_rate': sl_rate or 0.0,
    }


def count_cohort(conn, gate_min, gate_max, cut):
    cur = conn.cursor()
    cur.execute(Q_COUNT, (gate_min, gate_max, cut))
    return cur.fetchone()[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db', default='db/fie_prod.sqlite')
    p.add_argument('--cut', type=int, required=True)
    p.add_argument('--gate-min', type=float, default=0.1)
    p.add_argument('--gate-max', type=float, default=0.3)
    p.add_argument('--batch', type=int, default=20)
    p.add_argument('--poll', type=int, default=10)
    p.add_argument('--once', action='store_true')
    args = p.parse_args()

    dbp = Path(args.db)
    if not dbp.exists():
        print('DB not found:', dbp)
        return

    conn = sqlite3.connect(str(dbp))
    gate_min = args.gate_min
    gate_max = args.gate_max
    cut = args.cut
    batch = args.batch
    poll = args.poll

    last_count = count_cohort(conn, gate_min, gate_max, cut)
    # initial print
    m = compute_metrics(conn, gate_min, gate_max, cut)
    ts = datetime.utcnow().isoformat()
    if m:
        print(TEMPLATE.format(ts=ts, **m))
    else:
        print(ts, 'no data')

    if args.once:
        return

    try:
        while True:
            time.sleep(poll)
            new_count = count_cohort(conn, gate_min, gate_max, cut)
            if new_count >= last_count + batch:
                m = compute_metrics(conn, gate_min, gate_max, cut)
                ts = datetime.utcnow().isoformat()
                if m:
                    print(TEMPLATE.format(ts=ts, **m))
                last_count = new_count
    except KeyboardInterrupt:
        print('stopped')


if __name__ == '__main__':
    main()

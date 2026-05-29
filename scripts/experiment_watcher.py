#!/usr/bin/env python3
from pathlib import Path
import sqlite3
import time
import argparse
import csv
import json
import os
from datetime import datetime

Q_COUNT = '''
SELECT COUNT(*) FROM trades WHERE gate_score >= ? AND gate_score < ? AND mfe IS NOT NULL AND pnl IS NOT NULL AND hold_min IS NOT NULL AND id >= ?
'''

Q_METRICS = '''
SELECT
  COUNT(*) as n,
  SUM(pnl) as total_pnl,
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
AND id {cmp} ?
'''

THRESHOLDS = [50, 100, 200]
DEFAULT_OUTPUT_DIR = Path('logs/experiment_watcher')
CSV_FIELDS = [
    'timestamp', 'label', 'n', 'total_pnl', 'avg_pnl', 'capture_ratio', 'avg_mfe',
    'avg_mae', 'avg_hold', 'tp_rate', 'sl_rate', 'timeout_rate',
    'baseline_n', 'baseline_total_pnl', 'baseline_avg_pnl', 'baseline_capture_ratio',
    'baseline_avg_mfe', 'baseline_avg_mae', 'baseline_avg_hold',
    'baseline_tp_rate', 'baseline_sl_rate', 'baseline_timeout_rate',
    'delta_total_pnl', 'delta_avg_pnl', 'delta_capture_ratio', 'delta_avg_mfe',
    'delta_avg_mae', 'delta_avg_hold', 'delta_tp_rate', 'delta_sl_rate', 'delta_timeout_rate'
]


def metrics(conn, gate_min, gate_max, cut, cmp):
    cur = conn.cursor()
    cur.execute(Q_METRICS.replace('{cmp}', cmp), (gate_min, gate_max, cut))
    r = cur.fetchone()
    if not r:
        return None
    n, total_pnl, avg_pnl, avg_mfe, avg_mae, capture_ratio, avg_hold, timeout_rate, tp_rate, sl_rate = r
    return {
        'n': int(n or 0),
        'total_pnl': float(total_pnl or 0.0),
        'avg_pnl': float(avg_pnl or 0.0),
        'avg_mfe': float(avg_mfe or 0.0),
        'avg_mae': float(avg_mae or 0.0),
        'capture_ratio': float(capture_ratio or 0.0),
        'avg_hold': float(avg_hold or 0.0),
        'timeout_rate': float(timeout_rate or 0.0),
        'tp_rate': float(tp_rate or 0.0),
        'sl_rate': float(sl_rate or 0.0),
    }


def count(conn, gate_min, gate_max, cut):
    cur = conn.cursor()
    cur.execute(Q_COUNT, (gate_min, gate_max, cut))
    return int(cur.fetchone()[0] or 0)


def print_report(label, m):
    ts = datetime.utcnow().isoformat()
    print(f"{ts} | {label} | n={m['n']} | total_pnl={m['total_pnl']:+.8e} | avg_pnl={m['avg_pnl']:+.8e} | capture_ratio={m['capture_ratio']:+.6f} | avg_mfe={m['avg_mfe']:+.8e} | avg_mae={m['avg_mae']:+.8e} | avg_hold={m['avg_hold']:+.3f} | tp={m['tp_rate']:.3f} sl={m['sl_rate']:.3f} timeout={m['timeout_rate']:.3f}")


def delta_metrics(current, baseline):
    return {
        'n': current['n'],
        'total_pnl': current['total_pnl'] - baseline['total_pnl'],
        'avg_pnl': current['avg_pnl'] - baseline['avg_pnl'],
        'capture_ratio': current['capture_ratio'] - baseline['capture_ratio'],
        'avg_mfe': current['avg_mfe'] - baseline['avg_mfe'],
        'avg_mae': current['avg_mae'] - baseline['avg_mae'],
        'avg_hold': current['avg_hold'] - baseline['avg_hold'],
        'timeout_rate': current['timeout_rate'] - baseline['timeout_rate'],
        'tp_rate': current['tp_rate'] - baseline['tp_rate'],
        'sl_rate': current['sl_rate'] - baseline['sl_rate'],
    }


def print_delta(baseline, current):
    d = delta_metrics(current, baseline)
    print('BASELINE:')
    print_report('baseline', baseline)
    print('CURRENT:')
    print_report('current', current)
    print('DELTA:')
    print(f"  avg_pnl={d['avg_pnl']:+.8e} | capture_ratio={d['capture_ratio']:+.6f} | avg_mfe={d['avg_mfe']:+.8e} | timeout_rate={d['timeout_rate']:+.3f} | tp_rate={d['tp_rate']:+.3f} | avg_mae={d['avg_mae']:+.8e} | avg_hold={d['avg_hold']:+.3f} | total_pnl={d['total_pnl']:+.8e}")


def write_config_snapshot(out_dir, args):
    env_keys = [
        'FIE_EXIT_TIMEOUT_MULTIPLIER',
        'FIE_EXIT_TIMEOUT_GATE_MIN',
        'FIE_EXIT_TIMEOUT_GATE_MAX',
        'FIE_MARKET_GATE_ENABLED',
        'FIE_MARKET_GATE_WARMUP_MIN',
        'FIE_MARKET_GATE_PEAK_MIN',
        'FIE_LF_SKIP_ENABLED',
        'FIE_SKIP_LF_PAIRS',
        'FIE_EDGE_ENTER_LOG',
        'FIE_DEBUG_EDGE_PIPELINE',
        'FIE_VARIANCE_MIN',
        'FIE_VARIANCE_WARMUP',
        'FIE_HOLD_STEPS',
        'FIE_TP_PCT',
        'FIE_SL_PCT',
    ]
    cfg = {
        'timestamp': datetime.utcnow().isoformat(),
        'db': str(args.db),
        'cut_id': args.cut,
        'gate_min': args.gate_min,
        'gate_max': args.gate_max,
        'poll': args.poll,
        'baseline': args.baseline,
        'output_dir': str(args.out_dir),
        'env': {k: os.environ.get(k) for k in env_keys},
    }
    cfg_path = out_dir / f"experiment_config_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print('Wrote experiment config snapshot to', cfg_path)
    return cfg_path


def ensure_csv(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'experiment_snapshots.csv'
    if not csv_path.exists():
        with csv_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
    return csv_path


def append_csv_row(csv_path, label, m, baseline=None):
    d = delta_metrics(m, baseline) if baseline else {k: '' for k in ['total_pnl', 'avg_pnl', 'capture_ratio', 'avg_mfe', 'avg_mae', 'avg_hold', 'tp_rate', 'sl_rate', 'timeout_rate']}
    row = {
        'timestamp': datetime.utcnow().isoformat(),
        'label': label,
        'n': m['n'],
        'total_pnl': m['total_pnl'],
        'avg_pnl': m['avg_pnl'],
        'capture_ratio': m['capture_ratio'],
        'avg_mfe': m['avg_mfe'],
        'avg_mae': m['avg_mae'],
        'avg_hold': m['avg_hold'],
        'tp_rate': m['tp_rate'],
        'sl_rate': m['sl_rate'],
        'timeout_rate': m['timeout_rate'],
        'baseline_n': baseline['n'] if baseline else '',
        'baseline_total_pnl': baseline['total_pnl'] if baseline else '',
        'baseline_avg_pnl': baseline['avg_pnl'] if baseline else '',
        'baseline_capture_ratio': baseline['capture_ratio'] if baseline else '',
        'baseline_avg_mfe': baseline['avg_mfe'] if baseline else '',
        'baseline_avg_mae': baseline['avg_mae'] if baseline else '',
        'baseline_avg_hold': baseline['avg_hold'] if baseline else '',
        'baseline_tp_rate': baseline['tp_rate'] if baseline else '',
        'baseline_sl_rate': baseline['sl_rate'] if baseline else '',
        'baseline_timeout_rate': baseline['timeout_rate'] if baseline else '',
        'delta_total_pnl': d['total_pnl'] if baseline else '',
        'delta_avg_pnl': d['avg_pnl'] if baseline else '',
        'delta_capture_ratio': d['capture_ratio'] if baseline else '',
        'delta_avg_mfe': d['avg_mfe'] if baseline else '',
        'delta_avg_mae': d['avg_mae'] if baseline else '',
        'delta_avg_hold': d['avg_hold'] if baseline else '',
        'delta_tp_rate': d['tp_rate'] if baseline else '',
        'delta_sl_rate': d['sl_rate'] if baseline else '',
        'delta_timeout_rate': d['timeout_rate'] if baseline else '',
    }
    with csv_path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def append_event_log(out_dir, event_name, m, threshold=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'experiment_thresholds.log'
    ts = datetime.utcnow().isoformat()
    parts = [f'{ts} | {event_name}', f'n={m["n"]}', f'avg_pnl={m["avg_pnl"]:+.8e}', f'capture_ratio={m["capture_ratio"]:+.6f}', f'avg_mfe={m["avg_mfe"]:+.8e}', f'avg_mae={m["avg_mae"]:+.8e}', f'avg_hold={m["avg_hold"]:+.3f}', f'tp={m["tp_rate"]:.3f}', f'sl={m["sl_rate"]:.3f}', f'timeout={m["timeout_rate"]:.3f}']
    if threshold is not None:
        parts.append(f'threshold={threshold}')
    with log_path.open('a') as f:
        f.write(' | '.join(parts) + '\n')
    print('Wrote threshold event to', log_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db', default='db/fie_prod.sqlite')
    p.add_argument('--cut', type=int, required=True)
    p.add_argument('--gate-min', type=float, default=0.1)
    p.add_argument('--gate-max', type=float, default=0.3)
    p.add_argument('--poll', type=int, default=10)
    p.add_argument('--baseline', action='store_true', help='Print baseline delta on startup')
    p.add_argument('--once', action='store_true', help='Run one report and exit')
    p.add_argument('--out-dir', default=str(DEFAULT_OUTPUT_DIR), help='Directory for CSV snapshots and event logs')
    args = p.parse_args()

    dbp = Path(args.db)
    if not dbp.exists():
        print('DB not found:', dbp)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ensure_csv(out_dir)
    write_config_snapshot(out_dir, args)

    conn = sqlite3.connect(str(dbp))
    gate_min = args.gate_min
    gate_max = args.gate_max
    cut = args.cut
    poll = args.poll

    seen = 0
    reported = set()

    baseline = metrics(conn, gate_min, gate_max, cut, '<')
    current = metrics(conn, gate_min, gate_max, cut, '>=')

    if args.baseline:
        if baseline is None:
            print('No baseline data available.')
        else:
            print_delta(baseline, current)
    if current:
        print_report('current', current)
        append_csv_row(csv_path, 'current', current, baseline)
    else:
        print(datetime.utcnow().isoformat(), 'no data')

    if args.once:
        return

    try:
        while True:
            time.sleep(poll)
            cur_n = count(conn, gate_min, gate_max, cut)
            if cur_n >= seen + 20:
                current = metrics(conn, gate_min, gate_max, cut, '>=')
                if current:
                    print_report('current', current)
                    append_csv_row(csv_path, 'current', current, baseline)
                seen = cur_n
            for t in THRESHOLDS:
                if cur_n >= t and t not in reported:
                    current = metrics(conn, gate_min, gate_max, cut, '>=')
                    print('\n' + '='*40)
                    print(f"THRESHOLD REACHED: n >= {t}")
                    if current:
                        print_report('current', current)
                        append_csv_row(csv_path, f'threshold_{t}', current, baseline)
                        append_event_log(out_dir, 'threshold_reached', current, threshold=t)
                        if baseline is not None:
                            print_delta(baseline, current)
                    else:
                        print('no data')
                    print('='*40 + '\n')
                    reported.add(t)
    except KeyboardInterrupt:
        print('Watcher stopped by user')

if __name__ == '__main__':
    main()

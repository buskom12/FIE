#!/usr/bin/env bash
# Быстрый снимок: процессы + хвост лога + число сделок в SQLite (если есть БД)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB="$ROOT/db/fie_prod.sqlite"

echo "=== prod_loop processes ==="
pgrep -fl "services.prod_loop|prod_loop.py" 2>/dev/null || echo "(нет)"

echo ""
echo "=== last 12 lines /tmp/fie_loop.log ==="
tail -n 12 /tmp/fie_loop.log 2>/dev/null || echo "(нет лога)"

if [[ -f "$DB" ]]; then
  echo ""
  echo "=== trades (sqlite) ==="
  sqlite3 "$DB" "SELECT COUNT(*) AS closed FROM trades WHERE pnl IS NOT NULL;" 2>/dev/null | awk '{print "closed_total:", $1}'
  sqlite3 "$DB" "SELECT COUNT(*) FROM trades WHERE id > (SELECT COALESCE(MAX(id),0)-500 FROM trades) AND pnl IS NOT NULL AND p_model IS NOT NULL;" 2>/dev/null | awk '{print "last_window ~500 with p_model:", $1}' || true
fi

#!/usr/bin/env bash
# Аккуратный рестарт движка: убирает дубли prod_loop, поднимает один экземпляр через start_prod_loop.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[FIE] stopping prod_loop…"
pkill -9 -f "services.prod_loop" 2>/dev/null || true
pkill -9 -f "prod_loop.py" 2>/dev/null || true
pkill -9 -f "caffeinate -i python3 -m services.prod_loop" 2>/dev/null || true
sleep 2

if pgrep -f "services.prod_loop" >/dev/null 2>&1; then
  echo "[FIE] warn: процесс всё ещё жив; повторный pkill" >&2
  pkill -9 -f "services.prod_loop" 2>/dev/null || true
  sleep 1
fi

echo "[FIE] starting (log /tmp/fie_loop.log)…"
nohup "$ROOT/start_prod_loop.sh" >/dev/null 2>&1 </dev/null &
sleep 1
if pgrep -f "services.prod_loop" >/dev/null 2>&1; then
  echo "[FIE] ok pid: $(pgrep -f 'services.prod_loop' | tr '\n' ' ')"
else
  echo "[FIE] ошибка: процесс не поднялся, см. tail /tmp/fie_loop.log" >&2
  exit 1
fi

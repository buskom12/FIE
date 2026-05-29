#!/usr/bin/env bash
# FIE — Paper Mode Runner
# Запускает: prod_loop + prod API + дашборд
# Использование: bash run_paper_mode.sh [--hold 60] [--tp 0.05] [--sl 0.05]

set -e

HOLD_STEPS=60
TP_PCT=0.05
SL_PCT=0.05
POLL_SEC=5
API_PORT=8001
DASH_PORT=8501

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --hold) HOLD_STEPS="$2"; shift 2 ;;
        --tp)   TP_PCT="$2";    shift 2 ;;
        --sl)   SL_PCT="$2";    shift 2 ;;
        --poll) POLL_SEC="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; shift ;;
    esac
done

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  FIE Paper Mode"
echo "  HOLD_STEPS : $HOLD_STEPS  ($(( HOLD_STEPS * POLL_SEC ))s per position)"
echo "  TP / SL    : $TP_PCT / $SL_PCT"
echo "  API port   : $API_PORT"
echo "  Dash port  : $DASH_PORT"
echo "============================================"

# Kill previous instances
pkill -f "prod_loop"    2>/dev/null || true
pkill -f "prod_main"    2>/dev/null || true
pkill -f "prod_app.py"  2>/dev/null || true
sleep 1

# 1. prod_loop (engine)
echo "[1/3] Starting prod_loop..."
# ACS: 0=выкл, 1=live, 2=shadow (только лог [conf-scale-shadow])
FIE_CONF_SCALE=${FIE_CONF_SCALE:-0} \
FIE_POLL_SECONDS=$POLL_SEC \
FIE_HOLD_STEPS=$HOLD_STEPS \
FIE_TP_PCT=$TP_PCT \
FIE_SL_PCT=$SL_PCT \
    python3 -m services.prod_loop \
    > "$LOG_DIR/prod_loop.log" 2>&1 &
LOOP_PID=$!
echo "      PID=$LOOP_PID  log: logs/prod_loop.log"

# 2. prod API
echo "[2/3] Starting prod API on :$API_PORT..."
python3 -m uvicorn api.prod_main:app \
    --host 0.0.0.0 \
    --port $API_PORT \
    > "$LOG_DIR/prod_api.log" 2>&1 &
API_PID=$!
echo "      PID=$API_PID  log: logs/prod_api.log"

sleep 2

# 3. dashboard
echo "[3/3] Starting dashboard on :$DASH_PORT..."
python3 -m streamlit run dashboard/prod_app.py \
    --server.port $DASH_PORT \
    --server.headless true \
    > "$LOG_DIR/dashboard.log" 2>&1 &
DASH_PID=$!
echo "      PID=$DASH_PID  log: logs/dashboard.log"

sleep 3

echo ""
echo "✓ All services running:"
echo "   Dashboard  → http://localhost:$DASH_PORT"
echo "   API docs   → http://localhost:$API_PORT/docs"
echo "   Metrics    → http://localhost:$API_PORT/metrics"
echo ""
echo "PIDs: loop=$LOOP_PID  api=$API_PID  dash=$DASH_PID"
echo "Logs: $LOG_DIR/"
echo ""
echo "Ctrl+C to stop all."

# Trap: kill all on Ctrl+C
cleanup() {
    echo ""
    echo "Stopping all FIE processes..."
    kill $LOOP_PID $API_PID $DASH_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup INT TERM

# Tail loop log in foreground so you see live output
echo "--- prod_loop output (live) ---"
tail -f "$LOG_DIR/prod_loop.log"

#!/usr/bin/env bash
# Локальный запуск prod_loop — stdout+stderr ТОЛЬКО в /tmp/fie_loop.log (append).
#
# Фон:  ./start_prod_loop.sh &
# Рестарт одной командой:  scripts/fie_prod_restart.sh
# Статус:                 scripts/fie_status.sh
#
# Руками (эквивалент):
#   cd /path/to/FIE && python3 services/prod_loop.py >> /tmp/fie_loop.log 2>&1 &
#   # или: python3 -m services.prod_loop >> /tmp/fie_loop.log 2>&1 &
#
# По умолчанию: EDGE_ONLY + «чистый эксперимент» (MARKET_MAP не режет, zone allow=True, без skip rk, без corr).
# Вернуть защиту: FIE_SIGNAL_EXPERIMENT=0 FIE_MARKET_MAP_FILTER=1 и задай FIE_SKIP_RK_KEYS при необходимости.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Один живой процесс (macOS без flock — проверка по процессам).
if pgrep -f "services.prod_loop" >/dev/null 2>&1; then
  echo "[FIE] prod_loop уже запущен. Остановка: pkill -f 'services.prod_loop' или scripts/fie_prod_restart.sh" >&2
  exit 1
fi

# --- Стабилизация (см. [rollback] / [corr-defense] / [skip-rk] в логе) ---
# :- подставляет EDGE_ONLY и при unset, и при пустой строке (пустой env иначе отключает откат).
export FIE_SIZE_MODE="${FIE_SIZE_MODE:-EDGE_ONLY}"
export FIE_SAFE_MODE="${FIE_SAFE_MODE:-0}"
export FIE_SIMPLE_KELLY_SIZE="${FIE_SIMPLE_KELLY_SIZE:-0}"
export FIE_SIMPLE_BASE_SIZE="${FIE_SIMPLE_BASE_SIZE:-0.08}"
export FIE_EDGE_PM_FILTER="${FIE_EDGE_PM_FILTER:-0}"
export FIE_CONF_SCALE="${FIE_CONF_SCALE:-0}"
export FIE_ARC_ENABLED="${FIE_ARC_ENABLED:-0}"
export FIE_REGIME_FILTER="${FIE_REGIME_FILTER:-0}"
export FIE_EDGE_SIMPLE_SIZE="${FIE_EDGE_SIMPLE_SIZE:-0}"
export FIE_SIGNAL_EXPERIMENT="${FIE_SIGNAL_EXPERIMENT:-1}"
export FIE_MARKET_MAP_FILTER="${FIE_MARKET_MAP_FILTER:-0}"
# Инверсия BUY_YES↔BUY_NO (проверка знака)
export FIE_INVERT_SIGNAL="${FIE_INVERT_SIGNAL:-0}"
# Noise больше не используем (диагностика закончена)
export FIE_PROB_NOISE_STD=0
# Эксперимент "edge без sizing": sn_mult=0 и фиксированный размер 0.10
export FIE_EDGE_SN_MULT="${FIE_EDGE_SN_MULT:-0}"
export FIE_ROLLBACK_MIN_SIZE="${FIE_ROLLBACK_MIN_SIZE:-0.10}"
export FIE_ROLLBACK_MAX_SIZE="${FIE_ROLLBACK_MAX_SIZE:-0.10}"
# Временно выкидываем два худших rk (ядро убытка)
export FIE_SKIP_RK_KEYS="${FIE_SKIP_RK_KEYS:-7|mid|off,9|mid|off}"
export FIE_CORR_DEFENSE="${FIE_CORR_DEFENSE:-0}"
export FIE_CORR_THRESHOLD="${FIE_CORR_THRESHOLD:-0}"
export FIE_CORR_DEFENSE_MULT="${FIE_CORR_DEFENSE_MULT:-0.5}"

# Прочее (переопределяется из окружения при необходимости)
export FIE_ARC_STREAK_SOFT="${FIE_ARC_STREAK_SOFT:-3}"
export FIE_ARC_STREAK_HARD="${FIE_ARC_STREAK_HARD:-5}"
export FIE_SIZE_SOFT_CAP="${FIE_SIZE_SOFT_CAP:-0.25}"
export FIE_RK_WR_SOFT="${FIE_RK_WR_SOFT:-1}"
export FIE_HARD_LOSS_CAP_FRAC="${FIE_HARD_LOSS_CAP_FRAC:-0.8}"
export FIE_B_RK_MIN="${FIE_B_RK_MIN:-0.7}"
export FIE_PAYOFF_MIN_B="${FIE_PAYOFF_MIN_B:-1.0}"
export FIE_EDGE_PM_MIN="${FIE_EDGE_PM_MIN:-0.10}"
export FIE_EDGE_THRESHOLD="${FIE_EDGE_THRESHOLD:-0.10}"
export FIE_EDGE_DIRECTION_FROM_REAL="${FIE_EDGE_DIRECTION_FROM_REAL:-1}"
export FIE_SAFE_SIZE="${FIE_SAFE_SIZE:-0.08}"
export FIE_TRADE_COOLDOWN_SEC="${FIE_TRADE_COOLDOWN_SEC:-15}"
export FIE_MARKET_REFRESH_SEC="${FIE_MARKET_REFRESH_SEC:-15}"
export FIE_ALLOW_HIGH_VOL="${FIE_ALLOW_HIGH_VOL:-1}"
export FIE_EDGE_SOURCE="${FIE_EDGE_SOURCE:-market}"
# Локальный режимный фильтр (лучше не фиксировать hour; по умолчанию выключен)
export FIE_ONLY_RK="${FIE_ONLY_RK:-}"
# Hour-weight routing: scale size by learned hour profile inside (mid|on, p in [0.25,0.40))
export FIE_HOUR_WEIGHT_ENABLED="${FIE_HOUR_WEIGHT_ENABLED:-1}"
export FIE_HOUR_WEIGHT_LOOKBACK="${FIE_HOUR_WEIGHT_LOOKBACK:-4000}"
export FIE_HOUR_WEIGHT_MIN_N="${FIE_HOUR_WEIGHT_MIN_N:-20}"
export FIE_HOUR_WEIGHT_MIN_N_HOUR="${FIE_HOUR_WEIGHT_MIN_N_HOUR:-20}"
export FIE_HOUR_WEIGHT_REFRESH_SEC="${FIE_HOUR_WEIGHT_REFRESH_SEC:-600}"
export FIE_HOUR_WEIGHT_FLOOR="${FIE_HOUR_WEIGHT_FLOOR:-0.2}"
export FIE_HOUR_WEIGHT_PRIOR="${FIE_HOUR_WEIGHT_PRIOR:-0.2}"

# Entropy guard: защита от торговли на квантованном p_model.
# Сейчас для запуска "чистого эксперимента" допускаем более ранний прогрев.
export FIE_ENTROPY_GUARD="${FIE_ENTROPY_GUARD:-1}"
export FIE_ENTROPY_WINDOW="${FIE_ENTROPY_WINDOW:-60}"
export FIE_ENTROPY_MIN_UNIQUE="${FIE_ENTROPY_MIN_UNIQUE:-2}"

# Risk-control: вырезать доказанно убыточный режим по паре (liq_spike, funding_stress_48)
# из диагностики post-cut (см. логи [lf-skip]).
export FIE_LF_SKIP_ENABLED="${FIE_LF_SKIP_ENABLED:-1}"
export FIE_SKIP_LF_PAIRS="${FIE_SKIP_LF_PAIRS:-0.653:0.129}"

# Market gate: торгуем только когда рынок ACTIVE и режим "держится" достаточно долго,
# и амплитуда режима (active_peak=max(liq_range,fund_range)) не ниже порога.
export FIE_MARKET_GATE_ENABLED="${FIE_MARKET_GATE_ENABLED:-1}"
export FIE_MARKET_GATE_WARMUP_MIN="${FIE_MARKET_GATE_WARMUP_MIN:-3}"
export FIE_MARKET_GATE_PEAK_MIN="${FIE_MARKET_GATE_PEAK_MIN:-0.05}"
export FIE_MARKET_GATE_TRANSITION_LOG="${FIE_MARKET_GATE_TRANSITION_LOG:-1}"

find "$ROOT/services" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "=== start_prod_loop.sh boot $(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$ ===" >> /tmp/fie_loop.log
# ВАЖНО: не менять редирект — tail -f /tmp/fie_loop.log смотрит сюда.
exec caffeinate -i python3 -m services.prod_loop >> /tmp/fie_loop.log 2>&1

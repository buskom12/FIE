from __future__ import annotations

import sys
import time
import os
import math
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from sqlalchemy import func
from sqlmodel import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.prod.crud import (
    get_recent_pnls, get_recent_size_pnl, get_trades, load_regime_stats, save_snapshot,
    save_trade, upsert_regime_stat, upsert_signal_latest,
)
from db.prod.engine import get_engine, get_session, init_db
from db.prod.models import PortfolioSnapshot, SignalLatest, Trade
from execution.paper_broker import PaperBroker
from execution.polymarket_adapter import get_market_price, signal_to_action
from execution.kelly import compute_kelly_size, estimate_variance
from execution.sizing import apply_dd_risk_scaling
from services.live_alerts import (
    DynamicTailEventMonitor,
    alert_tail_anomalies_smart,
    check_live_alerts,
    log_alerts,
    slack_webhook_sender,
)
from markets.market_engine import EdgeZone, compute_edge_zone
from strategy_c_filter import strategy_c_score
from data.collectors.market import get_btc_ticker_price

# Последние realized PnL по (strategy, regime_key) для fast-edge brake (in-memory).
_FAST_EDGE_HISTORY: dict[tuple[str, tuple], deque] = {}

# Entropy guard cache: если p_model "залип" в одно значение — не торгуем шум.
_P_ENTROPY_CACHE: dict[str, deque] = {}

# Auto edge-level whitelist cache (in-memory, per process).
_EDGE_AUTO_CACHE: dict[str, object] = {
    "last_refresh_ts": 0.0,
    "last_closed_count": 0,
    "levels": set(),
    "best_avg_pnl": None,
    "last_log_ts": 0.0,
}

# Corr log cache (in-memory, per process).
_CORR_CACHE: dict[str, float] = {"last_log_ts": 0.0}

# Hour-weight cache (in-memory, per process).
_HOUR_WEIGHT_CACHE: dict[str, object] = {
    "last_refresh_ts": 0.0,
    "weights": None,  # dict[int,float]
    "counts": None,   # dict[int,int] number of samples per hour (in training slice)
    "last_log_ts": 0.0,
}

# Adaptive Risk Cap: подряд убытков по (strategy, regime_key), не глобальный streak.
local_loss_streaks: dict[tuple[str, tuple], int] = {}
_arc_init_logged = False

# Polymarket: slug или gamma id рынка по bucket rk (час|var|scen)
MARKET_MAP: dict[tuple, str] = {
    (13, "mid", "on"): "btc-up-today",
    (16, "mid", "on"): "btc-down-today",
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _adjust_timeout_hold_steps(hold_steps: int, gate_score_raw: object) -> int:
    """Experimental timeout expansion for mid gate scores."""
    multiplier = float(os.environ.get("FIE_EXIT_TIMEOUT_MULTIPLIER", "1.0"))
    if multiplier <= 1.0:
        return hold_steps

    try:
        gate_score = float(gate_score_raw)
    except Exception:
        return hold_steps

    gate_min = float(os.environ.get("FIE_EXIT_TIMEOUT_GATE_MIN", "0.1"))
    gate_max = float(os.environ.get("FIE_EXIT_TIMEOUT_GATE_MAX", "0.3"))
    if gate_min <= gate_score < gate_max:
        return max(1, int(round(hold_steps * multiplier)))
    return hold_steps


def _parse_lf_skip_pairs(raw: str) -> list[tuple[float, float]]:
    """
    Формат: "l1:f1,l2:f2" (запятая — разделитель пар).
    Пример (диагностический режим-дренаж):
      FIE_SKIP_LF_PAIRS="0.653:0.129"
    """
    out: list[tuple[float, float]] = []
    s = (raw or "").strip()
    if not s:
        return out
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        if ":" not in p:
            continue
        a, b = p.split(":", 1)
        try:
            out.append((float(a.strip()), float(b.strip())))
        except Exception:
            continue
    return out


def _lf_skip_trade(*, liq: float | None, funding: float | None) -> bool:
    """
    Жёсткий risk-control skip по паре (liq_spike, funding_stress_48) из payload/EdgeZone.

    Важно: это не "тюнинг модели", а удаление доказанно убыточного режима рынка,
    когда внутри него нет рычагов (например momentum константа).
    """
    if not _env_bool("FIE_LF_SKIP_ENABLED", False):
        return False
    if liq is None or funding is None:
        return False
    try:
        l = float(liq)
        f = float(funding)
    except Exception:
        return False

    pairs = _parse_lf_skip_pairs(os.environ.get("FIE_SKIP_LF_PAIRS", ""))
    if not pairs:
        return False

    l_dec = int(os.environ.get("FIE_LF_SKIP_ROUND_LIQ", "3"))
    f_dec = int(os.environ.get("FIE_LF_SKIP_ROUND_FUND", "3"))
    l_eps = float(os.environ.get("FIE_LF_SKIP_EPS_LIQ", "0.0005"))
    f_eps = float(os.environ.get("FIE_LF_SKIP_EPS_FUND", "0.0005"))

    lr = round(l, l_dec)
    fr = round(f, f_dec)
    for lt, ft in pairs:
        if abs(lr - round(float(lt), l_dec)) <= l_eps + 1e-12 and abs(
            fr - round(float(ft), f_dec)
        ) <= f_eps + 1e-12:
            return True
    return False


# ── Market-state detector (independent of trades) ─────────────────────────────
# Окно последних EdgeZone состояний: (liq_spike, funding_stress_48, momentum_up)
_MARKET_STATE_WIN: deque[tuple[float, float, float]] = deque(maxlen=120)
_MARKET_STATE_STATUS: str | None = None
_MARKET_STATE_SINCE_TS: float | None = None


def _update_market_state(z: EdgeZone) -> dict[str, float | str]:
    """
    Обновляет детектор состояния рынка на основе EdgeZone (не зависит от входов/сделок).

    Возвращает dict:
      - status: MARKET_SLEEP | MARKET_NOISE | MARKET_ACTIVE
      - active_min: сколько минут текущий статус держится
      - active_peak: max(liq_range, fund_range) внутри окна
      - liq_range / fund_range: амплитуды внутри окна
      - uniq_states / uniq_liq / uniq_fund / uniq_m: энтропия окна
    """
    global _MARKET_STATE_STATUS, _MARKET_STATE_SINCE_TS

    l = round(float(z.liq_spike), 3)
    f = round(float(z.funding_stress), 3)
    m = round(float(z.momentum_up), 3)
    _MARKET_STATE_WIN.append((l, f, m))

    uniq_states = len(set(_MARKET_STATE_WIN))
    uniq_liq = len({s[0] for s in _MARKET_STATE_WIN})
    uniq_fund = len({s[1] for s in _MARKET_STATE_WIN})
    uniq_m = len({s[2] for s in _MARKET_STATE_WIN})

    liq_vals = [s[0] for s in _MARKET_STATE_WIN]
    fund_vals = [s[1] for s in _MARKET_STATE_WIN]
    liq_range = (max(liq_vals) - min(liq_vals)) if liq_vals else 0.0
    fund_range = (max(fund_vals) - min(fund_vals)) if fund_vals else 0.0
    active_peak = max(liq_range, fund_range)

    # Classifier (жёсткие правила без "на глаз")
    if uniq_states == 1 or (uniq_liq == 1 and uniq_fund == 1):
        status = "MARKET_SLEEP"
    elif uniq_states >= 2 and liq_range < 0.05 and fund_range < 0.05:
        status = "MARKET_NOISE"
    elif (uniq_liq >= 2 or uniq_fund >= 2) and (liq_range >= 0.10 or fund_range >= 0.05):
        status = "MARKET_ACTIVE"
    else:
        status = "MARKET_NOISE"

    now_ts = time.monotonic()
    if _MARKET_STATE_STATUS is None or status != _MARKET_STATE_STATUS:
        # Лог перехода включаем только по флагу, чтобы не спамить прод
        if _env_bool("FIE_MARKET_GATE_TRANSITION_LOG", True):
            prev = _MARKET_STATE_STATUS
            prev_dur = 0.0
            if _MARKET_STATE_SINCE_TS is not None:
                prev_dur = (now_ts - float(_MARKET_STATE_SINCE_TS)) / 60.0
            print(
                f"[edge-watch] TRANSITION {prev}->{status} dur_prev_min={prev_dur:.1f} "
                f"state(l,f,m)={(l, f, m)} peak={active_peak:.3f} "
                f"ranges(liq,fund)=({liq_range:.3f},{fund_range:.3f}) "
                f"uniq(liq,fund,m,states)=({uniq_liq},{uniq_fund},{uniq_m},{uniq_states})",
                flush=True,
            )
        _MARKET_STATE_STATUS = status
        _MARKET_STATE_SINCE_TS = now_ts

    active_min = 0.0
    if _MARKET_STATE_SINCE_TS is not None:
        active_min = (now_ts - float(_MARKET_STATE_SINCE_TS)) / 60.0

    return {
        "status": status,
        "active_min": float(active_min),
        "active_peak": float(active_peak),
        "liq_range": float(liq_range),
        "fund_range": float(fund_range),
        "uniq_states": float(uniq_states),
        "uniq_liq": float(uniq_liq),
        "uniq_fund": float(uniq_fund),
        "uniq_m": float(uniq_m),
    }


def _market_gate_should_skip(meta: dict[str, object]) -> tuple[bool, str]:
    """
    Production-гейт:
      IF status != MARKET_ACTIVE → skip
      IF active_min < warmup → skip
      IF active_peak < peak_thr → skip

    Все пороги управляются env, по умолчанию включается из start_prod_loop.sh.
    """
    if not _env_bool("FIE_MARKET_GATE_ENABLED", False):
        return False, ""

    status = str(meta.get("market_status") or "")
    try:
        active_min = float(meta.get("market_active_min") or 0.0)
        active_peak = float(meta.get("market_active_peak") or 0.0)
    except Exception:
        active_min, active_peak = 0.0, 0.0

    warmup_min = float(os.environ.get("FIE_MARKET_GATE_WARMUP_MIN", "3"))
    peak_thr = float(os.environ.get("FIE_MARKET_GATE_PEAK_MIN", "0.05"))

    if status != "MARKET_ACTIVE":
        return True, f"status={status or 'UNKNOWN'}"
    if active_min < warmup_min:
        return True, f"warmup {active_min:.2f}<{warmup_min}"
    if active_peak < peak_thr:
        return True, f"peak {active_peak:.3f}<{peak_thr}"
    return False, ""

def _auto_allowed_edge_levels(session) -> set[float]:
    """
    Авто-whitelist уровней edge_score по rolling-окну закрытых сделок.

    Принцип:
    - берём последние lookback закрытых сделок
    - группируем по round(edge_score, 6)
    - оставляем уровни с n >= min_n и avg_pnl > ev_eps
    - берём top_k по avg_pnl
    - если кандидатов нет → возвращаем пустое множество (значит "пауза торговли")

    Стабилизация:
    - обновляем не чаще refresh_sec
    - или если появились +refresh_closed_delta новых закрытий
    - не прыгаем на новый whitelist, если он не пересекается со старым И не даёт явного улучшения
    """
    # Opt-out: если авто-режим выключен — пусть вызывающий код решает.
    if not _env_bool("FIE_ALLOWED_EDGE_AUTO", True):
        return set()

    lookback = int(os.environ.get("FIE_ALLOWED_EDGE_LOOKBACK", "400"))
    min_n = int(os.environ.get("FIE_ALLOWED_EDGE_MIN_N", "20"))
    top_k = int(os.environ.get("FIE_ALLOWED_EDGE_TOP_K", "2"))
    ev_eps = float(os.environ.get("FIE_ALLOWED_EDGE_EV_EPS", "0.000005"))

    refresh_sec = float(os.environ.get("FIE_ALLOWED_EDGE_REFRESH_SEC", "600"))
    refresh_closed_delta = int(os.environ.get("FIE_ALLOWED_EDGE_REFRESH_CLOSED_DELTA", "50"))
    stabilize_min_improve = float(os.environ.get("FIE_ALLOWED_EDGE_STABILIZE_IMPROVE", "0.000002"))

    now_ts = time.monotonic()
    last_refresh_ts = float(_EDGE_AUTO_CACHE.get("last_refresh_ts", 0.0) or 0.0)
    last_closed_count = int(_EDGE_AUTO_CACHE.get("last_closed_count", 0) or 0)
    prev_levels = set(_EDGE_AUTO_CACHE.get("levels", set()) or set())
    prev_best = _EDGE_AUTO_CACHE.get("best_avg_pnl", None)
    prev_best = float(prev_best) if prev_best is not None else None

    # Сколько закрытых сейчас (для refresh по дельте).
    from sqlmodel import select
    closed_count = int(
        session.exec(
            select(func.count()).select_from(Trade).where(Trade.pnl.is_not(None))
        ).one()
        or 0
    )

    need_refresh = False
    if now_ts - last_refresh_ts >= refresh_sec:
        need_refresh = True
    if (closed_count - last_closed_count) >= refresh_closed_delta:
        need_refresh = True

    if not need_refresh and prev_levels:
        return prev_levels

    # Собираем last N закрытых (маленький N — проще посчитать в Python без сложного SQL).
    from sqlmodel import desc

    rows = list(
        session.exec(
            select(Trade.edge_score, Trade.pnl)
            .where(Trade.pnl.is_not(None))
            .order_by(desc(Trade.id))
            .limit(int(lookback))
        ).all()
    )
    buckets: dict[float, list[float]] = {}
    for edge_score, pnl in rows:
        if edge_score is None or pnl is None:
            continue
        try:
            e = round(float(edge_score), 6)
            p = float(pnl)
        except Exception:
            continue
        buckets.setdefault(e, []).append(p)

    stats: list[tuple[float, int, float]] = []
    for e, pnls in buckets.items():
        n = len(pnls)
        if n <= 0:
            continue
        avg = sum(pnls) / n
        stats.append((float(e), int(n), float(avg)))

    # Кандидаты: только где хватает данных и есть "сила EV" выше eps.
    candidates = [(e, n, avg) for (e, n, avg) in stats if n >= min_n and avg > ev_eps]
    candidates.sort(key=lambda x: x[2], reverse=True)
    new_levels = {float(e) for (e, _, _) in candidates[: max(top_k, 0)]}
    new_best = float(candidates[0][2]) if candidates else None

    # Стабилизатор перескакивания режимов:
    # если новый список не пересекается со старым, применяем только при явном улучшении best avg_pnl.
    if prev_levels and new_levels and not (prev_levels & new_levels):
        if prev_best is None or new_best is None or (new_best <= prev_best + stabilize_min_improve):
            new_levels = prev_levels
            new_best = prev_best

    _EDGE_AUTO_CACHE["last_refresh_ts"] = now_ts
    _EDGE_AUTO_CACHE["last_closed_count"] = closed_count
    _EDGE_AUTO_CACHE["levels"] = new_levels
    _EDGE_AUTO_CACHE["best_avg_pnl"] = new_best

    # Логируем редко, чтобы не спамить.
    last_log_ts = float(_EDGE_AUTO_CACHE.get("last_log_ts", 0.0) or 0.0)
    if now_ts - last_log_ts >= max(30.0, refresh_sec / 4.0):
        if not new_levels:
            print(
                f"[EDGE AUTO] no positive EV levels (lookback={lookback} min_n={min_n} "
                f"top_k={top_k} ev_eps={ev_eps:g}) → trading paused",
                flush=True,
            )
        else:
            print(
                f"[EDGE AUTO] allowed_levels={sorted(new_levels)} best_avg_pnl={new_best:+.8f} "
                f"(lookback={lookback} min_n={min_n} top_k={top_k} ev_eps={ev_eps:g})",
                flush=True,
            )
        _EDGE_AUTO_CACHE["last_log_ts"] = now_ts

    return new_levels


def _sigmoid(x: float) -> float:
    x = _clamp(x, -20.0, 20.0)
    return 1.0 / (1.0 + math.exp(-x))


def score_to_prob(score: float) -> float:
    """Сжимает сырой score в вероятность (0..1); быстрый sigmoid по сигналу до size."""
    z = _clamp(float(score) * 5.0, -40.0, 40.0)
    return 1.0 / (1.0 + math.exp(-z))


def _log_recent_signal_corr(session) -> None:
    """
    Быстрая sanity-проверка: корреляции сигналов с realized pnl по последним закрытым сделкам.
    Не должна ломать цикл: любые ошибки глотаем.
    """
    try:
        every_sec = float(os.environ.get("FIE_CORR_LOG_SEC", "600"))
        lookback = int(os.environ.get("FIE_CORR_LOOKBACK", "400"))
        now_ts = time.monotonic()
        last_ts = float(_CORR_CACHE.get("last_log_ts", 0.0) or 0.0)
        if now_ts - last_ts < max(10.0, every_sec):
            return

        from sqlmodel import desc, select

        rows = list(
            session.exec(
                select(
                    Trade.pnl,
                    Trade.p_model,
                    Trade.p_market,
                    Trade.edge_real,
                    Trade.edge_score,
                    Trade.edge,
                )
                .where(Trade.pnl.is_not(None))
                .order_by(desc(Trade.id))
                .limit(int(lookback))
            ).all()
        )
        if not rows:
            return

        df = pd.DataFrame(
            rows,
            columns=["pnl", "p_model", "p_market", "edge_real", "edge_score", "edge"],
        )

        def _corr(a: str, b: str) -> float | None:
            x = df[[a, b]].dropna()
            if len(x) < 20:
                return None
            v = float(x[a].corr(x[b]))
            if math.isnan(v):
                return None
            return v

        c_em = _corr("edge_real", "pnl")
        c_es = _corr("edge_score", "pnl")
        c_pm = _corr("p_model", "pnl")
        c_e = _corr("edge", "pnl")

        print(
            "[corr] "
            f"n={len(df.dropna(subset=['pnl']))} "
            f"corr(edge_real,pnl)={c_em if c_em is not None else 'n/a'} "
            f"corr(edge_score,pnl)={c_es if c_es is not None else 'n/a'} "
            f"corr(p_model,pnl)={c_pm if c_pm is not None else 'n/a'} "
            f"corr(edge,pnl)={c_e if c_e is not None else 'n/a'}",
            flush=True,
        )

        _CORR_CACHE["last_log_ts"] = now_ts
    except Exception:
        return


def _compute_conf_and_scale(
    cur_rs: dict,
    fast_exp: Optional[float],
    fe_hist: Optional[deque],
    *,
    conf_fast_k: float,
    consistency_n: int,
    low_trades_cap: int,
) -> tuple[float, float]:
    """
    conf ∈ [0.5, 1.5] (как раньше: SNR × log trades × fast × consistency).
    conf_scale — ступени по порогам FIE_CONF_LOW/MID/HIGH и множителям FIE_CONF_SCALE_*.
    При conf ≥ HIGH: буст только если edge=ema_exp/ema_var > 0; иначе слабый множитель (0.9).
    Верх буста ограничен FIE_CONF_BOOST_MAX (дефолт 1.1).
    low_trades_cap оставлен в сигнатуре для совместимости; на scale не влияет.
    """
    _ = low_trades_cap  # совместимость вызова
    ema_exp = cur_rs.get("ema_exp")
    if ema_exp is None:
        return 1.0, 1.0
    ema_var = float(cur_rs.get("ema_var") or 0.0)
    snr = abs(float(ema_exp)) / max(ema_var, 1e-9)
    trades = int(cur_rs.get("trades", 0))
    conf = _sigmoid(snr) * (math.log1p(trades) / math.log1p(50.0))
    if fast_exp is not None:
        conf *= _clamp(1.0 + float(fast_exp) / max(conf_fast_k, 1e-12), 0.8, 1.2)
    wr = 0.5
    if fe_hist is not None and len(fe_hist) >= 2:
        recent = list(fe_hist)[-min(int(consistency_n), len(fe_hist)) :]
        wr = sum(1.0 for p in recent if float(p) > 0.0) / max(len(recent), 1)
    consistency = 1.0 - abs(wr - 0.5) * 2.0
    consistency = _clamp(consistency, 0.0, 1.0)
    conf *= consistency
    # ВАЖНО: не зажимать снизу до 0.5 — иначе conf часто “залипает” на 0.50 и scale всегда ~0.85.
    conf = _clamp(conf, 0.0, 1.5)
    low = float(os.environ.get("FIE_CONF_LOW", "0.5"))
    mid = float(os.environ.get("FIE_CONF_MID", "0.65"))
    high = float(os.environ.get("FIE_CONF_HIGH", "0.8"))
    s_low = float(os.environ.get("FIE_CONF_SCALE_LOW", "0.7"))
    s_mid = float(os.environ.get("FIE_CONF_SCALE_MID", "0.85"))
    s_neutral = float(os.environ.get("FIE_CONF_SCALE_NEUTRAL", "1.0"))
    s_high = float(os.environ.get("FIE_CONF_SCALE_HIGH", "1.15"))
    boost_max = float(os.environ.get("FIE_CONF_BOOST_MAX", "1.1"))
    s_weak_high = float(os.environ.get("FIE_CONF_SCALE_WEAK_EDGE", "0.9"))
    if conf < low:
        conf_scale = s_low
    elif conf < mid:
        conf_scale = s_mid
    elif conf < high:
        conf_scale = s_neutral
    else:
        # SNR-style edge (как в edge-pre): только положительный ema → буст
        regime_edge = float(ema_exp) / max(ema_var, 1e-9)
        if regime_edge > 0:
            conf_scale = min(s_high, boost_max)
        else:
            conf_scale = s_weak_high
    return conf, conf_scale


def _center01(x: float) -> float:
    return float(x) - 0.5


def _strategy_b_raw_logit(z: EdgeZone) -> float:
    """
    Непрерывный логит из фич EdgeZone (все в [0,1], кроме gate_score — тоже обычно ~[0,1]).
    Центрирование вокруг 0.5 даёт и отрицательный, и положительный вклад → P может быть <0.5.
    """
    c = _center01
    return (
        c(z.scen_accumulation) * 1.6
        - c(z.momentum_up) * 1.2
        + c(z.liq_spike) * 0.6
        + c(z.breakout_up) * 0.8
        + c(z.scen_trend_flow) * 0.5
        + c(z.funding_stress) * 0.4
        + c(z.oi_strength) * 0.4
        - c(z.scen_breakout_suspicious) * 0.35
    )


def _strategy_b_probability(z: EdgeZone) -> tuple[float, float]:
    """
    P(YES) для стратегии B: сжатый логит → sigmoid.
    Возвращает (raw_logit, probability).
    """
    raw = _clamp(_strategy_b_raw_logit(z), -4.0, 4.0)
    zz = _clamp(raw * 2.0, -40.0, 40.0)
    p = 1.0 / (1.0 + math.exp(-zz))
    return raw, _clamp(p, 1e-6, 1.0 - 1e-6)


def compute_probability(payload: dict) -> float | None:
    """
    Возвращает "настоящую" вероятность для decision layer.

    Важно: НЕ используем fallback на confidence и НЕ подставляем 0.5 по умолчанию.
    Если probability отсутствует/некорректен — сигнал считается отсутствующим.
    """
    p = payload.get("probability")
    if p is None:
        return None
    try:
        return float(p)
    except Exception:
        return None


def _parse_rk_key(rk_db: str | None) -> tuple[int | None, str | None, str | None]:
    """
    regime_key в БД: \"hour|vol|on/off\".
    Возвращает (hour, vol_bucket, scen_bucket) или (None, None, None).
    """
    if not rk_db:
        return None, None, None
    parts = str(rk_db).split("|")
    if len(parts) != 3:
        return None, None, None
    try:
        h = int(parts[0])
    except Exception:
        h = None
    return h, parts[1], parts[2]


def _hour_weights_from_trades(rows: list[tuple[int, float]]) -> dict[int, float]:
    """
    rows: (hour, pnl) for the target slice (e.g. mid|on + p-zone).
    Returns hour->weight in [0..1] after centering + tanh + floor.
    """
    if not rows:
        return {}
    by_h: dict[int, list[float]] = {}
    for h, pnl in rows:
        if h is None:
            continue
        by_h.setdefault(int(h), []).append(float(pnl))
    if not by_h:
        return {}
    # avg pnl per hour
    hp: dict[int, float] = {h: (sum(v) / len(v)) for h, v in by_h.items() if v}
    if not hp:
        return {}
    vals = list(hp.values())
    mu = sum(vals) / len(vals)
    centered = {h: (v - mu) for h, v in hp.items()}
    # robust-ish scale
    var = sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    denom = sd + 1e-9
    out: dict[int, float] = {}
    for h, v in centered.items():
        t = math.tanh(float(v) / denom)  # [-1,1]
        out[int(h)] = float((t + 1.0) / 2.0)  # -> [0,1]
    return out


def _get_hour_weight(session, *, hour: int, prob: float) -> float:
    """
    Continuous allocation by hour profile, learned from recent trades.
    Profile is computed on slice: vol_bucket=mid, scen=on, p in [0.25,0.40).
    """
    if not _env_bool("FIE_HOUR_WEIGHT_ENABLED", True):
        return 1.0
    # Apply only in the target p-zone; outside zone we don't scale (decision already HOLD/BUY via zones).
    if not (0.25 <= float(prob) < 0.40):
        return 1.0

    lookback = int(os.environ.get("FIE_HOUR_WEIGHT_LOOKBACK", "4000"))
    min_n = int(os.environ.get("FIE_HOUR_WEIGHT_MIN_N", "20"))
    min_n_hour = int(os.environ.get("FIE_HOUR_WEIGHT_MIN_N_HOUR", str(min_n)))
    refresh_sec = float(os.environ.get("FIE_HOUR_WEIGHT_REFRESH_SEC", "600"))
    floor = float(os.environ.get("FIE_HOUR_WEIGHT_FLOOR", "0.2"))
    floor = _clamp(floor, 0.0, 1.0)
    prior = float(os.environ.get("FIE_HOUR_WEIGHT_PRIOR", "0.2"))
    prior = _clamp(prior, 0.0, 1.0)

    now_ts = time.monotonic()
    last_refresh_ts = float(_HOUR_WEIGHT_CACHE.get("last_refresh_ts", 0.0) or 0.0)
    weights = _HOUR_WEIGHT_CACHE.get("weights")
    weights = weights if isinstance(weights, dict) else None

    need_refresh = (weights is None) or (now_ts - last_refresh_ts >= max(10.0, refresh_sec))
    if need_refresh:
        try:
            from sqlmodel import desc, select
            # pull recent closed trades where we have p_model and pnl (same slice as analysis)
            rows = list(
                session.exec(
                    select(Trade.regime_key, Trade.pnl, Trade.p_model)
                    .where(Trade.pnl.is_not(None))
                    .where(Trade.p_model.is_not(None))
                    .order_by(desc(Trade.id))
                    .limit(int(lookback))
                ).all()
            )
            # filter to vol=mid, scen=on, p-zone 0.25..0.40
            hp_rows: list[tuple[int, float]] = []
            hour_counts: dict[int, int] = {}
            for rk_db, pnl, pm in rows:
                if rk_db is None or pnl is None or pm is None:
                    continue
                h, vol, scen = _parse_rk_key(str(rk_db))
                if h is None or vol != "mid" or scen != "on":
                    continue
                p = float(pm)
                if 0.25 <= p < 0.40:
                    hp_rows.append((int(h), float(pnl)))
                    hour_counts[int(h)] = hour_counts.get(int(h), 0) + 1
            # enforce min_n per-hour later via fallback; but also require at least min_n total
            if len(hp_rows) >= min_n:
                w0 = _hour_weights_from_trades(hp_rows)
                # apply floor (don't fully shut off hours)
                w = {h: float(floor + (1.0 - floor) * ww) for h, ww in w0.items()}
                _HOUR_WEIGHT_CACHE["weights"] = w
                _HOUR_WEIGHT_CACHE["counts"] = hour_counts
                _HOUR_WEIGHT_CACHE["last_refresh_ts"] = now_ts

                last_log_ts = float(_HOUR_WEIGHT_CACHE.get("last_log_ts", 0.0) or 0.0)
                if now_ts - last_log_ts >= max(30.0, refresh_sec / 4.0):
                    # compact log: show top/bottom hours by weight
                    items = sorted(w.items(), key=lambda kv: kv[1])
                    lo = items[:3]
                    hi = items[-3:] if len(items) >= 3 else items
                    print(
                        f"[hour-weight] learned hours={len(items)} floor={floor:.2f} "
                        f"lo={[(h, round(v,3)) for h,v in lo]} hi={[(h, round(v,3)) for h,v in hi]}",
                        flush=True,
                    )
                    _HOUR_WEIGHT_CACHE["last_log_ts"] = now_ts
            else:
                # not enough data to learn; keep previous weights (or None)
                _HOUR_WEIGHT_CACHE["last_refresh_ts"] = now_ts
        except Exception:
            _HOUR_WEIGHT_CACHE["last_refresh_ts"] = now_ts

    weights = _HOUR_WEIGHT_CACHE.get("weights")
    weights = weights if isinstance(weights, dict) else None
    counts = _HOUR_WEIGHT_CACHE.get("counts")
    counts = counts if isinstance(counts, dict) else None
    if not weights or not counts:
        return 1.0

    h = int(hour)
    n_h = int(counts.get(h, 0))
    # Если по часу мало наблюдений — не интерполируем; используем prior (жёстче, чем 0.5).
    if n_h < min_n_hour:
        return float(prior)

    return float(weights.get(h, prior))


# Кэш EdgeZone: обновляем рынок не чаще раза в MARKET_REFRESH_SEC секунд
_zone_cache: dict = {"zone": None, "last_ts": 0.0}
_MARKET_REFRESH_SEC = float(os.environ.get("FIE_MARKET_REFRESH_SEC", "60"))

# Implied prob Polymarket (первый рынок из gamma API) — для edge vs модель
_p_market_cache: dict = {"p": None, "last_ts": 0.0}

# Реальная BTC цена: нормируется относительно первой известной (сессионный baseline)
_btc_baseline: dict = {"price": None}


def compute_signals() -> dict:
    """
    Реальный пайплайн сигналов:
      1. Получает live BTC OHLCV с Binance (с кэшем _MARKET_REFRESH_SEC)
      2. Вычисляет EdgeZone: gate_score, regime, scen_*, breakout_up, momentum_up
      3. Strategy A — trend-follower: probability = 0.5 + gate_score
         (входит в сторону рынка когда gate_score значительный)
      4. Strategy B — mean-reversion: probability строится на scen_accumulation
         (входит против тренда в зонах накопления)
      5. Применяет strategy_c_score для поля score_C
    На ошибке → возвращает {} → prod_loop выдаёт HOLD
    """
    now_ts = time.monotonic()
    try:
        if now_ts - _zone_cache["last_ts"] >= _MARKET_REFRESH_SEC or _zone_cache["zone"] is None:
            _zone_cache["zone"] = compute_edge_zone()
            _zone_cache["last_ts"] = now_ts
            print(
                f"[signals] zone refreshed: regime={_zone_cache['zone'].regime}/"
                f"{_zone_cache['zone'].volatility} "
                f"gate={_zone_cache['zone'].gate_score:.3f} "
                f"thr={_zone_cache['zone'].gate_threshold:.3f} "
                f"allow={_zone_cache['zone'].allow_prediction} "
                f"reason={_zone_cache['zone'].reason}",
                flush=True,
            )
    except Exception as exc:
        print(f"[signals] compute_edge_zone failed: {exc}", flush=True)
        return {}

    z = _zone_cache["zone"]
    # ── Market-state detector (EdgeZone-only, до сделок/входов) ─────────────
    ms = _update_market_state(z)

    # ── Polymarket P_market (implied probability) — тот же cadence что zone ──
    if (
        now_ts - _p_market_cache["last_ts"] >= _MARKET_REFRESH_SEC
        or _p_market_cache["p"] is None
    ):
        try:
            from markets.polymarket_real import get_predictions

            preds = get_predictions()
            if preds:
                _p_market_cache["p"] = _clamp(float(preds[0]["probability"]), 0.0, 1.0)
            else:
                _p_market_cache["p"] = 0.5
        except Exception as exc:
            print(f"[signals] p_market: {exc}", flush=True)
            if _p_market_cache["p"] is None:
                _p_market_cache["p"] = 0.5
        _p_market_cache["last_ts"] = now_ts
    p_market = float(_p_market_cache["p"] if _p_market_cache["p"] is not None else 0.5)

    # ── Реальная BTC цена → price proxy для PnL расчёта ───────────────────
    # BTC цена нормируется относительно сессионного baseline:
    #   price_proxy = 0.5 + (btc_now - btc_baseline) / btc_baseline
    # PnL = (exit_proxy - entry_proxy) * side * notional = % return * notional
    btc_now = get_btc_ticker_price()
    if btc_now is not None:
        if _btc_baseline["price"] is None:
            _btc_baseline["price"] = btc_now
        btc_ref = _btc_baseline["price"]
        btc_price_proxy = _clamp(0.5 + (btc_now - btc_ref) / btc_ref, 0.0, 1.0)
    else:
        btc_price_proxy = 0.5

    # ── Strategy A: trend-follower ──────────────────────────────────────────
    # gate_score > 0  → bullish → BUY_YES (prob > 0.55)
    # gate_score < 0  → bearish → BUY_NO  (prob < 0.45)
    # gate_score ≈ 0  → HOLD
    prob_a = _clamp(0.5 + z.gate_score, 0.0, 1.0)
    conf_a = _clamp(abs(z.gate_score) * 2.0, 0.0, 1.0)
    d_score_a = _clamp(z.gate_score - z.gate_threshold, -1.0, 1.0)

    trade_a = {
        "scenario": {
            "momentum_up": z.momentum_up,
            "breakout_up": z.breakout_up,
        },
        "confidence": conf_a,
    }
    score_c_a = strategy_c_score(trade_a)

    payload_a = {
        "probability":       prob_a,
        "price":             btc_price_proxy,   # реальная BTC цена → для PnL
        "btc_usd":           btc_now,
        "confidence":        conf_a,
        "alpha_multiplier":  1.0 + _clamp(z.oi_strength * 0.5, 0.0, 0.5),
        "gate_score":        round(z.gate_score, 6),
        "threshold":         0.55,
        "regime":            z.regime,
        "volatility":        z.volatility,
        "scen_accumulation": z.scen_accumulation,
        "breakout_up":       z.breakout_up,
        "score_C":           float(score_c_a) / 3.0,
        "d_score":           d_score_a,
        "tail_hit":          bool(z.funding_stress > 0.5 or z.liq_spike > 0.5),
        "var_limit":         200,
        "var_window":        50,
        "variance_floor":    0.05,
        # Market-state meta (для production gate + дебага)
        "market_status":      ms.get("status"),
        "market_active_min":  ms.get("active_min"),
        "market_active_peak": ms.get("active_peak"),
    }

    # ── Strategy B: multi-signal mean-reversion ────────────────────────────
    # Основной сигнал: scen_accumulation (накопление)
    # Вторичные сигналы (активны когда scen=0):
    #   gate_score  → направление рынка (0.15 вес)
    #   liq_spike   → ликвидации → разворот (0.10 вес, centered around 0.5)
    scen_acc = z.scen_accumulation
    momentum = z.momentum_up
    raw_b, prob_b = _strategy_b_probability(z)
    conf_b = _clamp(abs(prob_b - 0.5) * 2.0, 0.0, 1.0)
    d_score_b = _clamp(raw_b / 2.5, -1.0, 1.0)

    trade_b = {
        "scenario": {
            "momentum_up": momentum,
            "breakout_up": z.breakout_up,
        },
        "confidence": conf_b,
    }
    score_c_b = strategy_c_score(trade_b)

    payload_b = {
        "probability":       prob_b,
        "price":             btc_price_proxy,   # реальная BTC цена → для PnL
        "btc_usd":           btc_now,
        "confidence":        conf_b,
        "alpha_multiplier":  1.0,
        "gate_score":        round(z.gate_score, 6),
        "threshold":         0.502,  # 0.55 → 0.51 → 0.505 → 0.502
        "regime":            z.regime,
        "volatility":        z.volatility,
        "scen_accumulation": scen_acc,
        "breakout_up":       z.breakout_up,
        "score_C":           float(score_c_b) / 3.0,
        "d_score":           d_score_b,
        "tail_hit":          bool(z.funding_stress > 0.5 or z.liq_spike > 0.5),
        "var_limit":         200,
        "var_window":        50,
        "variance_floor":    0.05,
        "p_market":          p_market,
        # Диагностика: те же величины, что в EdgeZone (раньше здесь был несуществующий `signals` → NameError).
        "funding_stress_48": float(z.funding_stress),
        "liq_spike":         float(z.liq_spike),
        # держим старый ключ `scen_bo_s` для обратной совместимости,
        # но основным считаем совпадающий с колонкой trades.*
        "scen_breakout_suspicious": float(z.scen_breakout_suspicious),
        "scen_bo_s":                float(z.scen_breakout_suspicious),
        "momentum_up":       float(z.momentum_up),
        "oi_strength":       float(z.oi_strength),
        "model_logit_b":     float(raw_b),
        # Market-state meta (для production gate + дебага)
        "market_status":      ms.get("status"),
        "market_active_min":  ms.get("active_min"),
        "market_active_peak": ms.get("active_peak"),
    }

    if _env_bool("FIE_SIGNAL_AUDIT", False):
        print(
            f"[signal-audit] raw_b={raw_b:+.6f} prob_b={prob_b:.10f} prob_repr={prob_b!r} "
            f"gate={z.gate_score:.5f} scen_acc={scen_acc:.5f} mom={momentum:.5f} "
            f"p_market={p_market:.5f}",
            flush=True,
        )

    # Strategy A отключена — тянет систему вниз в range/high режиме
    return {"B": payload_b}


def run_portfolio_logic(signals: dict) -> dict | None:
    """
    Подключите вашу реальную портфельную логику (allocation + risk-layer + multipliers).
    Должна вернуть dict с ключевыми полями для Trade, либо None если трейда нет.
    """
    return None


# ── Auto-regime learning helpers ─────────────────────────────────────────────

_VBUCKET_EPS = 1e-9  # защита от ошибок плавающей точки при EMA с порогом

def _variance_bucket(v: float) -> str:
    if v < 0.05 - _VBUCKET_EPS:
        return "low"
    elif v < 0.1 - _VBUCKET_EPS:
        return "mid"
    return "high"


def _scen_bucket(s: float) -> str:
    return "on" if s > 0 else "off"


def _regime_key(hour: int, variance: float, scen_acc: float) -> tuple:
    """Ключ режима (кросс-дневной): (hour_utc, variance_bucket, scen_bucket)."""
    return (hour, _variance_bucket(variance), _scen_bucket(scen_acc))


def _apply_regime_time_decay(stats: dict, now_ts: float, decay_lambda: float) -> None:
    """
    Time decay по часам: decay = λ^dt_hours на pnl_sum, trades, wins, ema_exp, ema_var.
    Первое касание без last_update_ts — только ставим метку времени (без затухания).
    """
    lam = _clamp(float(decay_lambda), 0.01, 1.0)
    lu = stats.get("last_update_ts")
    if lu is None:
        stats["last_update_ts"] = now_ts
        return
    dt_hours = (now_ts - float(lu)) / 3600.0
    if dt_hours <= 0:
        return
    decay = lam**dt_hours
    stats["pnl_sum"] = float(stats.get("pnl_sum", 0.0)) * decay
    stats["trades"] = float(stats.get("trades", 0.0)) * decay
    stats["wins"] = float(stats.get("wins", 0.0)) * decay
    stats["ema_var"] = float(stats.get("ema_var", 0.0)) * decay
    if stats.get("ema_exp") is not None:
        stats["ema_exp"] = float(stats["ema_exp"]) * decay
    stats["last_update_ts"] = now_ts


def _pearson_size_pnl(pairs: list[tuple[float, float]]) -> float | None:
    """Pearson r между size и pnl (один ряд пар, время ascending)."""
    n = len(pairs)
    if n < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-18 or vy <= 1e-18:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def main(poll_seconds: int = 60) -> None:
    init_db()
    print("DEBUG_DB_PATH", get_engine().url, flush=True)

    safe_mode = _env_bool("FIE_SAFE_MODE", False)
    # Упрощённый sizing: base * Kelly + только глобальный streak-guard (без ACS/GSM/asym/DD/local-ARC…).
    simple_kelly = _env_bool("FIE_SIMPLE_KELLY_SIZE", True)
    if safe_mode:
        simple_kelly = False
    if safe_mode:
        print("[safe-mode] ON: fixed size, no ACS/ARC/asym/GSM", flush=True)
    if simple_kelly:
        print(
            "[simple-kelly-size] ON: size=base*(0.5+kelly_safe), base=FIE_SIMPLE_BASE_SIZE; "
            "ARC=global streak only; no ACS/GSM/asym/DD/local-ARC/conf-scale",
            flush=True,
        )

    arc_streak_soft = int(os.environ.get("FIE_ARC_STREAK_SOFT", "3"))
    arc_streak_hard = int(os.environ.get("FIE_ARC_STREAK_HARD", "5"))
    arc_cap_only = _env_bool("FIE_ARC_CAP_ONLY", False)
    if arc_cap_only:
        print(
            "[arc-cap-only] ON: streak/local ARC только min(size,cap), без ×0.5 и без skip",
            flush=True,
        )
    rk_wr_soft_enabled = _env_bool("FIE_RK_WR_SOFT", True)
    rk_wr_threshold = float(os.environ.get("FIE_RK_WR_THRESHOLD", "0.45"))
    rk_wr_min_trades = int(os.environ.get("FIE_RK_WR_MIN_TRADES", "15"))
    rk_wr_lookback = int(os.environ.get("FIE_RK_WR_LOOKBACK", "100"))
    size_soft_cap = float(os.environ.get("FIE_SIZE_SOFT_CAP", "0.25"))

    try:
        hard_loss_cap_frac = float(os.environ.get("FIE_HARD_LOSS_CAP_FRAC", "0.8") or 0)
    except ValueError:
        hard_loss_cap_frac = 0.0
    risk_bucket_filters = _env_bool("FIE_RISK_BUCKET_FILTERS", True)
    risk_b_min = float(os.environ.get("FIE_B_RK_MIN", "0.7"))
    risk_payoff_min_b = float(os.environ.get("FIE_PAYOFF_MIN_B", "1.0"))
    risk_b_lookback = int(os.environ.get("FIE_RK_B_LOOKBACK", "100"))
    risk_b_min_wl = int(os.environ.get("FIE_RK_B_MIN_WL", "5"))
    payoff_hard_skip = _env_bool("FIE_PAYOFF_HARD_SKIP", True)
    payoff_soft_mult = float(os.environ.get("FIE_PAYOFF_SOFT_MULT", "0.6"))
    payoff_sym_mult = float(os.environ.get("FIE_PAYOFF_SYM_MULT", "0.5"))
    if hard_loss_cap_frac > 0:
        print(
            f"[risk] hard_loss_cap: unrealized loss exit if pnl < -notional*{hard_loss_cap_frac:.2f}",
            flush=True,
        )
    if risk_bucket_filters:
        print(
            f"[risk] bucket filters: b>={risk_b_min}, payoff b>={risk_payoff_min_b} "
            f"(hard_skip={payoff_hard_skip})",
            flush=True,
        )

    _skip_rk_raw = (os.environ.get("FIE_SKIP_RK_KEYS") or "").strip()
    skip_rk_set: frozenset[str] = frozenset(x.strip() for x in _skip_rk_raw.split(",") if x.strip())
    if skip_rk_set:
        print(f"[skip-rk] FIE_SKIP_RK_KEYS: {len(skip_rk_set)} bucket(s)", flush=True)

    corr_defense = _env_bool("FIE_CORR_DEFENSE", False)
    corr_lookback = int(os.environ.get("FIE_CORR_LOOKBACK", "100"))
    corr_min_n = int(os.environ.get("FIE_CORR_MIN_N", "40"))
    corr_thr = float(os.environ.get("FIE_CORR_THRESHOLD", "0.0"))
    corr_mult = float(os.environ.get("FIE_CORR_DEFENSE_MULT", "0.5"))
    if corr_defense:
        print(
            f"[corr-defense] ON: lookback={corr_lookback} min_n={corr_min_n} "
            f"thr={corr_thr} mult={corr_mult}",
            flush=True,
        )

    if _env_bool("FIE_SIGNAL_EXPERIMENT", False):
        print(
            "[signal-experiment] ON: allow_prediction в zone принудительно true "
            "(FIE_SIGNAL_EXPERIMENT); MARKET_MAP/skip-rk/corr — по env",
            flush=True,
        )

    def _rk_to_db_key(rk: object) -> str | None:
        if isinstance(rk, tuple) and len(rk) == 3:
            try:
                return f"{int(rk[0])}|{str(rk[1])}|{str(rk[2])}"
            except Exception:
                return None
        return None

    def _ensure_regime_key_db(
        rk_value: object,
        *,
        fallback_tuple: tuple | None = None,
    ) -> str:
        """
        Гарантирует non-NULL regime_key при сохранении Trade.

        Причина: исторически часть сделок писалась с regime_key=NULL (≈25%),
        что ломает анализ режимов/часов и даёт ложные выводы по edge/drift.
        """
        # 1) Нормальный путь: rk_value уже tuple (hour, var_bucket, scen_bucket)
        key = _rk_to_db_key(rk_value)
        if key is not None:
            return key
        # 2) Fallback: используем заданный tuple
        if fallback_tuple is not None:
            key = _rk_to_db_key(fallback_tuple)
            if key is not None:
                return key
        # 3) Последний fallback: явный маркер вместо NULL
        return "UNKNOWN|unknown|unknown"

    def _win_rate_for_rk(
        session,
        strategy: str,
        rk: tuple,
        *,
        lookback: int,
        min_trades: int,
    ) -> Optional[float]:
        """Доля побед по закрытым сделкам bucket (strategy, rk); None если мало данных."""
        key = _rk_to_db_key(rk)
        if key is None:
            return None
        from sqlmodel import desc, select

        rows = list(
            session.exec(
                select(Trade.pnl)
                .where(Trade.strategy == str(strategy))
                .where(Trade.regime_key == key)
                .where(Trade.pnl.is_not(None))
                .order_by(desc(Trade.timestamp))
                .limit(int(lookback))
            ).all()
        )
        pnls = [float(x) for x in rows if x is not None and float(x) != 0.0]
        if len(pnls) < int(min_trades):
            return None
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        denom = wins + losses
        if denom <= 0:
            return None
        return float(wins) / float(denom)

    def _bucket_b_payoff(
        session,
        strategy: str,
        rk: tuple,
        *,
        lookback: int,
        min_wins: int,
        min_losses: int,
    ) -> Optional[tuple[float, float, float, int, int]]:
        """
        b = avg_win / |avg_loss| по bucket; avg_loss — среднее отрицательных pnl.
        None если мало сделок или нет wins/losses.
        """
        key = _rk_to_db_key(rk)
        if key is None:
            return None
        from sqlmodel import desc, select

        rows = list(
            session.exec(
                select(Trade.pnl)
                .where(Trade.strategy == str(strategy))
                .where(Trade.regime_key == key)
                .where(Trade.pnl.is_not(None))
                .order_by(desc(Trade.timestamp))
                .limit(int(lookback))
            ).all()
        )
        pnls = [float(x) for x in rows if x is not None and float(x) != 0.0]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        if len(wins) < int(min_wins) or len(losses) < int(min_losses):
            return None
        avg_win = sum(wins) / len(wins)
        avg_loss_neg = sum(losses) / len(losses)
        al = abs(avg_loss_neg)
        if al <= 1e-18:
            return None
        b = avg_win / al
        return (float(b), float(avg_win), float(avg_loss_neg), len(wins), len(losses))

    def _kelly_scale_for_bucket(
        *,
        session,
        strategy: str,
        rk_db_key: str,
        lookback: int = 200,
    ) -> float:
        """
        Kelly scaling по фактическим результатам в bucket (strategy, rk).
        Fractional Kelly: kelly_safe = min(kelly * 0.25, 0.25), мультипликатор base_size: 0.5 + kelly_safe ∈ [0.5, 0.75].
        """
        from sqlmodel import desc, select

        rows = list(
            session.exec(
                select(Trade.pnl)
                .where(Trade.strategy == str(strategy))
                .where(Trade.regime_key == rk_db_key)
                .where(Trade.pnl.is_not(None))
                .order_by(desc(Trade.timestamp))
                .limit(int(lookback))
            ).all()
        )
        pnls = [float(x) for x in rows if x is not None]
        pnls.reverse()
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]
        if len(wins) < 5 or len(losses) < 5:
            return 1.0
        p = len(wins) / (len(wins) + len(losses))
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)
        if avg_loss <= 0:
            return 1.0
        b = avg_win / avg_loss
        if b <= 0:
            return 1.0
        kelly = (b * p - (1 - p)) / b
        kelly = _clamp(kelly, 0.0, 1.0)
        kelly_safe = min(float(kelly) * 0.25, 0.25)
        return float(0.5 + kelly_safe)

    def _simple_kelly_size_for_bucket(
        *,
        session,
        strategy: str,
        rk_db_key: str,
        lookback: int = 200,
    ) -> tuple[float, float]:
        """
        Единственный драйвер размера (кроме streak-guard): size = base * (0.5 + kelly_safe),
        kelly_safe = min(kelly * 0.25, 0.25). Недостаточно сделок → kelly=0 → size = base*0.5.
        """
        from sqlmodel import desc, select

        base = float(os.environ.get("FIE_SIMPLE_BASE_SIZE", "0.08"))
        rows = list(
            session.exec(
                select(Trade.pnl)
                .where(Trade.strategy == str(strategy))
                .where(Trade.regime_key == rk_db_key)
                .where(Trade.pnl.is_not(None))
                .order_by(desc(Trade.timestamp))
                .limit(int(lookback))
            ).all()
        )
        pnls = [float(x) for x in rows if x is not None]
        pnls.reverse()
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]
        if len(wins) < 5 or len(losses) < 5:
            kelly = 0.0
        else:
            p = len(wins) / (len(wins) + len(losses))
            avg_win = sum(wins) / len(wins)
            avg_loss = sum(losses) / len(losses)
            if avg_loss <= 0:
                kelly = 0.0
            else:
                b = avg_win / avg_loss
                if b <= 0:
                    kelly = 0.0
                else:
                    kelly = (b * p - (1 - p)) / b
        kelly = _clamp(float(kelly), 0.0, 1.0)
        kelly_safe = min(kelly * 0.25, 0.25)
        size = base * (0.5 + kelly_safe)
        return float(size), float(kelly)

    hold_steps = int(os.environ.get("FIE_HOLD_STEPS", "5"))
    tp_pct = float(os.environ.get("FIE_TP_PCT", "0.002"))
    sl_pct = float(os.environ.get("FIE_SL_PCT", "0.002"))
    _hlc = hard_loss_cap_frac if hard_loss_cap_frac > 0 else 0.0
    broker_a = PaperBroker(
        capital=1.0,
        hold_steps=hold_steps,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        hard_loss_cap_frac=_hlc,
    )
    broker_b = PaperBroker(
        capital=1.0,
        hold_steps=hold_steps,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        hard_loss_cap_frac=_hlc,
    )
    peak_equity = broker_a.capital + broker_b.capital
    equity_window: list[float] = []
    equity_time_window: list[tuple[datetime, float]] = []
    current_dd = 0.0
    prev_variance: dict[str, float] = {"A": 0.05, "B": 0.05}  # стартуем с floor → стабильный bucket
    last_alert_at: dict[str, datetime] = {}
    last_tail_smart_alert: Optional[float] = None

    # ── Kill-switch state ────────────────────────────────────────────────────
    ks_dd_limit      = float(os.environ.get("FIE_KS_DD_LIMIT", "0.05"))
    ks_streak_limit  = int(os.environ.get("FIE_KS_STREAK_LIMIT", "20"))
    ks_soft_mult     = float(os.environ.get("FIE_KS_SOFT_SIZE_MULT", "0.25"))
    ks_neg_exp_n     = int(os.environ.get("FIE_KS_NEG_EXP_N", "100"))
    ks_cooldown_sec  = int(os.environ.get("FIE_KS_COOLDOWN_SEC", "1800"))   # 30 мин пауза
    # Потолок глобального streak (иначе после тысяч микро-лоссов лог показывает «streak=2793» — бессмысленно).
    loss_streak_cap = max(int(ks_streak_limit), int(os.environ.get("FIE_LOSS_STREAK_CAP", "250")))
    trading_disabled: dict[str, float] = {}   # reason → disabled_until timestamp
    recent_pnl_window: list[float] = []
    current_loss_streak = 0

    # ── Cooldown state ───────────────────────────────────────────────────────
    cooldown_sec    = int(os.environ.get("FIE_TRADE_COOLDOWN_SEC", "120"))  # 2 мин
    last_trade_ts: dict[str, float] = {}  # strategy → monotonic timestamp

    # ── Auto-regime learning ─────────────────────────────────────────────────
    # Ключ bucket: (hour, var, scen) — кросс-дневной; session_id только для логов / тегов.
    regime_session_id = (os.environ.get("FIE_REGIME_SESSION_ID") or "").strip()
    if not regime_session_id:
        regime_session_id = datetime.now(timezone.utc).strftime("%Y%m%d")
    with get_session() as _boot_session:
        regime_stats: dict[tuple, dict] = load_regime_stats(_boot_session)
    if regime_stats:
        print(f"[regime] loaded {len(regime_stats)} buckets from DB", flush=True)
    print(f"[regime] session_id={regime_session_id}", flush=True)
    regime_min_trades  = int(os.environ.get("FIE_REGIME_MIN_TRADES", "10"))
    regime_decay_lambda = float(os.environ.get("FIE_REGIME_DECAY", "0.98"))
    # strength = ema_exp / (ema_var + eps); k масштабирует влияние на size
    regime_strength_k  = float(os.environ.get("FIE_REGIME_STRENGTH_K", "0.1"))
    regime_adj_min     = float(os.environ.get("FIE_REGIME_ADJ_MIN", "-0.5"))   # max срезать -50%
    regime_adj_max     = float(os.environ.get("FIE_REGIME_ADJ_MAX",  "1.0"))   # max добавить +100%
    # Edge-aware sizing (второй слой поверх Kelly): см. блок после warmup cap
    edge_aware_enabled = _env_bool("FIE_EDGE_AWARE", True)
    if simple_kelly:
        edge_aware_enabled = False
    edge_k = float(os.environ.get("FIE_EDGE_K", "0.5"))
    edge_min = float(os.environ.get("FIE_EDGE_MIN", os.environ.get("FIE_EDGE_ADJ_MIN", "-0.9")))
    # edge_max: глобальный потолок adj (доверенный режим, ≥ trust_trades).
    # Снижен 1.0 → 0.5 чтобы max size x2.0 → x1.5 и убрать перегруз на «ложных плюсах».
    edge_max = float(os.environ.get("FIE_EDGE_MAX", os.environ.get("FIE_EDGE_ADJ_MAX", "0.5")))
    # edge_max_warmup: потолок adj до набора trust_trades — ещё консервативнее.
    edge_max_warmup = float(os.environ.get("FIE_EDGE_MAX_WARMUP", "0.25"))
    # Порог "доверия": до trust_trades используем edge_max_warmup, после — edge_max.
    edge_trust_trades = int(os.environ.get("FIE_EDGE_TRUST_TRADES", "50"))
    # Asymmetric sizing: усиливаем плюс сильнее, чем режем минус.
    # Позволяет системе агрессивнее ставить в хороших режимах и мягче срезать
    # в неопределённых — hard SKIP уже закрывает все устойчивые убытки.
    edge_pos_mult = float(os.environ.get("FIE_EDGE_POS_MULT", "1.1"))
    edge_neg_mult = float(os.environ.get("FIE_EDGE_NEG_MULT", "0.4"))
    # Порог SNR для hard SKIP: SNR = |ema_exp| / ema_var.
    # 1.0 = сигнал сильнее шума (старый, мягкий)
    # 0.5 = блокируем раньше — при SNR уже 0.5 (быстрее реагирует)
    # Поддерживаем оба ENV-имени: FIE_REGIME_SNR_SKIP (новый) и FIE_SKIP_SNR (legacy).
    skip_snr_threshold = float(
        os.environ.get("FIE_REGIME_SNR_SKIP", os.environ.get("FIE_SKIP_SNR", "0.5"))
    )
    # Early overbet protection: трёхуровневый потолок adj по кол-ву трейдов.
    # < early_trades → самый консервативный cap; warmup → средний; trust → полный.
    edge_max_early       = float(os.environ.get("FIE_EDGE_MAX_EARLY",  "0.15"))
    edge_early_trades    = int(os.environ.get("FIE_EDGE_EARLY_TRADES", "15"))

    # Fast edge (short-term brake): среднее по последним N закрытым сделкам этого rk.
    # Не заменяет EMA; ускоряет реакцию на локальную просадку. История только в RAM.
    fast_edge_n = int(os.environ.get("FIE_FAST_EDGE_N", "10"))
    fast_edge_soft_mult = float(os.environ.get("FIE_FAST_EDGE_SOFT_MULT", "0.5"))
    fast_edge_hard = float(os.environ.get("FIE_FAST_EDGE_HARD", "-0.00005"))
    fast_edge_min_trades = int(os.environ.get("FIE_FAST_EDGE_MIN_TRADES", "5"))
    # Если 1: fast_exp *= len(буфер)/N — при 5 из 10 точках реакция мягче (опционально).
    fast_edge_weighted = _env_bool("FIE_FAST_EDGE_WEIGHTED", False)
    # Mid risk guard: при отрицательном локальном edge не набирать «средний» размер.
    mid_guard_enabled = _env_bool("FIE_MID_GUARD", True)
    # Если 1: вместо SKIP — size *= 0.5 (mid → small), иначе полный пропуск входа.
    mid_guard_shrink = _env_bool("FIE_MID_GUARD_SHRINK", False)

    # Auto confidence scaling (ACS): после edge / guards, до gsm / enter.
    # FIE_CONF_SCALE: 0=выкл, 1=множитель к size, 2=shadow (только [conf-scale-shadow], size не трогаем).
    try:
        conf_scale_mode = int(os.environ.get("FIE_CONF_SCALE", "0").strip() or "0")
    except ValueError:
        conf_scale_mode = 0
    if conf_scale_mode not in (0, 1, 2):
        conf_scale_mode = 0
    if simple_kelly:
        conf_scale_mode = 0
    conf_fast_k = float(os.environ.get("FIE_CONF_FAST_K", "0.0001"))
    conf_consistency_n = int(os.environ.get("FIE_CONF_CONSISTENCY_N", "15"))
    conf_low_trades_cap = int(os.environ.get("FIE_CONF_TRADES_CAP", "20"))
    # ACS ступени: FIE_CONF_LOW/MID/HIGH, FIE_CONF_SCALE_LOW/MID/NEUTRAL/HIGH

    _size_mode_roll = (os.environ.get("FIE_SIZE_MODE") or "").strip().upper()
    rollback_edge_only = _size_mode_roll == "EDGE_ONLY" or _env_bool("FIE_ROLLBACK_SIMPLE", False)
    arc_layers_on = _env_bool("FIE_ARC_ENABLED", True)
    regime_sizing_on = _env_bool("FIE_REGIME_FILTER", True)
    if rollback_edge_only:
        conf_scale_mode = 0
        edge_aware_enabled = False
        mid_guard_enabled = False
        fast_edge_min_trades = 10**9  # глушит весь блок mid/fast/[edge-pre]/conf-scale внутри _fe_hist
        arc_layers_on = False
        regime_sizing_on = False
        rk_wr_soft_enabled = False
        risk_bucket_filters = False
        print(
            "[rollback] FIE_SIZE_MODE=EDGE_ONLY (или FIE_ROLLBACK_SIMPLE=1): "
            "size=SNR(ema); PM/регим/conf/ARC/risk-bucket/rk-wr OFF; meta edge_real только в БД",
            flush=True,
        )

    if safe_mode:
        # Жёсткая изоляция edge: полностью убираем усилители размера.
        edge_aware_enabled = False
        conf_scale_mode = 0
        mid_guard_enabled = False
        fast_edge_min_trades = 10**9  # effectively disables fast-edge

    # Global size multiplier: масштабируется по числу активных bucket-ов.
    # Чем меньше рынков с позитивным/нейтральным edge — тем осторожнее.
    gsm_low   = float(os.environ.get("FIE_GSM_LOW",  "0.5"))   # <=2 active
    gsm_mid   = float(os.environ.get("FIE_GSM_MID",  "1.0"))   # 3 active
    gsm_high  = float(os.environ.get("FIE_GSM_HIGH", "1.2"))   # >=4 active
    gsm_low_thresh  = int(os.environ.get("FIE_GSM_LOW_THRESH",  "2"))
    gsm_high_thresh = int(os.environ.get("FIE_GSM_HIGH_THRESH", "4"))

    # Live monitoring / alerts config
    alerts_enabled = _env_bool("FIE_ALERTS_ENABLED", True)
    alert_max_dd = float(os.environ.get("FIE_ALERT_MAX_DD", "0.40"))
    alert_min_size = float(os.environ.get("FIE_ALERT_MIN_SIZE", "0.02"))
    alert_max_size = float(os.environ.get("FIE_ALERT_MAX_SIZE", "0.25"))
    alert_max_kelly = float(os.environ.get("FIE_ALERT_MAX_KELLY", "10.0"))
    alert_eq_drop_1h = float(os.environ.get("FIE_ALERT_EQUITY_DROP_1H", "0.05"))
    alert_cooldown_sec = int(os.environ.get("FIE_ALERT_COOLDOWN_SEC", "300"))

    tail_anomaly_enabled = _env_bool("FIE_TAIL_ANOMALY_ALERTS_ENABLED", True)
    tail_anomaly_cooldown = float(os.environ.get("FIE_TAIL_ANOMALY_COOLDOWN_SEC", str(alert_cooldown_sec)))
    tail_event_window = int(os.environ.get("FIE_TAIL_EVENT_WINDOW", "3"))
    tail_smart_also = _env_bool("FIE_TAIL_SMART_ALERTS", False)
    tail_smart_size = float(os.environ.get("FIE_TAIL_SMART_SIZE_THRESHOLD", "0.05"))
    tail_smart_edge = float(os.environ.get("FIE_TAIL_SMART_EDGE_SCORE_THRESHOLD", "0.02"))
    tail_smart_tail_hit = float(os.environ.get("FIE_TAIL_SMART_TAIL_HIT_THRESHOLD", "0.5"))

    tail_dd_floor = float(os.environ.get("FIE_TAIL_DD_FLOOR", "0.3"))
    tail_dd_cap = float(os.environ.get("FIE_TAIL_DD_CAP", "0.6"))
    tail_thr_min = float(os.environ.get("FIE_TAIL_THRESHOLD_MIN", "0.2"))
    tail_thr_max = float(os.environ.get("FIE_TAIL_THRESHOLD_MAX", "0.7"))
    tail_base_thr = float(os.environ.get("FIE_TAIL_BASE_THRESHOLD", "0.5"))

    tail_monitor = DynamicTailEventMonitor(
        window=tail_event_window,
        cooldown_sec=tail_anomaly_cooldown,
        base_threshold=tail_base_thr,
        dd_floor=tail_dd_floor,
        dd_cap=tail_dd_cap,
        threshold_min=tail_thr_min,
        threshold_max=tail_thr_max,
    )

    slack_webhook = (os.environ.get("FIE_SLACK_WEBHOOK_URL") or os.environ.get("SLACK_WEBHOOK_URL") or "").strip()
    slack_send_tail: Optional[Callable[[str], None]] = None
    if slack_webhook:
        slack_send_tail = slack_webhook_sender(slack_webhook)

    def _telegram_send_tail(msg: str) -> None:
        try:
            from alerts.telegram_bot import send_telegram_alert

            send_telegram_alert(msg)
        except Exception as exc:
            print(f"[tail anomaly] telegram: {exc}", flush=True)

    telegram_send_tail: Optional[Callable[[str], None]] = (
        _telegram_send_tail if _env_bool("FIE_TAIL_ALERT_TELEGRAM", False) else None
    )

    def _disable_trading(reason: str) -> None:
        until = time.monotonic() + ks_cooldown_sec
        if reason not in trading_disabled or trading_disabled[reason] < until:
            trading_disabled[reason] = until
            print(f"[kill-switch] DISABLED: {reason} | cooldown={ks_cooldown_sec}s", flush=True)

    def _trading_allowed() -> tuple[bool, str]:
        """Возвращает (allowed, reason). Очищает истёкшие блокировки."""
        now_m = time.monotonic()
        expired = [r for r, until in trading_disabled.items() if now_m >= until]
        for r in expired:
            del trading_disabled[r]
            print(f"[kill-switch] RESTORED: {r}", flush=True)
        if trading_disabled:
            return False, next(iter(trading_disabled))
        return True, ""

    if conf_scale_mode != 0:
        print(
            f"[conf-scale] ACS mode={conf_scale_mode} "
            f"(0=off 1=live 2=shadow)",
            flush=True,
        )

    while True:
        if _env_bool("FIE_LOOP_HEARTBEAT", False):
            # Heartbeat для диагностики: включается только при FIE_LOOP_HEARTBEAT=1
            print("=== LOOP HEARTBEAT ===", flush=True)
        now = datetime.now(timezone.utc)
        signals = compute_signals()
        trade_dict = run_portfolio_logic(signals)  # опционально: ваш внутренний trade-лог

        # execution loop (paper mode): по стратегии A/B
        per_strategy: dict[str, dict] = {}
        if isinstance(signals, dict) and ("A" in signals or "B" in signals):
            for s in ("A", "B"):
                if isinstance(signals.get(s), dict):
                    per_strategy[s] = signals[s]
        elif isinstance(signals, dict):
            # fallback: единый payload
            per_strategy[str(signals.get("strategy", "A"))] = signals

        with get_session() as session:
            regime_now_ts = time.time()
            # signals_latest: ожидаем либо {"A": {...}, "B": {...}}, либо один сигнал + strategy
            if isinstance(signals, dict) and ("A" in signals or "B" in signals):
                for strategy in ("A", "B"):
                    payload = signals.get(strategy)
                    if not isinstance(payload, dict):
                        continue
                    upsert_signal_latest(
                        session,
                        SignalLatest(
                            timestamp=now,
                            strategy=strategy,
                            regime=payload.get("regime"),
                            volatility=payload.get("volatility"),
                            scen_accumulation=payload.get("scen_accumulation"),
                            breakout_up=payload.get("breakout_up"),
                            confidence=payload.get("confidence"),
                            gate_score=payload.get("gate_score"),
                        ),
                    )
            elif isinstance(signals, dict) and "strategy" in signals:
                upsert_signal_latest(
                    session,
                    SignalLatest(
                        timestamp=now,
                        strategy=str(signals["strategy"]),
                        regime=signals.get("regime"),
                        volatility=signals.get("volatility"),
                        scen_accumulation=signals.get("scen_accumulation"),
                        breakout_up=signals.get("breakout_up"),
                        confidence=signals.get("confidence"),
                        gate_score=signals.get("gate_score"),
                    ),
                )

            # ── Market-health: считаем активные/skip/warmup bucket-ы ────────
            # Активный = trades >= MIN_TRADES И не stable_loss (SNR <= skip_snr_threshold)
            _mh_active = _mh_skip = _mh_warmup = _mh_first_edge = 0
            for _rk, _rs in regime_stats.items():
                _rs_trades = int(_rs.get("trades", 0))
                _rs_exp    = float(_rs.get("ema_exp") or 0.0)
                _rs_var    = float(_rs.get("ema_var") or 0.0)
                _rs_snr    = abs(_rs_exp) / max(_rs_var, 1e-9)
                if _rs_trades < regime_min_trades:
                    _mh_warmup += 1
                    # [first-edge]: ранний позитивный сигнал в warmup
                    if _rs_trades >= 1 and _rs_exp > 0 and _rs_exp > _rs_var:
                        _mh_first_edge += 1
                        print(
                            f"[first-edge] {_rk} trades={_rs_trades} "
                            f"ema_exp={_rs_exp:+.6f} ema_var={_rs_var:.6f} "
                            f"snr={_rs_snr:.2f}",
                            flush=True,
                        )
                elif _rs_exp < 0 and _rs_snr > skip_snr_threshold:
                    _mh_skip += 1
                else:
                    _mh_active += 1

            # global_size_mult: осторожнее когда рынок плохой
            if rollback_edge_only:
                global_size_mult = 1.0
            elif _mh_active <= gsm_low_thresh:
                global_size_mult = gsm_low
            elif _mh_active >= gsm_high_thresh:
                global_size_mult = gsm_high
            else:
                global_size_mult = gsm_mid

            print(
                f"[market-health] active={_mh_active} skip={_mh_skip} "
                f"warmup={_mh_warmup} first_edge={_mh_first_edge} "
                f"gsm={global_size_mult:.1f}",
                flush=True,
            )

            # Paper execution -> positions
            for strategy, payload in per_strategy.items():
                prob_raw = compute_probability(payload)
                if prob_raw is None:
                    continue
                prob = _clamp(prob_raw, 0.0, 1.0)
                threshold = float(payload.get("threshold", 0.55))
                _action0 = signal_to_action(prob, threshold=threshold)

                # ── Entropy guard: если вероятность "залипла", не торгуем ──
                # Защита от режима "noise trader" (когда p_model схлопнулся в константу).
                if _env_bool("FIE_ENTROPY_GUARD", True):
                    win = int(os.environ.get("FIE_ENTROPY_WINDOW", "60"))
                    min_unique = int(os.environ.get("FIE_ENTROPY_MIN_UNIQUE", "3"))
                    key = str(strategy)
                    dq = _P_ENTROPY_CACHE.setdefault(key, deque(maxlen=max(5, win)))
                    dq.append(round(float(prob), 6))
                    if len(dq) >= max(10, min(win, 60)):
                        u = len(set(dq))
                        if u < min_unique:
                            print(
                                f"[entropy-guard] skip: strat={strategy} unique_p={u} "
                                f"window={len(dq)} p~={dq[-1]:.6f}",
                                flush=True,
                            )
                            continue

                # ── Market gate (production): торгуем только в MARKET_ACTIVE ──
                # И только после прогрева + минимальной амплитуды режима.
                _skip_mkt, _skip_reason = _market_gate_should_skip(
                    {
                        "market_status": payload.get("market_status"),
                        "market_active_min": payload.get("market_active_min"),
                        "market_active_peak": payload.get("market_active_peak"),
                    }
                )
                if _skip_mkt:
                    print(
                        f"[edge-gate] skip: strat={strategy} reason={_skip_reason} "
                        f"status={payload.get('market_status')} "
                        f"active_min={payload.get('market_active_min')} "
                        f"peak={payload.get('market_active_peak')}",
                        flush=True,
                    )
                    continue

                # ── Regime-local strategy override (вытащить локальный edge) ──
                # По метрикам: единственный стабильный карман сигнала сейчас в rk=9|mid|on,
                # причём внутри него есть +зона (p~0.32) и -зона (p~0.17).
                only_rk = (os.environ.get("FIE_ONLY_RK") or "").strip()
                if only_rk:
                    # Быстрый preview rk: variance берём из prev_variance (до обновления в этом тике),
                    # scen_acc — из payload. Для фильтра "hour|var|scen" этого достаточно.
                    rk_preview = _regime_key(
                        now.hour,
                        float(prev_variance.get(strategy, 0.05)),
                        float(payload.get("scen_accumulation", 0.0) or 0.0),
                    )
                    rk_db = _rk_to_db_key(rk_preview)
                    if rk_db != only_rk:
                        continue
                    # Локальная интерпретация p_model (не как универсальная вероятность).
                    if prob < 0.25:
                        _action0 = "BUY_NO"
                    elif prob < 0.40:
                        _action0 = "BUY_YES"
                    else:
                        _action0 = "HOLD"

                # Hour-weight: вместо hard hour filter — непрерывный множитель размера в целевой p-зоне.
                # Применяется только когда есть session (ниже) и только в p∈[0.25,0.40).
                edge_dir = _env_bool("FIE_EDGE_DIRECTION_FROM_REAL", True)
                if rollback_edge_only:
                    edge_dir = False
                if _action0 == "HOLD" and not edge_dir:
                    continue

                broker = broker_a if strategy == "A" else broker_b

                action = _action0
                # Быстрый тест: инверсия сигнала (проверка "знак перепутан").
                if _env_bool("FIE_INVERT_SIGNAL", False):
                    if action == "BUY_YES":
                        action = "BUY_NO"
                    elif action == "BUY_NO":
                        action = "BUY_YES"

                trade_p_model: float | None = None
                trade_p_market: float | None = None
                trade_edge_real: float | None = None

                confidence = payload.get("confidence")
                confidence = 0.5 if confidence is None else _clamp(float(confidence), 0.0, 1.0)
                alpha_multiplier = payload.get("alpha_multiplier")
                alpha_multiplier = 1.0 if alpha_multiplier is None else float(alpha_multiplier)

                # Источник edge:
                # legacy: |P-0.5| (теряет знак и плохо для сравнения с рынком)
                # market: P_model - P_market (continuous и напрямую измеряет mispricing)
                edge_src = os.environ.get("FIE_EDGE_SOURCE", "legacy").strip().lower()
                p_market_payload = payload.get("p_market")
                p_market_payload = (
                    None
                    if p_market_payload is None
                    else _clamp(float(p_market_payload), 0.0, 1.0)
                )
                if edge_src == "market" and p_market_payload is not None:
                    edge = float(prob) - float(p_market_payload)
                    # Для анализа/БД: edge_real должен быть доступен даже без live Polymarket slug.
                    trade_p_model = float(prob)
                    trade_p_market = float(p_market_payload)
                    trade_edge_real = float(edge)
                else:
                    edge = float(prob) - 0.5
                # Continuous threshold (без уровней/whitelist).
                # Для market-edge разумно фильтровать по |P_model-P_market|.
                edge_threshold_cont = float(
                    os.environ.get(
                        "FIE_EDGE_THRESHOLD_CONT",
                        os.environ.get("FIE_MIN_EDGE", "0.00005"),
                    )
                )
                if abs(edge) < edge_threshold_cont:
                    continue
                edge_score = float(edge) * float(alpha_multiplier)

                # Level-based whitelist оставляем только как откат/эксперимент (по умолчанию выключено).
                if _env_bool("FIE_EDGE_LEVEL_FILTER", False):
                    # Фильтр edge_score: edge_score дискретный → работаем по whitelist уровней.
                    # По умолчанию: сбалансированный набор (качество + частота).
                    # Чтобы не "цементировать" систему под один день рынка, список можно переопределить ENV:
                    #   FIE_ALLOWED_EDGE_LEVELS="0.343393,0.264238,0.277164"
                    raw_allowed = os.environ.get("FIE_ALLOWED_EDGE_LEVELS", "").strip()
                    if raw_allowed:
                        try:
                            allowed_levels = {
                                round(float(x.strip()), 6)
                                for x in raw_allowed.split(",")
                                if x.strip()
                            }
                        except ValueError:
                            allowed_levels = {0.343393, 0.264238, 0.277164}
                    else:
                        # Auto режим: если по rolling окну нет +EV уровней, levels будет пустым → пауза торговли.
                        allowed_levels = _auto_allowed_edge_levels(session)
                        if not allowed_levels:
                            allowed = False
                            e = round(edge_score, 6)
                            print(
                                f"[EDGE FILTER] edge={edge_score:.6f} e={e:.6f} allowed=False "
                                f"reason=EDGE_AUTO_PAUSE",
                                flush=True,
                            )
                            edge_log_path = os.environ.get("FIE_EDGE_FILTER_LOG", "/tmp/edge_filter.log")
                            try:
                                with open(edge_log_path, "a", encoding="utf-8") as f:
                                    f.write(f"{edge_score:.6f},0\n")
                            except OSError:
                                pass
                            continue

                    eps = float(os.environ.get("FIE_ALLOWED_EDGE_EPS", "0.0005"))
                    # Не сравниваем по round() "в лоб": уровни могут слегка дрейфовать.
                    closest_lvl = None
                    closest_diff = None
                    allowed = False
                    for lvl in allowed_levels:
                        d = abs(edge_score - float(lvl))
                        if closest_diff is None or d < closest_diff:
                            closest_diff = d
                            closest_lvl = float(lvl)
                        if d <= eps:
                            allowed = True
                            break
                    e = round(edge_score, 6)
                    print(
                        f"[EDGE FILTER] edge={edge_score:.6f} e={e:.6f} allowed={str(allowed)} "
                        f"eps={eps:.6f} closest={closest_lvl if closest_lvl is not None else 'n/a'} "
                        f"diff={closest_diff if closest_diff is not None else 'n/a'} "
                        f"allowed_levels={sorted(allowed_levels)}",
                        flush=True,
                    )
                    edge_log_path = os.environ.get("FIE_EDGE_FILTER_LOG", "/tmp/edge_filter.log")
                    try:
                        with open(edge_log_path, "a", encoding="utf-8") as f:
                            f.write(f"{edge_score:.6f},{int(allowed)}\n")
                    except OSError:
                        # Лог-файл не должен ломать торговый цикл.
                        pass
                    if not allowed:
                        continue

                _log_recent_signal_corr(session)

                recent_pnls = get_recent_pnls(session, strategy=strategy, limit=int(payload.get("var_limit", 200)))
                raw_variance = estimate_variance(recent_pnls, window=int(payload.get("var_window", 50)))
                variance_floor = float(payload.get("variance_floor", 0.05))
                new_variance = max(float(raw_variance), variance_floor)
                ema_alpha = _clamp(float(payload.get("variance_ema_alpha", 0.3)), 0.0, 1.0)
                prev_v = float(prev_variance.get(strategy, new_variance))
                variance = (1.0 - ema_alpha) * prev_v + ema_alpha * new_variance
                prev_variance[strategy] = variance

                _cur_scen_acc = float(payload.get("scen_accumulation", 0) or 0)
                cur_rk = _regime_key(now.hour, float(variance), _cur_scen_acc)
                _rk_skip = _rk_to_db_key(cur_rk)
                if skip_rk_set and _rk_skip and _rk_skip in skip_rk_set:
                    print(f"[skip-rk] bucket rk={_rk_skip}", flush=True)
                    continue

                _use_market_map = _env_bool("FIE_MARKET_MAP_FILTER", True)
                market_id = MARKET_MAP.get(cur_rk)
                if _use_market_map and not market_id:
                    print(
                        f"[market-map] skip rk={cur_rk} not in MARKET_MAP",
                        flush=True,
                    )
                    continue

                # Mispricing: всегда считаем для meta/БД при market_id + live цене
                # (раньше trade_p_* заполнялись только внутри FIE_EDGE_PM_FILTER → при filter=off
                #  broker.enter получал edge_real=None на всех 2290 сделках.)
                if market_id:
                    _gm_live = get_market_price(market_id)
                    if _gm_live is not None:
                        trade_p_market = _clamp(float(_gm_live), 0.0, 1.0)
                        # P_model должен оставаться модельной вероятностью, а не функцией от edge_score.
                        # Иначе мы "сжимаем" сигнал в почти-константу и теряем информацию.
                        trade_p_model = float(prob)
                        trade_edge_real = float(trade_p_model) - float(trade_p_market)

                # edge_real = P_model − P_market; фильтр входа (отдельно от заполнения meta)
                if (not rollback_edge_only) and _env_bool("FIE_EDGE_PM_FILTER", True):
                    if not market_id:
                        print(
                            f"[edge-real] skip no market_id rk={cur_rk} "
                            f"(нужен slug в MARKET_MAP для Polymarket)",
                            flush=True,
                        )
                        continue
                    if trade_edge_real is None:
                        print(
                            f"[edge-real] skip P_market=None market_id={market_id!r} rk={cur_rk}",
                            flush=True,
                        )
                        continue
                    edge_threshold = float(
                        os.environ.get(
                            "FIE_EDGE_THRESHOLD",
                            os.environ.get("FIE_EDGE_PM_MIN", "0.10"),
                        )
                    )
                    if abs(trade_edge_real) < edge_threshold:
                        print(
                            f"[edge-real] skip |edge_real|={abs(trade_edge_real):.4f}<{edge_threshold} "
                            f"P_model={trade_p_model:.4f} P_market={trade_p_market:.4f} "
                            f"edge_score={edge_score:.6f} rk={cur_rk}",
                            flush=True,
                        )
                        continue
                    if edge_dir:
                        action = "BUY_YES" if trade_edge_real > 0 else "BUY_NO"

                # Volatility filter: блокирует вход при слишком низкой EMA-variance.
                # variance_floor=0.05 → raw PnL variance всегда >= floor, поэтому
                # variance_min должен быть < floor чтобы фильтр имел смысл.
                # Default 0 = отключён; регим-лёрнинг обеспечивает фильтрацию.
                variance_min = float(os.environ.get("FIE_VARIANCE_MIN", "0"))
                warmup_n = int(os.environ.get("FIE_VARIANCE_WARMUP", "100"))
                past_trades_n = len(recent_pnls)
                filter_active = variance_min > 0 and past_trades_n >= warmup_n

                if filter_active and variance < variance_min:
                    current_price = payload.get("price", prob)
                    for fill in broker.step(current_price=current_price):
                        _vf_pm = fill.get("p_model")
                        _vf_pmm = fill.get("p_market")
                        _vf_er = fill.get("edge_real")
                        save_trade(session, Trade(
                            timestamp=fill.get("open_timestamp", now),
                            strategy=fill.get("strategy", strategy),
                            entry_type=fill.get("entry_type", "filtered"),
                            regime_key=_ensure_regime_key_db(
                                fill.get("regime_key"),
                                fallback_tuple=cur_rk,
                            ),
                            confidence=fill.get("confidence"),
                            gate_score=fill.get("gate_score"),
                            score_C=fill.get("score_C"),
                            d_score=fill.get("d_score"),
                            alpha_multiplier=fill.get("alpha_multiplier"),
                            edge=fill.get("edge"), edge_score=fill.get("edge_score"),
                            p_model=float(_vf_pm) if _vf_pm is not None else None,
                            p_market=float(_vf_pmm) if _vf_pmm is not None else None,
                            edge_real=float(_vf_er) if _vf_er is not None else None,
                            funding_stress_48=fill.get("funding_stress_48"),
                            liq_spike=fill.get("liq_spike"),
                            scen_breakout_suspicious=fill.get("scen_breakout_suspicious"),
                            momentum_up=fill.get("momentum_up"),
                            oi_strength=fill.get("oi_strength"),
                            size=fill.get("size"), variance=fill.get("variance"),
                            kelly_fraction=fill.get("kelly_fraction"),
                            tail_hit=fill.get("tail_hit"),
                            entry_price=round(fill["entry_price"], 6),
                            exit_price=round(fill["exit_price"], 6),
                            side=fill["side"], holding_steps=fill["holding_steps"],
                            hold_min=fill.get("hold_min"),
                            exit_reason=fill.get("exit_reason"),
                            mfe=fill.get("mfe"),
                            mae=fill.get("mae"),
                            pnl=round(float(fill["pnl"]), 8),
                        ))
                    continue  # не входим в позицию при низкой волатильности

                if safe_mode:
                    size = float(os.environ.get("FIE_SAFE_SIZE", "0.08"))
                    size = _clamp(size, 0.0, 1.0)
                    kelly_fraction = 0.0
                elif rollback_edge_only:
                    _rs_sz = regime_stats.get(cur_rk)
                    if not _rs_sz:
                        continue
                    _ema_e = _rs_sz.get("ema_exp")
                    _ema_v = float(_rs_sz.get("ema_var", 0.0))
                    if _ema_e is None or _ema_v <= 0:
                        continue
                    _ee = float(_ema_e)
                    snr = abs(_ee) / math.sqrt(_ema_v)
                    sn_mult = float(os.environ.get("FIE_EDGE_SN_MULT", "0.05"))
                    # Если sn_mult=0 → фиксированный size (эксперимент "есть ли edge без sizing")
                    if sn_mult <= 0:
                        size = float(os.environ.get("FIE_ROLLBACK_MIN_SIZE", "0.10"))
                    else:
                        size = min(0.25, max(0.05, snr * sn_mult))
                    kelly_fraction = 0.0
                    size = _clamp(size, 0.0, 1.0)
                    _rb_min = float(os.environ.get("FIE_ROLLBACK_MIN_SIZE", "0.08"))
                    _rb_max = float(os.environ.get("FIE_ROLLBACK_MAX_SIZE", "0.15"))
                    # Фиксируем диапазон: min/max могут быть равны (например 0.10/0.10)
                    if size < _rb_min:
                        continue
                    size = min(size, _rb_max)
                    if _env_bool("FIE_ROLLBACK_LOG", False):
                        print(
                            f"[edge-only-size] rk={cur_rk} snr={snr:.4f} ema_exp={_ee:+.6f} "
                            f"var={_ema_v:.6f} size={size:.4f}",
                            flush=True,
                        )
                elif simple_kelly:
                    _rk_db_sz = _rk_to_db_key(cur_rk)
                    if _rk_db_sz is None:
                        _base = float(os.environ.get("FIE_SIMPLE_BASE_SIZE", "0.08"))
                        size = _base * 0.5
                        kelly_fraction = 0.0
                    else:
                        size, kelly_fraction = _simple_kelly_size_for_bucket(
                            session=session,
                            strategy=strategy,
                            rk_db_key=_rk_db_sz,
                            lookback=int(os.environ.get("FIE_KELLY_LOOKBACK", "200")),
                        )
                    size = _clamp(size, 0.0, 1.0)
                    if size <= 0:
                        size = 0.0001
                    if _env_bool("FIE_KELLY_LOG", True):
                        print(
                            f"[simple-kelly] rk={cur_rk} key={_rk_db_sz} "
                            f"kelly={kelly_fraction:.4f} size={size:.4f}",
                            flush=True,
                        )
                else:
                    size, kelly_fraction = compute_kelly_size(
                        edge=edge,
                        confidence=confidence,
                        alpha_multiplier=alpha_multiplier,
                        variance=variance,
                        k=float(payload.get("kelly_k", 0.5)),
                        min_size=float(payload.get("min_size", 0.02)),
                        max_size=float(payload.get("max_size", 0.25)),
                        max_kelly=float(payload.get("max_kelly", 10.0)),
                    )
                    size = apply_dd_risk_scaling(size, current_dd=current_dd)
                    size = _clamp(size, 0.0, 1.0)

                    # Kelly scaling (по фактическим сделкам в bucket) — только sizing слой
                    if _env_bool("FIE_KELLY_SCALE", True):
                        _rk = _regime_key(
                            now.hour,
                            float(variance),
                            float(payload.get("scen_accumulation", 0) or 0),
                        )
                        _rk_db = _rk_to_db_key(_rk)
                        if _rk_db is not None:
                            base_size = float(size)
                            mult = _kelly_scale_for_bucket(
                                session=session,
                                strategy=strategy,
                                rk_db_key=_rk_db,
                                lookback=int(os.environ.get("FIE_KELLY_LOOKBACK", "200")),
                            )
                            size = _clamp(base_size * float(mult), 0.0, 1.0)
                            if size <= 0:
                                size = 0.0001
                            if _env_bool("FIE_KELLY_LOG", True):
                                print(
                                    f"[kelly-scale] rk={_rk} key={_rk_db} "
                                    f"base={base_size:.4f} mult={mult:.3f} size={size:.4f}",
                                    flush=True,
                                )

                if alerts_enabled:
                    trade_for_alert = {"strategy": strategy, "size": size, "kelly_fraction": kelly_fraction}
                    snapshot_for_alert = {"drawdown": current_dd, "equity_drop_1h": 0.0}
                    alerts = check_live_alerts(
                        trade_for_alert,
                        snapshot_for_alert,
                        max_dd_thresh=alert_max_dd,
                        min_size=alert_min_size,
                        max_size=alert_max_size,
                        max_kelly=alert_max_kelly,
                        equity_drop_thresh=alert_eq_drop_1h,
                    )
                    alerts = [
                        f"{a}; strategy={strategy}; variance={variance:.6f}; edge={edge:.6f}"
                        for a in alerts
                    ]
                    log_alerts(alerts, last_alert_at=last_alert_at, cooldown_sec=alert_cooldown_sec)

                # ── Kill-switch check ────────────────────────────────
                allowed, ks_reason = _trading_allowed()
                if not allowed:
                    print(f"[kill-switch] skip entry: {ks_reason}", flush=True)
                    continue

                # ── Cooldown check ───────────────────────────────────
                now_m = time.monotonic()
                last_ts = last_trade_ts.get(strategy, 0.0)
                if now_m - last_ts < cooldown_sec:
                    continue
                last_trade_ts[strategy] = now_m

                # ── Soft kill-switch: серия убытков → уменьшаем size (входы не режем) ──
                if (
                    (not rollback_edge_only)
                    and (not simple_kelly)
                    and current_loss_streak >= ks_streak_limit
                ):
                    _s_ks = size
                    size *= ks_soft_mult
                    size = _clamp(size, 0.0, 1.0)
                    if size <= 0:
                        size = 0.0001
                    print(
                        f"[kill-switch] SOFT: size {_s_ks:.4f}→{size:.4f} "
                        f"(loss_streak={current_loss_streak}>={ks_streak_limit})",
                        flush=True,
                    )

                # Базовый размер: Kelly + DD + soft streak — до warmup / pre-edge / fast-edge.
                # Mid-guard смотрит на него, чтобы не терять «сигнал хотел mid» после cap.
                base_size = float(size)

                # ── Hour-weight allocation (мягко, без hard hour filter) ──
                # Включаем только в целевой p-зоне (0.25..0.40), где у нас найден локальный edge.
                hw = _get_hour_weight(session, hour=int(now.hour), prob=float(prob))
                if hw != 1.0:
                    _s0 = size
                    size = _clamp(float(size) * float(hw), 0.0, 1.0)
                    if size <= 0:
                        size = 0.0001
                    if _env_bool("FIE_HOUR_WEIGHT_LOG", False):
                        print(f"[hour-weight] hour={now.hour} w={hw:.3f} size {_s0:.4f}→{size:.4f}", flush=True)
                # Context for analysis (known vs unknown hours)
                _hw_counts = _HOUR_WEIGHT_CACHE.get("counts")
                _hw_counts = _hw_counts if isinstance(_hw_counts, dict) else {}
                _hw_min_n_hour = int(os.environ.get("FIE_HOUR_WEIGHT_MIN_N_HOUR", os.environ.get("FIE_HOUR_WEIGHT_MIN_N", "20")))
                _hw_hour_n = int(_hw_counts.get(int(now.hour), 0) or 0)
                _hw_is_prior = bool(_hw_hour_n < _hw_min_n_hour)
                _p_bucket = float(int(float(prob) * 100.0)) / 100.0

                # ── Auto-regime filter + continuous weighting ────────
                _cur_edge_adj = 0.0   # edge adjustment применённый к size (для edge→size→pnl)
                # cur_rk / _cur_scen_acc уже заданы выше (bucket filter + variance)
                cur_rs = regime_stats.get(cur_rk)
                print(
                    f"[debug-rk] strategy={strategy} variance={variance:.6f} "
                    f"bucket={cur_rk} trades={int(cur_rs['trades']) if cur_rs else 0}",
                    flush=True,
                )
                if cur_rs is not None:
                    _apply_regime_time_decay(cur_rs, regime_now_ts, regime_decay_lambda)
                # ── Warmup size cap: режим не проверен → ограничиваем риск ──
                # Cap убирает крупные потери в период обучения, не ломая сам фильтр.
                warmup_size_cap = float(os.environ.get("FIE_WARMUP_SIZE_CAP", "0.10"))
                cur_trades = int(cur_rs["trades"]) if cur_rs else 0
                if regime_sizing_on and (not simple_kelly) and cur_trades < regime_min_trades:
                    if size > warmup_size_cap:
                        print(
                            f"[warmup-cap] {cur_rk} trades={cur_trades}/{regime_min_trades} "
                            f"size {size:.4f}→{warmup_size_cap:.4f}",
                            flush=True,
                        )
                        size = warmup_size_cap

                    # Pre-edge: слабый сигнал даже в warmup (ускоряет обучение).
                    warmup_edge_k = float(os.environ.get("FIE_WARMUP_EDGE_K", "0.05"))
                    if (
                        cur_rs is not None
                        and cur_trades >= 1
                        and cur_rs.get("ema_exp") is not None
                    ):
                        _we_exp = float(cur_rs["ema_exp"])
                        _we_var = float(cur_rs.get("ema_var", 0.0))
                        _we_str = _we_exp / (_we_var + 1e-8)
                        _we_adj = _clamp(_we_str * warmup_edge_k, -0.5, 0.5)
                        _size_before = size
                        size = _clamp(size * (1.0 + _we_adj), 0.0, 1.0)
                        if size <= 0:
                            size = 0.0001
                        print(
                            f"[pre-edge] {cur_rk} trades={cur_trades}/{regime_min_trades} "
                            f"str={_we_str:+.2f} adj={_we_adj:+.3f} "
                            f"size {_size_before:.4f}→{size:.4f}",
                            flush=True,
                        )

                # Локальный fast_exp (один раз) для mid-guard и fast-edge.
                _fe_key = (str(strategy), cur_rk)
                _fe_hist = _FAST_EDGE_HISTORY.get(_fe_key)
                fast_exp = None
                if _fe_hist is not None and len(_fe_hist) >= fast_edge_min_trades:
                    fast_exp = sum(_fe_hist) / len(_fe_hist)
                    if fast_edge_weighted:
                        fast_exp *= len(_fe_hist) / max(fast_edge_n, 1)

                    # === MID RISK GUARD: по base_size (до warmup/pre-edge), не по ужатому size ===
                    if (
                        (not simple_kelly)
                        and mid_guard_enabled
                        and fast_exp is not None
                        and fast_exp < 0
                        and 0.25 <= base_size <= 0.35
                    ):
                        if mid_guard_shrink:
                            _old_mg = size
                            size = _clamp(size * 0.5, 0.0, 1.0)
                            if size <= 0:
                                size = 0.0001
                            print(
                                f"[mid-guard] SHRINK rk={cur_rk} base_size={base_size:.4f} "
                                f"exp={fast_exp:.6f} size {_old_mg:.4f}→{size:.4f}",
                                flush=True,
                            )
                        else:
                            print(
                                f"[mid-guard] SKIP rk={cur_rk} base_size={base_size:.4f} "
                                f"exp={fast_exp:.6f}",
                                flush=True,
                            )
                            continue

                    # === FAST EDGE (short-term brake): до медленного [edge-pre] / EMA ===
                    # История: _FAST_EDGE_HISTORY (deque), не broker API — см. append при закрытии.
                    try:
                        if fast_exp is not None:
                            if (not simple_kelly) and fast_exp < 0:
                                _old_sz_fe = size
                                size *= fast_edge_soft_mult
                                size = _clamp(size, 0.0, 1.0)
                                if size <= 0:
                                    size = 0.0001
                                print(
                                    f"[fast-edge] SOFT rk={cur_rk} exp={fast_exp:.6f} "
                                    f"size {_old_sz_fe:.4f}→{size:.4f}",
                                    flush=True,
                                )
                            if fast_exp < fast_edge_hard:
                                print(
                                    f"[fast-edge] HARD SKIP rk={cur_rk} exp={fast_exp:.6f}",
                                    flush=True,
                                )
                                continue
                    except Exception as _fe_err:
                        print(f"[fast-edge] error: {_fe_err}", flush=True)

                    # Активируем только после накопления достаточной статистики (после decay)
                    if cur_rs and int(cur_rs["trades"]) >= regime_min_trades:
                        cur_ema_exp = cur_rs.get("ema_exp")
                        cur_ema_var = cur_rs.get("ema_var", 0.0)
                        if cur_ema_exp is not None:
                            # === PRE-COMPUTE EDGE (до SKIP — логируем SNR даже при отключении режима) ===
                            rk = cur_rk
                            stats = cur_rs
                            regime_edge = 0.0
                            regime_edge_adj = 0.0
                            ema_exp = float(stats["ema_exp"])
                            ema_var = float(stats.get("ema_var", 0.0))
                            denom = ema_var if ema_var > 1e-9 else 1e-9
                            regime_edge = ema_exp / denom

                            # Трёхуровневый trust threshold (Patch 3):
                            # < early_trades  → самый жёсткий cap (система только учится)
                            # early … trust   → средний cap (накопили начальную статистику)
                            # ≥ trust_trades  → полный cap (достаточно данных)
                            _bucket_trades = int(stats.get("trades", 0))
                            if _bucket_trades < edge_early_trades:
                                _eff_edge_max = edge_max_early
                            elif _bucket_trades < edge_trust_trades:
                                _eff_edge_max = edge_max_warmup
                            else:
                                _eff_edge_max = edge_max

                            # Sqrt dampening: sign(e) * sqrt(|e|) * k
                            # Сжимает взрывные значения edge (+2.2 → +1.48)
                            # при сохранении направления и малых значений почти без изменений.
                            import math as _math
                            _damped_edge = (
                                _math.copysign(_math.sqrt(abs(regime_edge)), regime_edge)
                            )

                            # Asymmetric sizing: позитивный edge усиливается сильнее,
                            # негативный (до hard-skip) срезается мягче.
                            asym_mult = edge_pos_mult if regime_edge >= 0 else edge_neg_mult
                            regime_edge_adj = _damped_edge * edge_k * asym_mult
                            regime_edge_adj = max(edge_min, min(_eff_edge_max, regime_edge_adj))
                            _trust_tier = (
                                "early"  if _bucket_trades < edge_early_trades else
                                "warmup" if _bucket_trades < edge_trust_trades  else
                                "trust"
                            )
                            print(
                                f"[edge-pre] rk={rk} trades={_bucket_trades} "
                                f"tier={_trust_tier} "
                                f"ema_exp={ema_exp:+.6f} "
                                f"ema_var={ema_var:.6f} "
                                f"edge={regime_edge:+.2f} damp={_damped_edge:+.2f} "
                                f"adj={regime_edge_adj:+.2f} "
                                f"cap={_eff_edge_max} "
                                f"asym={'pos' if regime_edge >= 0 else 'neg'}×{asym_mult}",
                                flush=True,
                            )

                            # 1. Hard skip: ema_exp < 0 и SNR выше порога.
                            # SNR = |ema_exp| / ema_var; порог 0.5 блокирует быстрее чем 1.0
                            _snr = abs(cur_ema_exp) / max(cur_ema_var, 1e-9)
                            stable_loss = cur_ema_exp < 0 and _snr > skip_snr_threshold
                            if stable_loss:
                                print(
                                    f"[regime-filter] SKIP {rk} edge={regime_edge:+.2f} snr={_snr:.2f}",
                                    flush=True,
                                )
                                continue

                            # 2. Edge-aware sizing (второй слой поверх Kelly)
                            if edge_aware_enabled:
                                size = size * (1.0 + regime_edge_adj)
                                if size <= 0:
                                    size = 0.0001
                                size = _clamp(size, 0.0, 1.0)
                                print(
                                    f"[edge-size] rk={rk} trades={int(stats.get('trades', 0))} "
                                    f"ema_exp={ema_exp:+.6f} "
                                    f"ema_var={ema_var:.6f} "
                                    f"edge={regime_edge:+.2f} adj={regime_edge_adj:+.2f} size={size:.4f}",
                                    flush=True,
                                )
                                # Запоминаем adj для лога edge→size→pnl при закрытии
                                _cur_edge_adj = regime_edge_adj

                            # 3. Continuous weighting (legacy): только если edge-aware выключен
                            elif (not edge_aware_enabled) and (not simple_kelly):
                                strength = cur_ema_exp / (cur_ema_var + 1e-8)
                                adj = _clamp(strength * regime_strength_k, regime_adj_min, regime_adj_max)
                                size = _clamp(size * (1.0 + adj), 0.0, 1.0)
                                if adj != 0.0:
                                    print(
                                        f"[regime-weight] {cur_rk} "
                                        f"strength={strength:.2f} adj={adj:+.3f} "
                                        f"size_after={size:.4f}",
                                        flush=True,
                                    )

                    # === Auto confidence scaling (ACS): после edge / mid-guard / fast-edge ===
                    if (
                        (not simple_kelly)
                        and conf_scale_mode > 0
                        and cur_rs is not None
                        and cur_rs.get("ema_exp") is not None
                    ):
                        try:
                            conf, conf_scale = _compute_conf_and_scale(
                                cur_rs,
                                fast_exp,
                                _fe_hist,
                                conf_fast_k=conf_fast_k,
                                consistency_n=conf_consistency_n,
                                low_trades_cap=conf_low_trades_cap,
                            )
                            if conf_scale_mode == 2:
                                print(
                                    f"[conf-scale-shadow] rk={cur_rk} conf={conf:.2f} "
                                    f"scale={conf_scale:.2f}",
                                    flush=True,
                                )
                            else:
                                _sz_cs = size
                                _mns = float(payload.get("min_size", 0.02))
                                _mxs = float(payload.get("max_size", 0.25))
                                size *= conf_scale
                                size = _clamp(size, _mns, _mxs)
                                if size <= 0:
                                    size = 0.0001
                                print(
                                    f"[conf-scale] rk={cur_rk} conf={conf:.2f} scale={conf_scale:.2f} "
                                    f"size {_sz_cs:.4f}→{size:.4f}",
                                    flush=True,
                                )
                        except Exception as _cs_err:
                            print(f"[conf-scale] error: {_cs_err}", flush=True)

                # Открываем позицию с реальной BTC ценой
                entry_price = payload.get("price", prob)

                # Применяем global_size_mult (зависит от числа активных bucket-ов)
                if (
                    (not rollback_edge_only)
                    and (not safe_mode)
                    and (not simple_kelly)
                    and global_size_mult != 1.0
                ):
                    _size_before_gsm = size
                    size = _clamp(size * global_size_mult, 0.0, 1.0)
                    if size <= 0:
                        size = 0.0001

                # ❗ ARC — непосредственно перед broker.enter() (после gsm — финальный size)
                global _arc_init_logged
                if not _arc_init_logged:
                    print("[ARC-INIT] loaded", flush=True)
                    _arc_init_logged = True
                rk = cur_rk

                if (
                    arc_layers_on
                    and (not safe_mode)
                    and (not simple_kelly)
                    and _env_bool("FIE_ARC_LOCAL", False)
                ):
                    # Локальный streak по bucket (по умолчанию выкл — основной ARC: [arc-streak] выше)
                    rk_key = (str(strategy), rk)
                    streak = local_loss_streaks.get(rk_key, 0)
                    ARC_SOFT = int(os.getenv("FIE_ARC_SOFT", "3"))
                    ARC_HARD = int(os.getenv("FIE_ARC_HARD", "5"))
                    ARC_MIN_SIZE = float(os.getenv("FIE_ARC_MIN_SIZE", "0.25"))
                    _loc_cap_soft = float(os.getenv("FIE_ARC_LOCAL_CAP_SOFT", "0.12"))
                    _loc_cap_hard = float(os.getenv("FIE_ARC_LOCAL_CAP_HARD", "0.08"))
                    if arc_cap_only:
                        if size >= ARC_MIN_SIZE and streak >= ARC_HARD:
                            old_size = size
                            size = _clamp(min(size, _loc_cap_hard), 0.0, 1.0)
                            print(
                                f"[arc-cap] LOCAL hard streak={streak} "
                                f"size {old_size:.4f}→{size:.4f} cap={_loc_cap_hard} rk={rk}",
                                flush=True,
                            )
                        elif size >= ARC_MIN_SIZE and streak >= ARC_SOFT:
                            old_size = size
                            size = _clamp(min(size, _loc_cap_soft), 0.0, 1.0)
                            print(
                                f"[arc-cap] LOCAL soft streak={streak} "
                                f"size {old_size:.4f}→{size:.4f} cap={_loc_cap_soft} rk={rk}",
                                flush=True,
                            )
                    else:
                        if size >= ARC_MIN_SIZE and streak >= ARC_HARD:
                            print(f"[arc] HARD SKIP rk={rk} streak={streak}", flush=True)
                            continue
                        if size >= ARC_MIN_SIZE and streak >= ARC_SOFT:
                            old_size = size
                            size = _clamp(size * 0.5, 0.0, 1.0)
                            if size <= 0:
                                size = 0.0001
                            print(
                                f"[arc] SOFT {old_size:.4f}→{size:.4f} rk={rk} streak={streak}",
                                flush=True,
                            )

                # STREAK GUARD / ARC (глобальный current_loss_streak): мягко при ≥soft, стоп при ≥hard
                if arc_layers_on:
                    _ls = int(current_loss_streak)
                    _cap_streak_soft = float(os.environ.get("FIE_ARC_CAP_STREAK_SOFT", "0.12"))
                    _cap_streak_hard = float(os.environ.get("FIE_ARC_CAP_STREAK_HARD", "0.08"))
                    if arc_cap_only:
                        if _ls >= arc_streak_hard:
                            _sg = size
                            size = _clamp(min(size, _cap_streak_hard), 0.0, 1.0)
                            if size <= 0:
                                size = 0.0001
                            print(
                                f"[arc-cap] GLOBAL hard streak={_ls} "
                                f"size {_sg:.4f}→{size:.4f} cap={_cap_streak_hard}",
                                flush=True,
                            )
                        elif _ls >= arc_streak_soft:
                            _sg = size
                            size = _clamp(min(size, _cap_streak_soft), 0.0, 1.0)
                            if size <= 0:
                                size = 0.0001
                            print(
                                f"[arc-cap] GLOBAL soft streak={_ls} "
                                f"size {_sg:.4f}→{size:.4f} cap={_cap_streak_soft}",
                                flush=True,
                            )
                    else:
                        if _ls >= arc_streak_hard:
                            print(
                                f"[arc-streak] skip entry loss_streak={_ls}>={arc_streak_hard}",
                                flush=True,
                            )
                            continue
                        if _ls >= arc_streak_soft:
                            _sg = size
                            size *= 0.5
                            size = _clamp(size, 0.0, 1.0)
                            if size <= 0:
                                size = 0.0001
                            print(
                                f"[arc-streak] loss_streak={_ls}>={arc_streak_soft} "
                                f"size {_sg:.4f}→{size:.4f}",
                                flush=True,
                            )

                if rk_wr_soft_enabled:
                    _wr = _win_rate_for_rk(
                        session,
                        strategy=strategy,
                        rk=cur_rk,
                        lookback=rk_wr_lookback,
                        min_trades=rk_wr_min_trades,
                    )
                    if _wr is not None and _wr < rk_wr_threshold:
                        _sg = size
                        size *= 0.5
                        size = _clamp(size, 0.0, 1.0)
                        if size <= 0:
                            size = 0.0001
                        print(
                            f"[rk-wr-soft] wr={_wr:.3f}<{rk_wr_threshold} "
                            f"size {_sg:.4f}→{size:.4f} rk={cur_rk}",
                            flush=True,
                        )

                if risk_bucket_filters:
                    _bp = _bucket_b_payoff(
                        session,
                        strategy=strategy,
                        rk=cur_rk,
                        lookback=risk_b_lookback,
                        min_wins=risk_b_min_wl,
                        min_losses=risk_b_min_wl,
                    )
                    if _bp is not None:
                        b_rk, aw, al_neg, _nw, _nl = _bp
                        abs_al = abs(al_neg)
                        if b_rk < risk_b_min:
                            print(
                                f"[rk-b] skip b={b_rk:.3f}<{risk_b_min} rk={cur_rk}",
                                flush=True,
                            )
                            continue
                        if aw < abs_al:
                            if payoff_hard_skip:
                                print(
                                    f"[payoff] skip avg_win={aw:.8f}<|avg_loss|={abs_al:.8f} "
                                    f"(b={b_rk:.3f}) rk={cur_rk}",
                                    flush=True,
                                )
                                continue
                            _sg = size
                            size *= payoff_soft_mult
                            size = _clamp(size, 0.0, 1.0)
                            if size <= 0:
                                size = 0.0001
                            print(
                                f"[payoff-soft] size {_sg:.4f}→{size:.4f} mult={payoff_soft_mult} rk={cur_rk}",
                                flush=True,
                            )
                        elif b_rk < risk_payoff_min_b:
                            _sg = size
                            size *= payoff_sym_mult
                            size = _clamp(size, 0.0, 1.0)
                            if size <= 0:
                                size = 0.0001
                            print(
                                f"[payoff-sym] b={b_rk:.3f}<{risk_payoff_min_b} "
                                f"size {_sg:.4f}→{size:.4f}",
                                flush=True,
                            )

                if size_soft_cap > 0 and size > size_soft_cap:
                    _sc = size
                    size = size_soft_cap
                    print(
                        f"[size-soft-cap] {_sc:.4f}→{size:.4f} cap={size_soft_cap:.2f}",
                        flush=True,
                    )

                if (
                    (not rollback_edge_only)
                    and trade_edge_real is not None
                    and _env_bool("FIE_EDGE_SIMPLE_SIZE", True)
                ):
                    _sz_floor = float(
                        os.environ.get(
                            "FIE_EDGE_THRESHOLD",
                            os.environ.get("FIE_EDGE_PM_MIN", "0.10"),
                        )
                    )
                    _ae = abs(trade_edge_real)
                    _raw = (_ae - _sz_floor) ** 2 * 5.0
                    size = min(0.25, max(0.05, _raw))
                    _big_cut = float(os.environ.get("FIE_EDGE_BIG_SIZE_MULT", "0.7"))
                    _big_thr = float(os.environ.get("FIE_EDGE_BIG_SIZE_THR", "0.15"))
                    if size > _big_thr:
                        size *= _big_cut
                    size = _clamp(size, 0.0, 1.0)

                if corr_defense:
                    _pairs = get_recent_size_pnl(session, strategy=strategy, limit=corr_lookback)
                    _r_sp = _pearson_size_pnl(_pairs) if len(_pairs) >= corr_min_n else None
                    if _r_sp is not None and _r_sp < corr_thr:
                        _sz0 = size
                        size = _clamp(size * corr_mult, 0.0, 1.0)
                        if size <= 0:
                            size = 0.0001
                        print(
                            f"[corr-defense] r(size,pnl)={_r_sp:+.3f}<{corr_thr} "
                            f"size {_sz0:.4f}→{size:.4f} n={len(_pairs)}",
                            flush=True,
                        )

                if trade_edge_real is not None:
                    # Для continuous mispricing режима: направление сделки берём из знака edge_real,
                    # независимо от threshold-based signal_to_action(prob).
                    # Иначе action часто остаётся HOLD (prob≈0.5) и сделок не будет вовсе.
                    if _env_bool("FIE_ACTION_FROM_EDGE_REAL", True):
                        action = "BUY_YES" if float(trade_edge_real) > 0 else "BUY_NO"
                    print(
                        f"[edge-real] rk={cur_rk} "
                        f"P_model={trade_p_model:.3f} "
                        f"P_market={trade_p_market:.3f} "
                        f"edge={trade_edge_real:+.4f} "
                        f"size={size:.3f}",
                        flush=True,
                    )

                if action == "HOLD":
                    continue

                # ── LF regime skip (risk-control): вырезать доказанно убыточные (liq,funding) пары ──
                # Включается только через FIE_LF_SKIP_ENABLED=1 (см. start_prod_loop.sh / env).
                try:
                    _liq_skip = payload.get("liq_spike")
                    _f_skip = payload.get("funding_stress_48")
                    _liq_skip_f = None if _liq_skip is None else float(_liq_skip)
                    _f_skip_f = None if _f_skip is None else float(_f_skip)
                except Exception:
                    _liq_skip_f, _f_skip_f = None, None
                if _lf_skip_trade(liq=_liq_skip_f, funding=_f_skip_f):
                    print(
                        f"[lf-skip] strat={strategy} rk={cur_rk} "
                        f"liq={_liq_skip_f} fund={_f_skip_f} action={action} size={size:.4f}",
                        flush=True,
                    )
                    continue

                _gsm_log = 1.0 if (simple_kelly or rollback_edge_only) else global_size_mult
                print(
                    f"[enter] strategy={strategy} side={action} size={size:.4f} "
                    f"prob={prob:.3f} rk={cur_rk} gsm={_gsm_log:.1f} "
                    f"status={payload.get('market_status')} "
                    f"active_min={payload.get('market_active_min')} "
                    f"peak={payload.get('market_active_peak')}",
                    flush=True,
                )
                if _env_bool("FIE_EDGE_ENTER_LOG", False):
                    print(
                        f"[EDGE ENTER] strat={strategy} edge_real={trade_edge_real} "
                        f"p_model={trade_p_model} p_market={trade_p_market}",
                        flush=True,
                    )
                # Только trade_* — не локальный edge_real / p_model (другие имена в скоупе)
                _meta_mispricing = {
                    "edge_real": trade_edge_real,
                    "p_model": trade_p_model,
                    "p_market": trade_p_market,
                }
                adjusted_hold_steps = _adjust_timeout_hold_steps(
                    hold_steps, payload.get("gate_score")
                )
                if adjusted_hold_steps != hold_steps:
                    print(
                        f"[exit-timeout-exp] gate_score={payload.get('gate_score')} "
                        f"hold_steps={hold_steps}->{adjusted_hold_steps}",
                        flush=True,
                    )
                broker.enter(
                    action=action,
                    size=size,
                    entry_price=entry_price,
                    hold_steps=adjusted_hold_steps,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                    meta={
                        "strategy": strategy,
                        "entry_type": action,
                        "confidence": confidence,
                        "gate_score": payload.get("gate_score"),
                        "score_C": payload.get("score_C"),
                        "d_score": payload.get("d_score"),
                        # Market-state context (чтобы анализировать lag/режимы в БД)
                        "market_status": payload.get("market_status"),
                        "market_active_min": payload.get("market_active_min"),
                        "market_active_peak": payload.get("market_active_peak"),
                        "alpha_multiplier": alpha_multiplier,
                        "c_size_multiplier": payload.get("c_size_multiplier"),
                        "strong_multiplier": payload.get("strong_multiplier"),
                        "edge": round(edge, 6),
                        "edge_score": round(edge_score, 6),
                        "variance": round(float(variance), 6),
                        "kelly_fraction": round(float(kelly_fraction), 6),
                        "tail_hit": payload.get("tail_hit"),
                        # Diagnostics for quantization debugging
                        "funding_stress_48": payload.get("funding_stress_48"),
                        "liq_spike": payload.get("liq_spike"),
                        "scen_breakout_suspicious": payload.get("scen_breakout_suspicious"),
                        "momentum_up": payload.get("momentum_up"),
                        "oi_strength": payload.get("oi_strength"),
                        "model_logit_b": payload.get("model_logit_b"),
                        # Allocation context (known vs unknown hours)
                        "hour_utc": int(now.hour),
                        "p_bucket": _p_bucket,
                        "hour_weight": float(hw),
                        "hour_n": _hw_hour_n,
                        "is_prior": int(_hw_is_prior),
                        "open_timestamp": now,
                        # Сохраняем ключ режима чтобы при закрытии
                        # обновлять ТОТЖЕ bucket, в котором открывались.
                        "regime_key": cur_rk,
                        # Edge adj для трейсинга edge→size→pnl
                        "edge_adj": round(_cur_edge_adj, 4),
                        **_meta_mispricing,
                    },
                )
                if _env_bool("FIE_DEBUG_EDGE_PIPELINE", False):
                    print(
                        f"[EDGE-PIPE after enter] strat={strategy} "
                        f"edge_real={trade_edge_real} p_model={trade_p_model} p_market={trade_p_market}",
                        flush=True,
                    )

                # Закрываем позиции по реальной BTC цене → realized PnL
                current_price = payload.get("price", prob)
                for fill in broker.step(current_price=current_price):
                    trade_pnl = float(fill["pnl"])
                    # p_model / p_market / edge_real: из meta при enter → PaperBroker копирует в fill при close
                    _pm = fill.get("p_model")
                    _pmm = fill.get("p_market")
                    _er = fill.get("edge_real")
                    _edge_for_db = float(_er) if _er is not None else None
                    # Вкл. только на чистом окне: старые закрытия без meta → падение
                    if _env_bool("FIE_EDGE_STRICT_SAVE", False):
                        if trade_pnl is not None and _edge_for_db is None:
                            raise RuntimeError("EDGE_REAL LOST BEFORE SAVE")
                    if _env_bool("FIE_DEBUG_EDGE_PIPELINE", False):
                        print(
                            f"[EDGE-PIPE save] strat={strategy} "
                            f"edge_real={_er} p_model={_pm} p_market={_pmm} "
                            f"'edge_real' in fill={'edge_real' in fill}",
                            flush=True,
                        )
                    # Только для «чистого окна»: иначе закрытия старых позиций без meta → ложный raise
                    if _env_bool("FIE_EDGE_LOST_RAISE", False):
                        if _pm is None and _pmm is None and _er is None:
                            raise RuntimeError(
                                "EDGE LOST BEFORE SAVE: fill без p_model/p_market/edge_real"
                            )
                    print(
                        "DEBUG_SAVE",
                        fill.get("mfe"),
                        fill.get("mae"),
                        fill.get("hold_min"),
                        fill.get("exit_reason"),
                        fill.get("timeout_hit"),
                        fill.get("reverse_hit"),
                        flush=True,
                    )
                    saved_trade = save_trade(
                        session,
                        Trade(
                            timestamp=fill.get("open_timestamp", now),
                            strategy=fill.get("strategy", strategy),
                            entry_type=fill.get("entry_type", action),
                            regime_key=_ensure_regime_key_db(
                                fill.get("regime_key"),
                                fallback_tuple=cur_rk,
                            ),
                            confidence=fill.get("confidence"),
                            gate_score=fill.get("gate_score"),
                            score_C=fill.get("score_C"),
                            d_score=fill.get("d_score"),
                            alpha_multiplier=fill.get("alpha_multiplier"),
                            c_size_multiplier=fill.get("c_size_multiplier"),
                            strong_multiplier=fill.get("strong_multiplier"),
                            edge=fill.get("edge"),
                            edge_score=fill.get("edge_score"),
                            p_model=float(_pm) if _pm is not None else None,
                            p_market=float(_pmm) if _pmm is not None else None,
                            edge_real=_edge_for_db,
                            funding_stress_48=fill.get("funding_stress_48"),
                            liq_spike=fill.get("liq_spike"),
                            scen_breakout_suspicious=fill.get("scen_breakout_suspicious"),
                            momentum_up=fill.get("momentum_up"),
                            oi_strength=fill.get("oi_strength"),
                            hour_utc=fill.get("hour_utc"),
                            p_bucket=fill.get("p_bucket"),
                            hour_weight=fill.get("hour_weight"),
                            hour_n=fill.get("hour_n"),
                            is_prior=bool(fill.get("is_prior")) if fill.get("is_prior") is not None else None,
                            market_active_min=fill.get("market_active_min"),
                            size=fill.get("size"),
                            variance=fill.get("variance"),
                            kelly_fraction=fill.get("kelly_fraction"),
                            tail_hit=fill.get("tail_hit"),
                            entry_price=round(fill["entry_price"], 6),
                            exit_price=round(fill["exit_price"], 6),
                            side=fill["side"],
                            holding_steps=fill["holding_steps"],
                            hold_min=fill.get("hold_min"),
                            exit_reason=fill.get("exit_reason"),
                            mfe=fill.get("mfe"),
                            mae=fill.get("mae"),
                            pnl=round(trade_pnl, 8),
                        ),
                    )
                    last = session.exec(
                        select(Trade)
                        .order_by(Trade.id.desc())
                        .limit(1)
                    ).first()
                    if last is not None:
                        print(
                            "DEBUG_READBACK",
                            last.id,
                            last.mfe,
                            last.mae,
                            last.hold_min,
                            last.exit_reason,
                            last.timeout_hit,
                            last.reverse_hit,
                            flush=True,
                        )
                    else:
                        print("DEBUG_READBACK NONE", flush=True)

                    # ── Kill-switch: обновляем loss streak ───────────────
                    if trade_pnl > 0:
                        current_loss_streak = 0
                    else:
                        current_loss_streak = min(current_loss_streak + 1, loss_streak_cap)

                    # ── Kill-switch: rolling expectancy ──────────────────
                    recent_pnl_window.append(trade_pnl)
                    if len(recent_pnl_window) > ks_neg_exp_n:
                        recent_pnl_window.pop(0)

                    # ── Trigger checks (HARD: только NEG_EDGE; streak — soft на входе) ──
                    if len(recent_pnl_window) >= ks_neg_exp_n:
                        rolling_exp = sum(recent_pnl_window) / len(recent_pnl_window)
                        if rolling_exp < 0:
                            _disable_trading(f"NEG_EDGE(exp={rolling_exp:.6f})")

                    # ── Regime stats: обновляем после каждой сделки ──────
                    # Приоритет: ключ, сохранённый при открытии позиции.
                    # Это гарантирует, что закрытие обновляет тот же bucket,
                    # в котором была принята торговая decision, даже если
                    # variance/scen_acc изменились за время удержания позиции.
                    fill_confidence = float(fill.get("confidence") or 0.5)
                    _saved_rk = fill.get("regime_key")
                    if isinstance(_saved_rk, tuple) and len(_saved_rk) == 3:
                        rk = _saved_rk
                    else:
                        # Fallback для позиций, открытых до этого фикса
                        fill_variance = float(fill.get("variance") or variance)
                        _cur_b = signals.get("B")
                        fill_scen_acc = float(
                            _cur_b.get("scen_accumulation", 0)
                            if isinstance(_cur_b, dict) else 0
                        )
                        rk = _regime_key(now.hour, fill_variance, fill_scen_acc)
                    rk_key = (str(fill.get("strategy", strategy)), rk)
                    # update streak (ARC v3)
                    if trade_pnl < 0:
                        local_loss_streaks[rk_key] = local_loss_streaks.get(rk_key, 0) + 1
                    else:
                        local_loss_streaks[rk_key] = 0
                    _FAST_EDGE_HISTORY.setdefault(rk_key, deque(maxlen=fast_edge_n)).append(
                        trade_pnl
                    )
                    rs = regime_stats.setdefault(
                        rk,
                        {
                            "trades": 0.0,
                            "pnl_sum": 0.0,
                            "wins": 0.0,
                            "ema_exp": None,
                            "ema_var": 0.0,
                            "last_update_ts": None,
                        },
                    )
                    _apply_regime_time_decay(rs, regime_now_ts, regime_decay_lambda)
                    rs["trades"] += 1.0
                    # confidence-weighted PnL для накопленной суммы
                    rs["pnl_sum"] += trade_pnl * fill_confidence
                    if trade_pnl > 0:
                        rs["wins"] += 1.0
                    # EMA expectancy: α=0.2 → эффективное окно ~5 сделок
                    # Patch 2 (confidence-weighted EMA): вес обновления пропорционален
                    # уверенности модели при открытии. Слабые сигналы почти не двигают
                    # EMA → убираем ложные positive-edge bucket'ы от случайных побед.
                    # fill_confidence уже [0,1]; типичный диапазон 0.5–0.8.
                    _ema_update_pnl = trade_pnl * fill_confidence
                    if rs["ema_exp"] is None:
                        rs["ema_exp"] = _ema_update_pnl
                    else:
                        rs["ema_exp"] = 0.8 * rs["ema_exp"] + 0.2 * _ema_update_pnl
                    # EMA variance (стабильность): насколько устойчив убыток/профит
                    rs["ema_var"] = 0.8 * rs["ema_var"] + 0.2 * abs(trade_pnl - rs["ema_exp"])
                    _tr = float(rs["trades"])
                    # Edge→size→pnl: связываем решение о размере с реальным исходом
                    _fill_edge_adj = fill.get("edge_adj")
                    if _fill_edge_adj is not None and _fill_edge_adj != 0.0:
                        _outcome = "WIN" if trade_pnl > 0 else "LOSS"
                        print(
                            f"[edge-outcome] rk={rk} "
                            f"adj={float(_fill_edge_adj):+.3f} "
                            f"size={fill.get('size', 0):.4f} "
                            f"pnl={trade_pnl:+.6f} "
                            f"conf_w={fill_confidence:.2f} {_outcome}",
                            flush=True,
                        )
                    print(
                        f"[regime-learn] {rk} trades={int(_tr)} "
                        f"ema_exp={rs['ema_exp']:.6f} ema_var={rs['ema_var']:.6f} "
                        f"avg_exp={rs['pnl_sum']/max(_tr, 1e-12):.6f} "
                        f"wr={rs['wins']/max(_tr, 1e-12):.2f}",
                        flush=True,
                    )
                    # Персистим в БД чтобы пережить рестарт
                    upsert_regime_stat(session, rk, rs)

            if isinstance(trade_dict, dict):
                # Если пользовательская портфельная логика вернула трейд, но regime_key не проставлен,
                # не пишем NULL: это ломает анализ режимов и может скрывать drift.
                _td_rk = trade_dict.get("regime_key")
                _td_rk_db = None
                if isinstance(_td_rk, str) and _td_rk:
                    _td_rk_db = _td_rk
                elif isinstance(_td_rk, tuple) and len(_td_rk) == 3:
                    _td_rk_db = _rk_to_db_key(_td_rk)
                if _td_rk_db is None:
                    # fallback к текущему (hour, var, scen) тика
                    _td_rk_db = _rk_to_db_key(cur_rk) or "UNKNOWN|unknown|unknown"
                save_trade(
                    session,
                    Trade(
                        timestamp=now,
                        strategy=str(trade_dict.get("strategy", "A")),
                        entry_type=str(trade_dict.get("entry_type", "enter")),
                        regime_key=_td_rk_db,
                        confidence=trade_dict.get("confidence"),
                        gate_score=trade_dict.get("gate_score"),
                        score_C=trade_dict.get("score_C"),
                        d_score=trade_dict.get("d_score"),
                        alpha_multiplier=trade_dict.get("alpha_multiplier"),
                        c_size_multiplier=trade_dict.get("c_size_multiplier"),
                        strong_multiplier=trade_dict.get("strong_multiplier"),
                        p_model=trade_dict.get("p_model"),
                        p_market=trade_dict.get("p_market"),
                        edge_real=trade_dict.get("edge_real"),
                        tail_hit=trade_dict.get("tail_hit"),
                        hold_min=trade_dict.get("hold_min"),
                        mfe=trade_dict.get("mfe"),
                        mae=trade_dict.get("mae"),
                        pnl=trade_dict.get("pnl"),
                    ),
                )

            # snapshot: либо ваш, либо минимальный от paper broker
            snap = signals.get("snapshot") if isinstance(signals, dict) else None
            if isinstance(snap, dict) and "equity" in snap:
                equity = float(snap["equity"])
                drawdown = float(snap.get("drawdown", 0.0))
                sharpe_rolling = snap.get("sharpe_rolling")
                capital_a = float(snap.get("capital_A", 0.0))
                capital_b = float(snap.get("capital_B", 0.0))
                cap_hits = int(snap.get("cap_hits", 0))
                tail_hits = int(snap.get("tail_hits", 0))
            else:
                capital_a = float(broker_a.capital)
                capital_b = float(broker_b.capital)
                equity = capital_a + capital_b
                peak_equity = max(peak_equity, equity)
                drawdown = 0.0 if peak_equity <= 0 else max(0.0, (peak_equity - equity) / peak_equity)
                current_dd = drawdown

                # ── Kill-switch: DD check ────────────────────────────────
                if drawdown > ks_dd_limit:
                    _disable_trading(f"DD_LIMIT({drawdown:.3f}>{ks_dd_limit:.3f})")

                equity_window.append(equity)
                equity_window = equity_window[-360:]  # ~ 6 часов при 60s
                sharpe_rolling = None
                if len(equity_window) >= 20:
                    rets = []
                    for i in range(1, len(equity_window)):
                        prev = equity_window[i - 1]
                        cur = equity_window[i]
                        rets.append((cur - prev) / max(prev, 1e-12))
                    mu = sum(rets) / len(rets)
                    var = sum((r - mu) ** 2 for r in rets) / max(len(rets) - 1, 1)
                    if var > 0:
                        sharpe_rolling = mu / (var**0.5) * (365 ** 0.5)
                cap_hits = 0
                tail_hits = 0

            # ── Regime tagging: логируем контекст каждого снапшота ──────────
            _b_payload = signals.get("B") if isinstance(signals, dict) else None
            _regime_tag = {
                "session_id": regime_session_id,
                "regime":   _b_payload.get("regime", "?") if _b_payload else "?",
                "volatility": _b_payload.get("volatility", "?") if _b_payload else "?",
                "scen_acc": round(float(_b_payload.get("scen_accumulation", 0)), 4) if _b_payload else 0,
                "gate":     round(float(_b_payload.get("gate_score", 0)), 4) if _b_payload else 0,
                "variance": round(float(list(prev_variance.values())[0]), 6),
                "hour_utc": now.hour,
                "ks_active": list(trading_disabled.keys()) if trading_disabled else [],
                "loss_streak": current_loss_streak,
                "rolling_exp": round(sum(recent_pnl_window) / max(len(recent_pnl_window), 1), 8),
            }
            print(
                f"[regime] {now.strftime('%H:%M')} | sid={regime_session_id} | "
                f"{_regime_tag['regime']}/{_regime_tag['volatility']} "
                f"scen={_regime_tag['scen_acc']} gate={_regime_tag['gate']} "
                f"eq={equity:.6f} dd={drawdown:.4f} streak={current_loss_streak} "
                f"exp={_regime_tag['rolling_exp']:.6f}",
                flush=True,
            )

            save_snapshot(
                session,
                PortfolioSnapshot(
                    timestamp=now,
                    equity=equity,
                    drawdown=drawdown,
                    sharpe_rolling=sharpe_rolling,
                    capital_A=capital_a,
                    capital_B=capital_b,
                    cap_hits=cap_hits,
                    tail_hits=tail_hits,
                ),
            )

            equity_time_window.append((now, equity))
            cutoff = now - timedelta(hours=1)
            equity_time_window = [(ts, eq) for ts, eq in equity_time_window if ts >= cutoff]
            if len(equity_time_window) >= 2:
                start_eq = equity_time_window[0][1]
                if start_eq > 0:
                    drop = max(0.0, (start_eq - equity) / start_eq)
                    if alerts_enabled:
                        snapshot_alert = {
                            "drawdown": drawdown,
                            "equity_drop_1h": drop,
                            "tail_hits": tail_hits,
                            "cap_hits": cap_hits,
                        }
                        alerts = check_live_alerts(
                            {},
                            snapshot_alert,
                            max_dd_thresh=alert_max_dd,
                            min_size=alert_min_size,
                            max_size=alert_max_size,
                            max_kelly=alert_max_kelly,
                            equity_drop_thresh=alert_eq_drop_1h,
                        )
                        alerts = [
                            f"{a}; tail_hits={tail_hits}; cap_hits={cap_hits}; start_eq={start_eq:.6f}; now_eq={equity:.6f}"
                            for a in alerts
                        ]
                        log_alerts(alerts, last_alert_at=last_alert_at, cooldown_sec=alert_cooldown_sec)

            if alerts_enabled and tail_anomaly_enabled:
                recent_for_tail = get_trades(session, since=now - timedelta(days=30), limit=2000)
                if recent_for_tail:
                    tdf_tail = pd.DataFrame(
                        [
                            {
                                "timestamp": t.timestamp,
                                "strategy": t.strategy,
                                "pnl": t.pnl,
                                "size": t.size,
                                "tail_hit": t.tail_hit,
                                "kelly_fraction": t.kelly_fraction,
                                "edge_score": t.edge_score,
                                "edge": t.edge,
                            }
                            for t in recent_for_tail
                        ]
                    )
                    snap_tail = {"equity": equity, "drawdown": drawdown}
                    tail_monitor.check_trades(
                        tdf_tail,
                        snap_tail,
                        slack_client=slack_send_tail,
                        telegram_client=telegram_send_tail,
                        size_threshold=tail_smart_size,
                        edge_score_threshold=tail_smart_edge,
                    )
                    if tail_smart_also:
                        last_tail_smart_alert = alert_tail_anomalies_smart(
                            tdf_tail,
                            snap_tail,
                            slack_client=slack_send_tail,
                            telegram_client=telegram_send_tail,
                            cooldown_sec=tail_anomaly_cooldown,
                            last_alert_at=last_tail_smart_alert,
                            size_threshold=tail_smart_size,
                            edge_score_threshold=tail_smart_edge,
                            tail_hit_threshold=tail_smart_tail_hit,
                        )

        time.sleep(poll_seconds)


if __name__ == "__main__":
    poll = int((os.environ.get("FIE_POLL_SECONDS", "5") or "5").strip())
    main(poll_seconds=max(1, poll))


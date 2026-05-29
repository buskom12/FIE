"""
Derivatives data (positions + flows) for BTC.

Приоритет источников:
  1) CSV (FIE_DERIVATIVES_CSV) — самый надёжный и повторяемый вариант
  2) Coinglass API (COINGLASS_API_KEY) — опционально, если есть ключ

Формат данных, который ожидаем дальше по пайплайну:
  список записей, отсортированных по timestamp (ms):
    {
      "timestamp": int,
      "funding_rate": float | None,            # напр. 0.0001
      "open_interest": float | None,           # USD или контрактный OI (как в источнике)
      "liq_long_usd": float | None,            # ликвидации лонгов
      "liq_short_usd": float | None,           # ликвидации шортов
    }

Важно:
  - API разных провайдеров часто меняется → поэтому CSV fallback обязателен.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import requests

_DATA_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE = _DATA_DIR / "cache" / "btc_derivatives_1h.json"

BASE_URL = "https://fapi.binance.com"

def _floor_to_hour_ms(ts_ms: int) -> int:
    """Floor timestamp (ms) to the start of its hour."""
    return int(ts_ms) - (int(ts_ms) % 3_600_000)


def _fetch_binance_json(url: str, params: dict[str, Any]) -> Any:
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=45)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    if last_err:
        raise last_err
    return None


def get_open_interest_hist(symbol: str = "BTCUSDT", *, period: str = "1h", limit: int = 500) -> Any:
    """
    Binance USD-M Futures open interest history.
    Endpoint: /futures/data/openInterestHist (public).
    """
    url = f"{BASE_URL}/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": period, "limit": int(limit)}
    return _fetch_binance_json(url, params)


def get_funding_rate(symbol: str = "BTCUSDT", *, limit: int = 100) -> Any:
    """
    Binance USD-M Futures funding history.
    Возвращает JSON как есть (list[dict] при успехе).
    """
    url = f"{BASE_URL}/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": int(limit)}
    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    return r.json()


def get_open_interest(symbol: str = "BTCUSDT") -> Any:
    """
    Binance USD-M Futures current open interest (snapshot).
    Возвращает JSON как есть (dict при успехе).
    """
    url = f"{BASE_URL}/fapi/v1/openInterest"
    params = {"symbol": symbol}
    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    return r.json()


def get_klines(symbol: str = "BTCUSDT", *, interval: str = "1h", limit: int = 200) -> Any:
    """
    Binance USD-M Futures klines (candles).
    Возвращает JSON как есть (list[list] при успехе).
    """
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": int(limit)}
    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    return r.json()


def _cache_path(explicit: Optional[str | Path]) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("FIE_DERIVATIVES_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_CACHE


def _is_cache_fresh(path: Path, max_age_hours: Optional[float]) -> bool:
    if max_age_hours is None:
        return True
    age_sec = time.time() - path.stat().st_mtime
    return age_sec < max_age_hours * 3600


def _load_json_cache(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Кэш {path} должен быть JSON-массивом объектов.")
    return [x for x in data if isinstance(x, dict)]


def _save_json_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _normalize_ts(ts: float) -> int:
    # поддерживаем секунды или миллисекунды
    if ts < 1e12:
        ts *= 1000
    return int(ts)


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _load_derivatives_csv(path: Path) -> list[dict[str, Any]]:
    """
    CSV ожидается с колонками (минимум timestamp):
      timestamp, funding_rate, open_interest, liq_long_usd, liq_short_usd

    Допускаем синонимы:
      time, date, datetime
      funding, fundingRate
      oi, openInterest
      long_liq, longLiquidations, liq_long
      short_liq, shortLiquidations, liq_short
    """
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Пустой или без заголовка: {path}")

        def norm_key(k: str) -> str:
            return k.strip().lower().replace(" ", "_")

        fieldmap = {norm_key(k): k for k in reader.fieldnames}

        def pick(*names: str) -> Optional[str]:
            for n in names:
                k = fieldmap.get(n)
                if k is not None:
                    return k
            return None

        ts_col = pick("timestamp", "time", "date", "datetime")
        fr_col = pick("funding_rate", "funding", "fundingrate")
        oi_col = pick("open_interest", "openinterest", "oi")
        ll_col = pick("liq_long_usd", "long_liq", "longliquidations", "liq_long")
        sl_col = pick("liq_short_usd", "short_liq", "shortliquidations", "liq_short")

        if not ts_col:
            raise ValueError(f"{path}: нужна колонка timestamp/time/date/datetime; есть: {reader.fieldnames}")

        for raw in reader:
            ts = _to_float(raw.get(ts_col))
            if ts is None:
                continue
            rows.append(
                {
                    "timestamp": _normalize_ts(ts),
                    "funding_rate": _to_float(raw.get(fr_col)) if fr_col else None,
                    "open_interest": _to_float(raw.get(oi_col)) if oi_col else None,
                    "liq_long_usd": _to_float(raw.get(ll_col)) if ll_col else None,
                    "liq_short_usd": _to_float(raw.get(sl_col)) if sl_col else None,
                }
            )

    rows.sort(key=lambda x: x["timestamp"])
    return rows


def _fetch_coinglass(
    *,
    endpoint: str,
    params: dict[str, Any],
    api_key: str,
) -> Any:
    """
    Очень тонкий слой: API Coinglass иногда меняется.
    Мы делаем best-effort fetch и возвращаем json.
    """
    url = f"https://open-api.coinglass.com/public/v2/{endpoint.lstrip('/')}"
    headers = {"coinglassSecret": api_key}
    r = requests.get(url, params=params, headers=headers, timeout=45)
    r.raise_for_status()
    return r.json()


def get_btc_derivatives(
    *,
    force_refresh: bool = False,
    max_age_hours: Optional[float] = 12.0,
    cache_path: Optional[str | Path] = None,
    csv_path: Optional[str | Path] = None,
    interval: str = "1h",
) -> list[dict[str, Any]]:
    """
    Возвращает ряд derivatives-данных для BTC, по умолчанию 1h.

    Если нет CSV и нет COINGLASS_API_KEY — вернёт пустой список.
    """
    # 1) CSV
    env_csv = os.environ.get("FIE_DERIVATIVES_CSV")
    candidates: list[Path] = []
    if csv_path is not None:
        candidates.append(Path(csv_path).expanduser().resolve())
    if env_csv:
        candidates.append(Path(env_csv).expanduser().resolve())
    for p in candidates:
        if p.is_file():
            return _load_derivatives_csv(p)

    # 2) cache
    cpath = _cache_path(cache_path)
    if cpath.is_file() and not force_refresh and _is_cache_fresh(cpath, max_age_hours):
        return _load_json_cache(cpath)

    # 3) Coinglass API (опционально)
    api_key = os.environ.get("COINGLASS_API_KEY")
    if not api_key:
        # 3a) Binance public fallback: fundingRate history + openInterestHist.
        # Это хуже, чем Coinglass (частота/покрытие), но лучше, чем "вечные нули".
        rows_by_ts: dict[int, dict[str, Any]] = {}

        def upsert(ts: int, **kwargs: Any) -> None:
            if ts not in rows_by_ts:
                rows_by_ts[ts] = {
                    "timestamp": ts,
                    "funding_rate": None,
                    "open_interest": None,
                    "liq_long_usd": None,
                    "liq_short_usd": None,
                }
            rows_by_ts[ts].update(kwargs)

        try:
            fr_js = get_funding_rate(symbol="BTCUSDT", limit=1000)
            if isinstance(fr_js, list):
                for it in fr_js:
                    if not isinstance(it, dict):
                        continue
                    ts = _to_float(it.get("fundingTime"))
                    fr = _to_float(it.get("fundingRate"))
                    if ts is None:
                        continue
                    ts_ms = _normalize_ts(ts)
                    upsert(_floor_to_hour_ms(ts_ms), funding_rate=fr)
        except Exception:
            pass

        try:
            oi_js = get_open_interest_hist(symbol="BTCUSDT", period="1h", limit=500)
            if isinstance(oi_js, list):
                for it in oi_js:
                    if not isinstance(it, dict):
                        continue
                    ts = _to_float(it.get("timestamp"))
                    oi = _to_float(it.get("sumOpenInterestValue", it.get("openInterest", it.get("openInterestValue"))))
                    if ts is None:
                        continue
                    ts_ms = _normalize_ts(ts)
                    upsert(_floor_to_hour_ms(ts_ms), open_interest=oi)
        except Exception:
            pass

        out = sorted(rows_by_ts.values(), key=lambda x: x["timestamp"])
        if out:
            try:
                _save_json_cache(cpath, out)
            except Exception:
                pass
        return out

    # Best-effort: если какой-то endpoint не сработает — вернём то, что смогли.
    # NOTE: из-за нестабильности внешних схем парсинг намеренно консервативный.
    rows_by_ts: dict[int, dict[str, Any]] = {}

    def upsert(ts: int, **kwargs: Any) -> None:
        if ts not in rows_by_ts:
            rows_by_ts[ts] = {
                "timestamp": ts,
                "funding_rate": None,
                "open_interest": None,
                "liq_long_usd": None,
                "liq_short_usd": None,
            }
        rows_by_ts[ts].update(kwargs)

    symbol = "BTC"

    # Funding
    try:
        js = _fetch_coinglass(endpoint="funding", params={"symbol": symbol, "interval": interval}, api_key=api_key)
        data = js.get("data") if isinstance(js, dict) else None
        items = data if isinstance(data, list) else (data.get("data") if isinstance(data, dict) else None)
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                ts = _to_float(it.get("time", it.get("timestamp")))
                fr = _to_float(it.get("fundingRate", it.get("funding_rate", it.get("funding"))))
                if ts is None:
                    continue
                upsert(_normalize_ts(ts), funding_rate=fr)
    except Exception:
        pass

    # Open interest
    try:
        js = _fetch_coinglass(endpoint="openInterest", params={"symbol": symbol, "interval": interval}, api_key=api_key)
        data = js.get("data") if isinstance(js, dict) else None
        items = data if isinstance(data, list) else (data.get("data") if isinstance(data, dict) else None)
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                ts = _to_float(it.get("time", it.get("timestamp")))
                oi = _to_float(it.get("openInterest", it.get("open_interest", it.get("oi"))))
                if ts is None:
                    continue
                upsert(_normalize_ts(ts), open_interest=oi)
    except Exception:
        pass

    # Liquidations
    try:
        js = _fetch_coinglass(endpoint="liquidation", params={"symbol": symbol, "interval": interval}, api_key=api_key)
        data = js.get("data") if isinstance(js, dict) else None
        items = data if isinstance(data, list) else (data.get("data") if isinstance(data, dict) else None)
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                ts = _to_float(it.get("time", it.get("timestamp")))
                ll = _to_float(it.get("longVolUsd", it.get("long_liq", it.get("liq_long_usd"))))
                sl = _to_float(it.get("shortVolUsd", it.get("short_liq", it.get("liq_short_usd"))))
                if ts is None:
                    continue
                upsert(_normalize_ts(ts), liq_long_usd=ll, liq_short_usd=sl)
    except Exception:
        pass

    out = sorted(rows_by_ts.values(), key=lambda x: x["timestamp"])
    if out:
        _save_json_cache(cpath, out)
    return out


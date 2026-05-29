"""
Источник OHLCV для BTC: Binance (основной), кэш на диске, опционально CSV.

Приоритет:
  1. Явный csv_path или файл из env FIE_BTC_CSV (локальный экспорт / Kaggle)
  2. Кэш JSON (если есть и не протух / не force_refresh / достаточно свечей)
  3. Binance API (пагинация до min_candles) → сохранение в кэш
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
_DEFAULT_CACHE = _DATA_DIR / "cache" / "btc_usdt_1h.json"
_DEFAULT_CSV = _DATA_DIR / "cache" / "btc_ohlcv.csv"

_BINANCE_URL = "https://api.binance.com/api/v3/klines"


def _cache_path(explicit: Optional[str | Path]) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("FIE_BTC_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_CACHE


def _csv_candidates(csv_path: Optional[str | Path]) -> list[Path]:
    paths: list[Path] = []
    if csv_path is not None:
        paths.append(Path(csv_path).expanduser().resolve())
    env = os.environ.get("FIE_BTC_CSV")
    if env:
        paths.append(Path(env).expanduser().resolve())
    paths.append(_DEFAULT_CSV)
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _normalize_row(k: list[Any]) -> dict[str, Any]:
    """Одна свеча Binance kline → dict с OHLCV."""
    return {
        "timestamp": int(k[0]),
        "open": float(k[1]),
        "high": float(k[2]),
        "low": float(k[3]),
        "price": float(k[4]),
        "volume": float(k[7]),
    }


def _migrate_legacy_row(row: dict[str, Any]) -> dict[str, Any]:
    """Старый кэш только с close → дублируем high/low/open."""
    p = float(row["price"])
    out = dict(row)
    out.setdefault("open", p)
    out.setdefault("high", p)
    out.setdefault("low", p)
    return out


def _load_json_cache(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Кэш {path} должен быть JSON-массивом объектов.")
    return [_migrate_legacy_row(x) if isinstance(x, dict) else x for x in data]


def _save_json_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _is_cache_fresh(path: Path, max_age_hours: Optional[float]) -> bool:
    if max_age_hours is None:
        return True
    age_sec = time.time() - path.stat().st_mtime
    return age_sec < max_age_hours * 3600


def _load_ohlcv_csv(path: Path) -> list[dict[str, Any]]:
    """CSV: timestamp (ms или s), close или price; опц. high, low, open, volume."""
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

        ts_col = pick("timestamp", "time", "open_time", "datetime", "date")
        price_col = pick("close", "price", "close_usd")
        vol_col = pick("volume", "quote_volume", "vol", "volume_usd")
        high_col = pick("high")
        low_col = pick("low")
        open_col = pick("open")

        if not ts_col or not price_col:
            raise ValueError(
                f"{path}: нужны колонки timestamp/time и close/price, "
                f"есть: {reader.fieldnames}"
            )

        for raw in reader:
            ts_raw = raw[ts_col].strip()
            try:
                ts = float(ts_raw)
            except ValueError:
                continue
            if ts < 1e12:
                ts = ts * 1000
            price = float(raw[price_col].strip())
            vol_s = raw.get(vol_col, "0") if vol_col else "0"
            vol = float(vol_s.strip() or 0) if isinstance(vol_s, str) else float(vol_s)
            high = float(raw[high_col].strip()) if high_col and raw.get(high_col) else price
            low = float(raw[low_col].strip()) if low_col and raw.get(low_col) else price
            op = float(raw[open_col].strip()) if open_col and raw.get(open_col) else price
            rows.append(
                {
                    "timestamp": int(ts),
                    "open": op,
                    "high": high,
                    "low": low,
                    "price": price,
                    "volume": vol,
                }
            )

    rows.sort(key=lambda x: x["timestamp"])
    return rows


def _fetch_binance_chunk(
    *,
    symbol: str,
    interval: str,
    limit: int,
    end_time: Optional[int] = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000),
    }
    if end_time is not None:
        params["endTime"] = end_time

    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            r = requests.get(_BINANCE_URL, params=params, timeout=45)
            r.raise_for_status()
            klines = r.json()
            break
        except (requests.RequestException, ValueError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    else:
        raise last_err  # type: ignore[misc]

    return [_normalize_row(k) for k in klines]


def _fetch_binance_paginated(
    min_candles: int,
    *,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
) -> list[dict[str, Any]]:
    """
    Собирает не менее min_candles свечей, идя в прошлое пачками по 1000.
    Результат: от старых к новым (хронология).
    """
    all_rows: list[dict[str, Any]] = []
    end_time: Optional[int] = None

    while len(all_rows) < min_candles:
        chunk = _fetch_binance_chunk(
            symbol=symbol,
            interval=interval,
            limit=1000,
            end_time=end_time,
        )
        if not chunk:
            break
        # chunk: от старой к новой внутри ответа; следующий запрос — ещё старее
        all_rows = chunk + all_rows
        end_time = chunk[0]["timestamp"] - 1
        time.sleep(0.15)

    return all_rows


def get_btc_ohlcv(
    *,
    force_refresh: bool = False,
    max_age_hours: Optional[float] = 24.0,
    cache_path: Optional[str | Path] = None,
    csv_path: Optional[str | Path] = None,
    binance_symbol: str = "BTCUSDT",
    binance_interval: str = "1h",
    min_candles: int = 12000,
) -> list[dict[str, Any]]:
    """
    Список свечей: timestamp, open, high, low, price (close), volume (quote USDT).

    min_candles: целевой размер истории с Binance (по умолчанию ~12k ≈ 1.3 года 1h).
    Переменная окружения FIE_MIN_CANDLES переопределяет значение по умолчанию.
    """
    env_min = os.environ.get("FIE_MIN_CANDLES")
    if env_min:
        try:
            min_candles = max(int(env_min), 1000)
        except ValueError:
            pass

    for csv_p in _csv_candidates(csv_path):
        if csv_p.is_file():
            return _load_ohlcv_csv(csv_p)

    cpath = _cache_path(cache_path)
    if cpath.is_file() and not force_refresh and _is_cache_fresh(cpath, max_age_hours):
        cached = _load_json_cache(cpath)
        if len(cached) >= min_candles:
            return cached

    rows = _fetch_binance_paginated(
        min_candles,
        symbol=binance_symbol,
        interval=binance_interval,
    )
    _save_json_cache(cpath, rows)
    return rows


_BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price"
_ticker_cache: dict = {"price": None, "ts": 0.0}
_TICKER_TTL = 5.0  # seconds


def get_btc_ticker_price(symbol: str = "BTCUSDT") -> float | None:
    """
    Realtime BTC цена с Binance ticker (легковесный endpoint).
    Кэшируется на _TICKER_TTL секунд, чтобы не спамить API.
    Возвращает float или None при ошибке.
    """
    now = time.time()
    if _ticker_cache["price"] is not None and now - _ticker_cache["ts"] < _TICKER_TTL:
        return _ticker_cache["price"]
    try:
        resp = requests.get(_BINANCE_TICKER_URL, params={"symbol": symbol}, timeout=3)
        resp.raise_for_status()
        price = float(resp.json()["price"])
        _ticker_cache["price"] = price
        _ticker_cache["ts"] = now
        return price
    except Exception:
        return _ticker_cache.get("price")  # вернуть последнее известное

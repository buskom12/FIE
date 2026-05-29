from __future__ import annotations

import time
import json
import logging
from typing import Any, Optional

import requests

try:
    import redis as redis_mod
except ImportError:
    redis_mod = None  # type: ignore

logger = logging.getLogger(__name__)

CACHE_TTL = 300  # секунд (5 минут)
RATE_LIMIT_DELAY = 1.0  # секунд между запросами

_redis: Optional[Any] = None
_redis_unavailable: bool = False


def _get_redis() -> Any | None:
    """Клиент Redis или None (нет пакета redis или сервер недоступен)."""
    global _redis, _redis_unavailable
    if redis_mod is None:
        if not _redis_unavailable:
            logger.debug("Пакет redis не установлен — кэш Polymarket отключён.")
            _redis_unavailable = True
        return None
    if _redis is not None:
        return _redis
    try:
        client = redis_mod.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        client.ping()
        _redis = client
        return _redis
    except redis_mod.exceptions.ConnectionError:
        logger.warning("Redis недоступен — кэш отключён, работаем напрямую.")
        return None


def get_polymarket_data(limit: int = 10) -> list[dict]:
    cache_key = f"polymarket:markets:{limit}"
    r = _get_redis()

    if r is not None:
        cached = r.get(cache_key)
        if cached:
            logger.debug("Cache HIT: %s", cache_key)
            return json.loads(cached)

    logger.debug("Cache MISS: %s — запрос к Polymarket API", cache_key)
    time.sleep(RATE_LIMIT_DELAY)

    url = "https://gamma-api.polymarket.com/markets"
    response = requests.get(url, params={"limit": limit}, timeout=10)
    response.raise_for_status()
    data = response.json()

    if r is not None:
        r.setex(cache_key, CACHE_TTL, json.dumps(data))

    return data


def extract_probabilities(markets: list[dict]) -> list[dict]:
    results = []
    for m in markets:
        try:
            prob = float(m["outcomePrices"][0])
            results.append({
                "question": m["question"],
                "probability": prob,
            })
        except (KeyError, ValueError, TypeError, IndexError):
            continue
    return results


def get_predictions() -> list[dict]:
    markets = get_polymarket_data()
    return extract_probabilities(markets)


def _parse_outcome_prices_yes(m: dict) -> float | None:
    """Первая цена исхода (YES) из outcomePrices, 0..1."""
    op = m.get("outcomePrices")
    if op is None:
        return None
    if isinstance(op, str):
        try:
            op = json.loads(op)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(op, (list, tuple)) or len(op) < 1:
        return None
    try:
        return float(op[0])
    except (TypeError, ValueError):
        return None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def get_market(market_id: str) -> dict | None:
    """
    Один рынок по id (gamma-api: GET /markets/{id}).
    Возвращает исходный объект + нормализованное поле \"price\" (YES, 0..1) при успехе.
    """
    if not market_id or not str(market_id).strip():
        return None
    mid = str(market_id).strip()
    cache_key = f"polymarket:market:{mid}"
    r = _get_redis()
    if r is not None:
        cached = r.get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                pass

    time.sleep(RATE_LIMIT_DELAY)
    url = f"https://gamma-api.polymarket.com/markets/{mid}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        m = response.json()
    except Exception as exc:
        logger.debug("get_market(%s): %s", mid, exc)
        return None

    if not isinstance(m, dict):
        return None

    yes = _parse_outcome_prices_yes(m)
    out = dict(m)
    if yes is not None:
        out["price"] = _clamp01(yes)
    else:
        out["price"] = 0.5

    if r is not None:
        try:
            r.setex(cache_key, CACHE_TTL, json.dumps(out))
        except Exception:
            pass

    return out

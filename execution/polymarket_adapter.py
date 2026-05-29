from __future__ import annotations

from markets.polymarket_real import get_market


def get_market_price(market_id: str) -> float | None:
    """
    Implied цена YES на Polymarket (0..1) по id рынка.
    Возвращает None при ошибке или отсутствии данных.
    """
    try:
        m = get_market(market_id)
        if m is None:
            return None
        p = m.get("price")
        if p is None:
            return None
        return float(p)
    except Exception:
        return None


def signal_to_action(probability: float, threshold: float = 0.55) -> str:
    """
    Minimal mapping:
    - prob > threshold      -> BUY_YES
    - prob < 1 - threshold  -> BUY_NO
    - иначе                -> HOLD
    """
    if probability > threshold:
        return "BUY_YES"
    if probability < 1.0 - threshold:
        return "BUY_NO"
    return "HOLD"


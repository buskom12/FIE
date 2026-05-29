import random


def get_market_probability(event: str) -> float:
    """
    Возвращает вероятность события с рынка предсказаний.

    TODO: заменить заглушку на реальный вызов Polymarket API.
    """
    return round(random.uniform(0.3, 0.7), 2)

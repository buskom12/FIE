"""
Portfolio Engine — управление позициями на основе сигналов FIE.

Подход: Kelly Criterion (quarter-Kelly) для расчёта размера позиции.
edge = P(FIE) - P(рынок); quarter-Kelly снижает дисперсию просадок.
"""


def kelly_fraction(probability: float, market_prob: float) -> float:
    """
    Упрощённая формула Келли для бинарных исходов.

    probability  — оценка вероятности FIE
    market_prob  — имплицитная вероятность рынка (из котировок)

    Возвращает долю капитала [0, 1].
    """
    edge = probability - market_prob
    if edge <= 0:
        return 0.0
    return edge / (1 - market_prob)


class PortfolioEngine:
    def __init__(self, capital: float = 10_000.0):
        self.capital = capital
        self.positions: list[dict] = []

    def calculate_position_size(self, probability: float, market_prob: float) -> float:
        """
        Размер позиции = капитал × Kelly × 0.25 (quarter-Kelly).

        Quarter-Kelly сохраняет математическое преимущество,
        существенно уменьшая волатильность кривой капитала.
        """
        kelly = kelly_fraction(probability, market_prob)
        return max(0.0, round(self.capital * kelly * 0.25, 2))

    def add_position(self, event: str, signal: str, probability: float, market_prob: float) -> dict:
        """Создаёт позицию и добавляет её в портфель."""
        size = self.calculate_position_size(probability, market_prob)
        position = {
            "event": event,
            "signal": signal,
            "probability": probability,
            "market_prob": market_prob,
            "kelly": kelly_fraction(probability, market_prob),
            "size": size,
        }
        self.positions.append(position)
        return position

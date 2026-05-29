"""
Backtest Engine — тестирование стратегии FIE на исторических данных.

Цикл: событие → агентский рой → агрегация → edge-сигнал → позиция → результат.
"""

from markets.market_engine import detect_edge
from portfolio.portfolio_engine import PortfolioEngine

# Модули agents и prediction будут подключены по мере их реализации
try:
    from agents.agent_swarm import run_swarm
    from prediction.aggregation import aggregate_predictions
except ImportError:
    run_swarm = None
    aggregate_predictions = None


class BacktestEngine:
    def __init__(self, capital: float = 10_000.0):
        self.portfolio = PortfolioEngine(capital=capital)
        self.results: list[dict] = []

    def run_event(self, event: str, real_outcome: bool) -> dict:
        """
        Прогоняет одно историческое событие через полный пайплайн FIE.

        event        — описание события
        real_outcome — фактический исход (True = событие произошло)
        """
        if run_swarm is None or aggregate_predictions is None:
            raise RuntimeError(
                "Модули agents.agent_swarm и prediction.aggregation ещё не реализованы."
            )

        agent_results = run_swarm(event)
        prediction = aggregate_predictions(agent_results)

        market_signal = detect_edge(event, prediction["probability"])

        position = None
        if market_signal["signal"]:
            position = self.portfolio.add_position(
                event,
                market_signal["signal"],
                market_signal["fie_probability"],
                market_signal["market_probability"],
            )

        result = {
            "event": event,
            "prediction": prediction["probability"],
            "market_probability": market_signal["market_probability"],
            "edge": market_signal["edge"],
            "signal": market_signal["signal"],
            "position": position,
            "real_outcome": real_outcome,
        }
        self.results.append(result)
        return result

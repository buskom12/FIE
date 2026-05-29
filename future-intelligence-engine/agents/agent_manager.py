from agents.swarm_factory import generate_swarm
from prediction.aggregation import aggregate_predictions


class AgentManager:

    def __init__(self):
        self.agents = generate_swarm(100)
        self._evolution_engine = None

    def _get_evolution_engine(self):
        """Ленивая инициализация EvolutionEngine во избежание циклических импортов."""
        if self._evolution_engine is None:
            import sys, os
            _root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
            if _root not in sys.path:
                sys.path.insert(0, os.path.abspath(_root))
            from learning.evolution_engine import EvolutionEngine
            self._evolution_engine = EvolutionEngine()
        return self._evolution_engine

    def evaluate_event(self, event: str) -> list[dict]:
        results = []

        for agent in self.agents:
            result = agent.evaluate_event(event)
            results.append(result)

        return results

    def update_after_outcome(self, actual_outcome: int) -> dict:
        """
        Запускает эволюцию весов агентов после получения реального исхода.

        Args:
            actual_outcome: Фактический исход события (0 или 1).

        Returns:
            Статистика итерации эволюции.
        """
        engine = self._get_evolution_engine()
        return engine.update(self.agents, actual_outcome)

    def run_simulation(self, event: str) -> dict:
        results = self.evaluate_event(event)
        aggregated = aggregate_predictions(results)

        return {
            "probability": aggregated["final_probability"],
            "confidence": aggregated["confidence"],
            "agents_count": aggregated["agents_count"],
        }

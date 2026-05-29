"""
Демонстрация эволюции агентов (без LLM).

Симулируем 50 итераций, где каждый агент имеет свою "встроенную точность"
(скрытый параметр). Через эволюцию система сама выявляет сильных.
"""

import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from learning.evolution_engine import EvolutionEngine, EvolutionConfig


# ---------------------------------------------------------------------------
# Минимальный стаб агента для демо (без LLM)
# ---------------------------------------------------------------------------

class _MockPersona:
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.accuracy = 0.5
        self.predictions_count = 0


class _MockAgent:
    """Агент с фиксированной точностью — имитирует реальный LLMAgent."""

    def __init__(self, name: str, true_accuracy: float, weight: float = 1.0):
        self.persona = _MockPersona(name, weight)
        self._true_accuracy = true_accuracy  # скрытая истинная точность
        self.last_prediction: float | None = None

    def predict(self, actual: int) -> float:
        """Возвращает вероятность с учётом истинной точности агента."""
        if random.random() < self._true_accuracy:
            # угадываем: выдаём вероятность по нужную сторону
            prob = random.uniform(0.55, 0.95) if actual == 1 else random.uniform(0.05, 0.45)
        else:
            # ошибаемся: выдаём вероятность по неверную сторону
            prob = random.uniform(0.05, 0.45) if actual == 1 else random.uniform(0.55, 0.95)
        self.last_prediction = prob
        return prob


# ---------------------------------------------------------------------------
# Эволюционная симуляция
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(42)

    # Создаём агентов с разной истинной точностью
    agents = [
        _MockAgent("smart_whale",     true_accuracy=0.75, weight=1.0),
        _MockAgent("macro_analyst",   true_accuracy=0.68, weight=1.0),
        _MockAgent("avg_trader",      true_accuracy=0.52, weight=1.0),
        _MockAgent("noise_gambler",   true_accuracy=0.40, weight=1.0),
        _MockAgent("contra_bear",     true_accuracy=0.35, weight=1.0),
        _MockAgent("random_retail",   true_accuracy=0.50, weight=1.0),
        _MockAgent("quant_model",     true_accuracy=0.72, weight=1.0),
        _MockAgent("risk_analyst",    true_accuracy=0.63, weight=1.0),
    ]

    engine = EvolutionEngine(EvolutionConfig(
        reward_factor=1.1,
        penalty_factor=0.9,
        weight_min=0.1,
        weight_max=10.0,
    ))

    print("=" * 55)
    print("  ЭВОЛЮЦИЯ АГЕНТОВ — 50 ИТЕРАЦИЙ")
    print("=" * 55)
    print(f"  {'Агент':<20} {'Точность':>8} {'Начальный вес':>14}")
    print("-" * 55)
    for a in agents:
        print(f"  {a.persona.name:<20} {a._true_accuracy:>7.0%}  {a.persona.weight:>13.2f}")
    print("=" * 55)

    # Генерируем события и прогоняем эволюцию
    iterations = 50
    for i in range(iterations):
        actual_outcome = random.randint(0, 1)

        # Каждый агент делает предсказание
        for agent in agents:
            agent.predict(actual_outcome)

        # Обновляем веса
        stats = engine.update(agents, actual_outcome)

        if (i + 1) % 10 == 0:
            dist = engine.get_weight_distribution(agents)
            print(f"  Итерация {i+1:3d} | "
                  f"batch acc: {stats['batch_accuracy']:.2%} | "
                  f"weights min={dist['min']:.2f} max={dist['max']:.2f} spread={dist['spread']:.2f}")

    # Финальный рейтинг
    print("\n" + "=" * 55)
    print("  ФИНАЛЬНЫЙ РЕЙТИНГ АГЕНТОВ")
    print("=" * 55)
    print(f"  {'Агент':<20} {'Вес':>8} {'EMA точность':>13} {'Предсказаний':>13}")
    print("-" * 55)
    sorted_agents = sorted(agents, key=lambda a: a.persona.weight, reverse=True)
    for a in sorted_agents:
        marker = " ★" if a.persona.weight >= 1.5 else ("  " if a.persona.weight >= 0.8 else " ↓")
        print(f"  {a.persona.name:<20} {a.persona.weight:>7.2f}{marker} "
              f"{a.persona.accuracy:>12.2%} {a.persona.predictions_count:>12}")
    print("=" * 55)

    top = engine.get_top_agents(agents, top_n=3)
    print(f"\n  Топ-3 агента: {', '.join(a.persona.name for a in top)}")
    print("  (система сама нашла сильных без знания их истинной точности)")


if __name__ == "__main__":
    main()

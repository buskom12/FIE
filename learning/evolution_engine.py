"""
EvolutionEngine — эволюция весов агентов на основе результатов предсказаний.

Принцип:
  - Агент угадал → weight *= reward_factor  (усиление)
  - Агент ошибся  → weight *= penalty_factor (ослабление)

После 100+ итераций слабые агенты стремятся к weight_min и перестают влиять,
сильные — доминируют в агрегации.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.llm_agent import LLMAgent


@dataclass
class EvolutionConfig:
    reward_factor: float = 1.1    # +10% за правильное предсказание
    penalty_factor: float = 0.9   # -10% за ошибку
    weight_min: float = 0.1       # нижний порог (агент не исчезает полностью)
    weight_max: float = 10.0      # верхний порог (защита от взрывного роста)
    decision_threshold: float = 0.5


class EvolutionEngine:
    """
    Управляет эволюцией весов агентов.

    Использование:
        engine = EvolutionEngine()
        engine.update(agents, actual_outcome=1)
    """

    def __init__(self, config: EvolutionConfig | None = None) -> None:
        self.config = config or EvolutionConfig()
        self.iteration: int = 0

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def update(self, agents: list["LLMAgent"], actual_outcome: int) -> dict:
        """
        Обновляет веса всех агентов на основе фактического исхода.

        Args:
            agents: Список LLMAgent с заполненным last_prediction.
            actual_outcome: Фактический исход события (0 или 1).

        Returns:
            Словарь со статистикой итерации.
        """
        self.iteration += 1
        correct_count = 0
        skipped_count = 0

        for agent in agents:
            if agent.last_prediction is None:
                skipped_count += 1
                continue

            predicted_class = 1 if agent.last_prediction > self.config.decision_threshold else 0
            is_correct = predicted_class == actual_outcome

            if is_correct:
                agent.persona.weight *= self.config.reward_factor
                correct_count += 1
            else:
                agent.persona.weight *= self.config.penalty_factor

            # Удерживаем вес в допустимых пределах
            agent.persona.weight = max(
                self.config.weight_min,
                min(self.config.weight_max, agent.persona.weight),
            )

            # Обновляем накопленную точность и счётчик
            self._update_accuracy(agent.persona, is_correct)

        total = len(agents) - skipped_count
        accuracy = correct_count / total if total > 0 else 0.0

        return {
            "iteration": self.iteration,
            "total_agents": len(agents),
            "correct": correct_count,
            "skipped": skipped_count,
            "batch_accuracy": round(accuracy, 4),
        }

    def get_top_agents(
        self, agents: list["LLMAgent"], top_n: int = 10
    ) -> list["LLMAgent"]:
        """Возвращает top_n агентов по текущему весу."""
        return sorted(agents, key=lambda a: a.persona.weight, reverse=True)[:top_n]

    def get_weight_distribution(self, agents: list["LLMAgent"]) -> dict:
        """Статистика распределения весов в текущем рое."""
        weights = [a.persona.weight for a in agents]
        if not weights:
            return {}
        return {
            "min": round(min(weights), 4),
            "max": round(max(weights), 4),
            "mean": round(sum(weights) / len(weights), 4),
            "spread": round(max(weights) - min(weights), 4),
        }

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @staticmethod
    def _update_accuracy(persona, is_correct: bool) -> None:
        """Обновляет скользящую точность персоны (exponential moving average)."""
        persona.predictions_count += 1
        alpha = 0.1  # коэффициент сглаживания EMA
        persona.accuracy = (1 - alpha) * persona.accuracy + alpha * (1.0 if is_correct else 0.0)


# ---------------------------------------------------------------------------
# Функциональный интерфейс (для обратной совместимости с заданием)
# ---------------------------------------------------------------------------

_default_engine = EvolutionEngine()


def update_agent_weights(
    agents: list[dict],
    actual_outcome: int,
    *,
    reward_factor: float = 1.1,
    penalty_factor: float = 0.9,
    weight_min: float = 0.1,
    weight_max: float = 10.0,
) -> list[dict]:
    """
    Обновляет веса агентов (dict-формат, совместимый с результатами run_swarm).

    Изменяет список in-place и возвращает его.

    Args:
        agents: Список словарей с ключами "probability" и "weight".
        actual_outcome: Фактический исход (0 или 1).
    """
    threshold = 0.5
    for agent in agents:
        predicted = 1 if agent["probability"] > threshold else 0
        if predicted == actual_outcome:
            agent["weight"] = min(weight_max, agent["weight"] * reward_factor)
        else:
            agent["weight"] = max(weight_min, agent["weight"] * penalty_factor)

    return agents

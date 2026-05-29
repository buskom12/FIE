from __future__ import annotations

from typing import TYPE_CHECKING

from agents.personas import AgentPersona
from memory.society_memory import SocietyMemory

if TYPE_CHECKING:
    from patterns.pattern_engine import PatternEngine


class BaseAgent:

    def __init__(self, persona: AgentPersona, memory: SocietyMemory = None):
        self.persona = persona
        self.memory = memory

    def evaluate_event(self, event: str) -> dict:
        past = []

        if self.memory:
            past = self.memory.retrieve_similar(event)

        probability = 0.5
        reasoning = f"{self.persona.role} analyzing event."

        if past:
            reasoning += f" Remembering {len(past)} similar past events."

        return {
            "agent": self.persona.name,
            "role": self.persona.role,
            "probability": probability,
            "reasoning": reasoning,
        }

    def evaluate_signals(self, signals: list[dict], pattern_engine: PatternEngine) -> dict:
        """
        Оценивает текущие сигналы через PatternEngine.

        data → signals → pattern match → probability → reasoning
        """
        result = pattern_engine.analyze(signals)
        probability = result["probability"]
        matched = result["matched_cases"]
        confidence = result["confidence"]

        reasoning = (
            f"{self.persona.role}: на основе {matched} исторических случаев "
            f"(уверенность: {confidence})"
        )

        return {
            "agent": self.persona.name,
            "role": self.persona.role,
            "probability": probability,
            "matched_cases": matched,
            "confidence": confidence,
            "reasoning": reasoning,
        }

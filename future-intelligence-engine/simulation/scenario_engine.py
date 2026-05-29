"""Scenario Engine — LLM-powered structured scenario generation via ollama."""

from __future__ import annotations

import ollama
from knowledge.graph_builder import update_graph

MODEL = "llama3"


def generate_scenarios(event: str) -> list[str]:
    """
    Calls LLM to generate 3 realistic and distinct scenarios for the given event.
    Returns a list of scenario strings (e.g. ["Scenario 1: ...", ...]).
    Falls back to deterministic scenarios if Ollama is unavailable.
    """
    prompt = f"""
You are a geopolitical and financial analyst AI.

Given the event:

{event}

Generate 3 possible future scenarios.
Each scenario must be realistic and different.

Format:

Scenario 1:
Scenario 2:
Scenario 3:
"""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response["message"]["content"]
        scenarios = [s.strip() for s in text.split("\n") if "Scenario" in s]
        scenarios = scenarios[:3] if scenarios else _fallback_scenarios(event)
        update_graph(event, scenarios)
        return scenarios
    except Exception as exc:
        print(f"[scenario_engine] Ollama not available — using fallback. ({exc})")
        scenarios = _fallback_scenarios(event)
        update_graph(event, scenarios)
        return scenarios


def _fallback_scenarios(event: str) -> list[dict]:
    """Deterministic scenarios used when LLM is unavailable."""
    return [
        {
            "title":       "Positive Outcome",
            "description": f"{event} leads to strong positive market reaction.",
            "probability": 0.40,
            "outcome":     "bullish",
        },
        {
            "title":       "Negative Outcome",
            "description": f"{event} triggers fear and negative market reaction.",
            "probability": 0.35,
            "outcome":     "bearish",
        },
        {
            "title":       "Neutral Outcome",
            "description": f"{event} is already priced in — market remains stable.",
            "probability": 0.25,
            "outcome":     "neutral",
        },
    ]

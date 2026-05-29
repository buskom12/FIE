"""Timeline Simulator — runs per-step agent evaluation across a generated timeline."""

from __future__ import annotations

from simulation.timeline_engine import generate_timeline
from agents.agent_manager import AgentManager
from prediction.aggregation import aggregate_predictions


def simulate_timeline(scenario: str, n_agents: int = 10) -> list[dict]:
    """
    Generates a timeline from a scenario and evaluates each step
    using a pool of LLM agents.

    Args:
        scenario:  Natural-language description of the event/situation.
        n_agents:  Number of agent personas to spin up per simulation run.

    Returns:
        List of dicts, one per timeline step::

            [
                {
                    "step":        str,   # full step label + description
                    "probability": float, # weighted average across agents
                    "confidence":  float, # 1 - disagreement
                },
                ...
            ]
    """
    manager = AgentManager()
    manager.create_agents(n_agents)

    steps = generate_timeline(scenario)

    timeline_results = []
    for step in steps:
        predictions = manager.evaluate_event(step)
        result = aggregate_predictions(predictions)
        timeline_results.append({
            "step":        step,
            "probability": result["final_probability"],
            "confidence":  result["confidence"],
        })

    return timeline_results

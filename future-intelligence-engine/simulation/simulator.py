"""Simulator — runs agent predictions across all scenarios for a given event."""

from simulation.scenario_generator import generate_scenarios
from agents.agent_manager import AgentManager
from prediction.aggregation import aggregate_predictions


def simulate_event(event: str, n_agents: int = 5) -> list[dict]:
    """
    Generates scenarios for an event and runs agent predictions on each.

    Returns a list of results, one per scenario:
        [{"scenario": str, "probability": float, "confidence": float}, ...]
    """
    manager = AgentManager()
    manager.create_agents(n=n_agents)

    scenarios = generate_scenarios(event)
    results = []

    for scenario in scenarios:
        raw_predictions = manager.evaluate_event(scenario)
        result = aggregate_predictions(raw_predictions)

        results.append({
            "scenario":    scenario,
            "probability": result["final_probability"],
            "confidence":  result["confidence"],
        })

    return results

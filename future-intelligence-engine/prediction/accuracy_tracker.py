import json
from datetime import datetime


OUTCOMES_FILE = "outcomes_history.json"


def record_outcome(event: str, agent_results: list, real_outcome: float) -> list:
    """
    Saves the real outcome (0.0 or 1.0) for an event and computes
    accuracy score for each agent based on how close their prediction was.

    real_outcome: 1.0 = event happened, 0.0 = did not happen
    Returns list of accuracy records per agent.
    """

    records = []

    for r in agent_results:
        error = abs(r["probability"] - real_outcome)
        accuracy = round(1.0 - error, 3)

        records.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "agent": r["agent"],
            "role": r["role"],
            "predicted": r["probability"],
            "real_outcome": real_outcome,
            "accuracy": accuracy,
        })

    try:
        with open(OUTCOMES_FILE, "r") as f:
            history = json.load(f)
    except Exception:
        history = []

    history.extend(records)

    with open(OUTCOMES_FILE, "w") as f:
        json.dump(history, f, indent=2)

    return records


def compute_agent_accuracy() -> dict:
    """
    Reads outcomes history and computes average accuracy per agent.
    Returns dict: {agent_name: avg_accuracy}
    """

    try:
        with open(OUTCOMES_FILE, "r") as f:
            history = json.load(f)
    except Exception:
        return {}

    scores: dict = {}
    counts: dict = {}

    for record in history:
        agent = record["agent"]
        scores[agent] = scores.get(agent, 0.0) + record["accuracy"]
        counts[agent] = counts.get(agent, 0) + 1

    return {
        agent: round(scores[agent] / counts[agent], 3)
        for agent in scores
    }

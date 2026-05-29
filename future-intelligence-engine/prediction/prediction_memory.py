import json
from datetime import datetime

MEMORY_FILE = "prediction_history.json"


def save_prediction(event, prediction, agent_results):

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        "prediction": prediction,
        "agents": agent_results
    }

    try:
        with open(MEMORY_FILE, "r") as f:
            history = json.load(f)
    except Exception:
        history = []

    history.append(record)

    with open(MEMORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

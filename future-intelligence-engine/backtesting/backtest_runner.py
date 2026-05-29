import numpy as np

from agents.agent_manager import AgentManager
from backtesting.historical_events import historical_events
from prediction.aggregation import aggregate_predictions


def run_backtest() -> float:
    manager = AgentManager()
    manager.create_agents()

    errors = []

    for event_data in historical_events:
        event = event_data["event"]
        real_outcome = event_data["real_outcome"]

        predictions = manager.evaluate_event(event)
        result = aggregate_predictions(predictions)
        predicted = result["final_probability"]

        error = abs(predicted - real_outcome)
        errors.append(error)

        print("Event:", event)
        print("Prediction:", predicted)
        print("Real:", real_outcome)
        print("Error:", error)
        print("-----")

    mean_error = float(np.mean(errors))

    print("\nBACKTEST RESULT")
    print("Events tested:", len(errors))
    print("Mean Error:", round(mean_error, 4))

    return mean_error

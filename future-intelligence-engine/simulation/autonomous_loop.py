import time

from database.store_prediction import save_prediction
from discovery.event_discovery import discover_events
from evaluation.evolution_engine import update_agent_weights
from prediction.event_impact import score_event_impact
from prediction.market_inefficiency import detect_inefficiency


def autonomous_loop(agent_manager, interval: int = 600) -> None:
    while True:
        print("\n=== DISCOVERING EVENTS ===\n")

        events = discover_events()

        for event in events:
            print("\nEvent:", event["title"])

            impact = score_event_impact(event["title"])
            print("Impact:", impact)

            if impact < 0.4:
                print("Low impact — skipping.")
                continue

            result = agent_manager.run_simulation(event["title"])
            print("Prediction:", result)

            market_probability = 0.50  # placeholder

            save_prediction(
                event["title"],
                result["probability"],
                market_probability,
                impact,
            )

            analysis = detect_inefficiency(
                result["probability"],
                market_probability,
            )

            print("\n=== MARKET ANALYSIS ===")
            print(analysis)

            # Симулируем реальный исход: если FIE >= 0.5 → событие произошло
            real_outcome = 1.0 if result["probability"] >= 0.5 else 0.0
            update_agent_weights(agent_manager.agents, real_outcome)
            print(f"Agent weights updated (outcome: {real_outcome})")

        print("\nSleeping...\n")
        time.sleep(interval)

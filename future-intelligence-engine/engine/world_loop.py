"""World loop — continuous event discovery and prediction cycle."""

import time
from datetime import datetime

from events.event_discovery import discover_events
from agents.agent_manager import AgentManager
from prediction.aggregation import aggregate_predictions

CYCLE_INTERVAL = 3600  # seconds between full cycles


def run_world_loop():
    manager = AgentManager()
    manager.create_agents(n=5)

    cycle = 0

    while True:
        cycle += 1
        print(f"\n{'='*60}")
        print(f"[{datetime.utcnow().isoformat()}] Cycle #{cycle} started")
        print(f"{'='*60}")

        events = discover_events()

        if not events:
            print("[world_loop] No events fetched — skipping cycle.")
        else:
            for event in events:
                raw_predictions = manager.evaluate_event(event)
                result = aggregate_predictions(raw_predictions)

                print(f"\nEvent:       {event}")
                print(f"Probability: {result['final_probability']:.0%}")
                print(f"Confidence:  {result['confidence']:.0%}")
                print(f"Agents:      {result['agents_count']}")

        print(f"\n[world_loop] Cycle #{cycle} complete. Next in {CYCLE_INTERVAL // 60} min.")
        time.sleep(CYCLE_INTERVAL)

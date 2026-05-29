def simulate_timeline(agent_manager, event: str, steps: int = 3) -> list[dict]:
    timeline = []
    current_event = event

    for step in range(steps):
        print(f"\n--- Simulation step {step} ---\n")

        result = agent_manager.run_simulation(current_event)
        print(f"probability: {result['probability']}")

        timeline.append(result)

        current_event = f"{event} (future step {step})"

    return timeline

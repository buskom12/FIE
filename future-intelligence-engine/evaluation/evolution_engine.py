def update_agent_weights(agents: list, real_outcome: float) -> None:
    for agent in agents:
        prediction = agent.last_prediction

        error = abs(prediction - real_outcome)

        agent.persona.predictions_count += 1

        agent.persona.accuracy = (
            (agent.persona.accuracy * (agent.persona.predictions_count - 1))
            + (1 - error)
        ) / agent.persona.predictions_count

        if error < 0.2:
            agent.persona.weight *= 1.05
        else:
            agent.persona.weight *= 0.95

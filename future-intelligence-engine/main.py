from agents.agent_manager import AgentManager
from agents.reputation import get_agent_weights, print_reputation_table
from prediction.accuracy_tracker import compute_agent_accuracy
from simulation.autonomous_loop import autonomous_loop


def main():

    manager = AgentManager()
    manager.create_agents()

    accuracy = compute_agent_accuracy()
    agent_names = [a.persona.name for a in manager.agents]
    weights = get_agent_weights(agent_names)

    # Apply reputation weights back to personas
    for agent in manager.agents:
        agent.persona.weight = weights[agent.persona.name]

    if any(v != 1.0 for v in weights.values()):
        print("\n--- Agent Reputation ---")
        print_reputation_table(weights, accuracy)

    autonomous_loop(manager)


if __name__ == "__main__":
    main()

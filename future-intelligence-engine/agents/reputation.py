from prediction.accuracy_tracker import compute_agent_accuracy

DEFAULT_WEIGHT = 1.0
MIN_WEIGHT = 0.3
MAX_WEIGHT = 2.0


def get_agent_weights(agent_names: list) -> dict:
    """
    Returns weight for each agent based on historical accuracy.
    Agents with no history get DEFAULT_WEIGHT = 1.0.
    Weight range: 0.3 (poor) → 2.0 (excellent).
    """

    accuracy = compute_agent_accuracy()

    weights = {}

    for name in agent_names:
        if name in accuracy:
            # Linear mapping: accuracy 0.0 → weight MIN, accuracy 1.0 → weight MAX
            w = MIN_WEIGHT + accuracy[name] * (MAX_WEIGHT - MIN_WEIGHT)
            weights[name] = round(w, 3)
        else:
            weights[name] = DEFAULT_WEIGHT

    return weights


def print_reputation_table(weights: dict, accuracy: dict) -> None:

    print(f"\n{'Agent':<15} {'Accuracy':>10} {'Weight':>8}")
    print("-" * 36)

    for agent, weight in sorted(weights.items(), key=lambda x: -x[1]):
        acc = accuracy.get(agent)
        acc_str = f"{acc:.0%}" if acc is not None else "no data"
        print(f"{agent:<15} {acc_str:>10} {weight:>8.2f}")

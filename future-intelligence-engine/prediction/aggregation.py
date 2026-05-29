def aggregate_predictions(agent_results):

    weighted_sum = 0
    total_weight = 0

    probabilities = []

    for r in agent_results:

        prob = r["probability"]
        weight = r.get("weight", 1)

        weighted_sum += prob * weight
        total_weight += weight

        probabilities.append(prob)

    final_probability = weighted_sum / total_weight

    disagreement = max(probabilities) - min(probabilities)

    confidence = 1 - disagreement

    return {
        "final_probability": round(final_probability, 3),
        "confidence": round(confidence, 3),
        "agents_count": len(probabilities)
    }

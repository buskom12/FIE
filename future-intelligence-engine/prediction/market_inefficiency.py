def detect_inefficiency(fie_probability: float, market_probability: float) -> dict:
    difference = fie_probability - market_probability

    return {
        "fie_probability": fie_probability,
        "market_probability": market_probability,
        "difference": round(difference, 4),
        "inefficiency_score": round(abs(difference), 4),
    }

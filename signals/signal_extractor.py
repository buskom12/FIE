SIGNAL_RULES = [
    {"keyword": "low volume", "type": "low_volume", "strength": 0.8},
    {"keyword": "no whales", "type": "no_whales", "strength": 0.9},
    {"keyword": "volatility compression", "type": "volatility_compression", "strength": 0.7},
]


def extract_signals(text: str) -> list[dict]:
    lowered = text.lower()
    return [
        {"type": rule["type"], "strength": rule["strength"]}
        for rule in SIGNAL_RULES
        if rule["keyword"] in lowered
    ]

from __future__ import annotations


def match_event_to_market(event: str, markets: list[dict]) -> dict | None:
    event_lower = event.lower()
    for m in markets:
        if event_lower in m.get("question", "").lower():
            return m
    return None


def match_all_events(events: list[str], markets: list[dict]) -> list[dict]:
    results = []
    for event in events:
        match = match_event_to_market(event, markets)
        if match:
            results.append({
                "event": event,
                "question": match["question"],
                "probability": match.get("outcomePrices", [None])[0],
            })
    return results

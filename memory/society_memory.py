"""
Society memory module.
Stores and manages persistent state for society simulation.
"""


class SocietyMemory:

    def __init__(self):
        self.events = []

    def store_event(self, event: str, outcome: str) -> None:
        memory = {
            "event": event,
            "outcome": outcome,
        }
        self.events.append(memory)

    def retrieve_similar(self, event: str) -> list:
        results = []
        for e in self.events:
            if any(word in event.lower() for word in e["event"].lower().split()):
                results.append(e)
        return results

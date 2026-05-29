"""
Knowledge graph module.
Stores and manages relationships between events.
"""


class KnowledgeGraph:

    def __init__(self):
        self.graph = {}

    def add_event(self, event: str) -> None:
        if event not in self.graph:
            self.graph[event] = []

    def add_relation(self, event_a: str, event_b: str) -> None:
        if event_a not in self.graph:
            self.graph[event_a] = []
        self.graph[event_a].append(event_b)

    def get_related(self, event: str) -> list:
        return self.graph.get(event, [])

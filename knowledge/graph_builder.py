from knowledge.knowledge_graph import KnowledgeGraph

graph = KnowledgeGraph()


def update_graph(event: str, scenarios: list) -> None:
    graph.add_event(event)

    for s in scenarios:
        graph.add_event(s)
        graph.add_relation(event, s)

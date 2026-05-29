import random
from agents.personas import AgentPersona
from agents.base_agent import BaseAgent


ROLES = [
    "crypto whale",
    "retail trader",
    "macro economist",
    "journalist",
    "hedge fund manager",
    "venture capitalist",
    "political analyst",
    "central banker",
    "startup founder",
    "developer",
    "AI researcher",
    "gambler",
    "market maker",
    "risk analyst",
]


def generate_personas(n: int = 50) -> list[AgentPersona]:
    """Генерирует список персон со случайными весами (используется в AgentManager)."""
    personas = []

    for i in range(n):
        role = random.choice(ROLES)
        persona = AgentPersona(
            name=f"Agent_{i}",
            role=role,
            description=f"{role} analyzing markets",
            risk_tolerance=random.uniform(0.2, 0.9),
            weight=random.uniform(0.5, 1.5),
        )
        personas.append(persona)

    return personas


def generate_swarm(size: int = 100) -> list[BaseAgent]:
    """
    Генерирует рой агентов типа BaseAgent.

    Все агенты стартуют с нейтральными метриками (weight=1.0, accuracy=0.5)
    и набирают репутацию в процессе работы.
    """
    agents = []

    for i in range(size):
        role = random.choice(ROLES)
        persona = AgentPersona(
            name=f"Agent_{i}",
            role=role,
            description=f"{role} analyzing markets",
            risk_tolerance=random.uniform(0.2, 0.9),
            weight=1.0,
            accuracy=0.5,
            predictions_count=0,
        )
        agents.append(BaseAgent(persona))

    return agents

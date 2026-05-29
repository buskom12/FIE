from agents.swarm_factory import generate_swarm

agents = generate_swarm(100)

print("Agents created:", len(agents))

for a in agents[:10]:
    print(a.persona.name, a.persona.role, a.persona.risk_tolerance)

from simulation.timeline_simulator import simulate_timeline

scenario = "Ethereum ETF approval leads to institutional inflows"

results = simulate_timeline(scenario)

for r in results:
    print(r)

from simulation.simulator import simulate_event

event = "SEC delays Ethereum ETF decision"

results = simulate_event(event)

for r in results:
    print(r)

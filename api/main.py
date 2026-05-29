from fastapi import FastAPI, HTTPException

from agents.agent_swarm import run_swarm
from prediction.aggregation import aggregate_predictions
from markets.market_engine import detect_edge

app = FastAPI(title="FIE — Future Intelligence Engine")


@app.get("/predictions")
def get_predictions():
    return [
        {
            "event": "ETH ETF approval",
            "probability": 0.62,
            "trend": "up",
        }
    ]


@app.get("/live")
def live_predictions(event: str = "Ethereum ETF approval"):
    agents = run_swarm(event)

    if not agents:
        raise HTTPException(status_code=503, detail="Рой агентов не вернул результатов.")

    prediction = aggregate_predictions(agents)
    market = detect_edge(event, prediction["probability"])

    return {
        "event": event,
        "prediction": prediction,
        "market": market,
        "agents": agents,
    }

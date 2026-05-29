"""Лёгкий mock-сервер для локального просмотра дашборда."""
from fastapi import FastAPI

app = FastAPI(title="FIE Mock Server")


@app.get("/live")
def live():
    return {
        "event": "Ethereum ETF approval",
        "prediction": {
            "probability": 0.72,
            "confidence": 0.85,
            "trend": "up",
        },
        "market": {
            "market_probability": 0.55,
            "edge": 0.17,
            "signal": "BUY",
        },
        "agents": [
            {
                "agent": "MacroAgent",
                "probability": 0.75,
                "reasoning": "Institutional demand and regulatory clarity support approval.",
            },
            {
                "agent": "SentimentAgent",
                "probability": 0.68,
                "reasoning": "Social media sentiment strongly positive over last 48h.",
            },
            {
                "agent": "TechnicalAgent",
                "probability": 0.73,
                "reasoning": "On-chain metrics show accumulation pattern consistent with pre-approval.",
            },
        ],
    }

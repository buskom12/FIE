"""
Исторические события для бэктестинга стратегии FIE.

outcome: 1 — событие произошло, 0 — не произошло.
"""

historical_events: list[dict] = [
    {
        "event": "SEC approves Bitcoin ETF",
        "outcome": 1,
    },
    {
        "event": "FTX collapse",
        "outcome": 1,
    },
    {
        "event": "ETH ETF rejected",
        "outcome": 0,
    },
    {
        "event": "US recession 2022",
        "outcome": 0,
    },
]

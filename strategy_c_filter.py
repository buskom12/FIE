from __future__ import annotations

from typing import Any


def strategy_c_filter(trade: dict[str, Any]) -> bool:
    scen = trade.get("scenario", {}) if isinstance(trade.get("scenario"), dict) else {}
    momentum_up = float(scen.get("momentum_up", 0.0))
    breakout_up = float(scen.get("breakout_up", 0.0))
    confidence = float(trade.get("confidence", 0.0))
    return (
        momentum_up <= 0.2985
        and breakout_up <= 0.4247
        and confidence >= 0.4425
    )


def strategy_c_score(trade: dict[str, Any]) -> int:
    scen = trade.get("scenario", {}) if isinstance(trade.get("scenario"), dict) else {}
    momentum_up = float(scen.get("momentum_up", 0.0))
    breakout_up = float(scen.get("breakout_up", 0.0))
    confidence = float(trade.get("confidence", 0.0))

    score = 0
    if momentum_up <= 0.3:
        score += 1
    if breakout_up <= 0.42:
        score += 1
    if confidence >= 0.44:
        score += 1
    return score

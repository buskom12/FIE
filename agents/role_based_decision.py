from __future__ import annotations

from typing import Any

from learning.role_weights import load_role_stats, load_role_weights
from markets.market_engine import EdgeZone


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# ---------------------------------------------------------------------------
# Natural-language reasoning builders (Smart Whale style)
# ---------------------------------------------------------------------------

def _smart_whale_reasoning(acc: float, oi: float, fstress: float, trend_flow: float) -> str:
    parts: list[str] = []
    if acc > 0.4:
        parts.append(f"сильное накопление ({acc:.2f})")
    elif acc > 0.25:
        parts.append(f"признаки накопления ({acc:.2f})")

    if oi > 0.35:
        parts.append(f"рост OI ({oi:.2f})")
    elif oi > 0.2:
        parts.append(f"умеренный рост OI ({oi:.2f})")

    if trend_flow > 0.35:
        parts.append(f"подтверждённый trend flow ({trend_flow:.2f})")

    if fstress < 0.15:
        parts.append("funding нейтральный")
    elif fstress > 0.4:
        parts.append(f"⚠️ funding перегрет ({fstress:.2f})")

    signal_str = ", ".join(parts) if parts else "сигналы слабые"

    if acc > 0.35 and oi > 0.25 and fstress < 0.3:
        conclusion = "→ идёт накопление → вероятен breakout вверх"
    elif trend_flow > 0.4:
        conclusion = "→ тренд подтверждён → продолжение движения"
    elif fstress > 0.4:
        conclusion = "→ осторожно, funding перегрет → breakout может быть ложным"
    else:
        conclusion = "→ накопление не подтверждено, жду сигнала"

    return f"{signal_str} {conclusion}"


def _breakout_trader_reasoning(bo: float, momentum: float, false_bo: float) -> str:
    parts: list[str] = []
    if bo > 0.5:
        parts.append(f"сильный пробой ({bo:.2f})")
    elif bo > 0.3:
        parts.append(f"пробой в процессе ({bo:.2f})")

    if momentum > 0.5:
        parts.append(f"импульс подтверждён ({momentum:.2f})")

    if false_bo > 0.4:
        parts.append(f"⚠️ риск ложного пробоя ({false_bo:.2f})")
    elif false_bo > 0.25:
        parts.append(f"осторожно: сухой объём ({false_bo:.2f})")

    signal_str = ", ".join(parts) if parts else "пробой не подтверждён"

    if bo > 0.4 and false_bo < 0.25:
        conclusion = "→ вхожу на пробой, объём подтверждает"
    elif bo > 0.3 and false_bo > 0.35:
        conclusion = "→ пробой подозрительный, жду ретест"
    else:
        conclusion = "→ нет чёткого пробоя"

    return f"{signal_str} {conclusion}"


def _risk_manager_reasoning(fstress: float, liq: float, false_bo: float, acc: float) -> str:
    risks: list[str] = []
    positives: list[str] = []

    if fstress > 0.4:
        risks.append(f"funding стресс={fstress:.2f}")
    if liq > 0.4:
        risks.append(f"ликвидационный спайк={liq:.2f}")
    if false_bo > 0.35:
        risks.append(f"ложный пробой={false_bo:.2f}")
    if acc > 0.3:
        positives.append(f"накопление поддерживает ({acc:.2f})")

    max_risk = max(fstress, liq, false_bo)

    if max_risk > 0.5:
        verdict = "→ ВЫСОКИЙ риск, сокращаю позицию"
    elif max_risk > 0.3:
        verdict = "→ умеренный риск, с осторожностью"
    else:
        verdict = "→ риски в норме"

    risk_str = ", ".join(risks) if risks else "явных рисков нет"
    pos_str = (", ".join(positives) + " ") if positives else ""
    return f"{risk_str} | {pos_str}{verdict}"


def _contrarian_reasoning(bo: float, false_bo: float, fstress: float, acc: float) -> str:
    traps: list[str] = []

    if fstress > 0.45:
        traps.append(f"funding перегрет ({fstress:.2f}) → breakout может быть ловушкой")
    if false_bo > 0.35:
        traps.append(f"сухой объём на пробое ({false_bo:.2f}) → вероятно ложный")
    if bo > 0.5 and fstress > 0.4:
        traps.append("большинство в лонгах + пробой → классический long trap")

    if not traps and acc > 0.35:
        return f"накопление реальное ({acc:.2f}), контрарий смягчает позицию → не разворот"

    trap_str = "; ".join(traps) if traps else "явных ловушек не вижу"

    if traps:
        conclusion = "→ высока вероятность разворота / ложного пробоя"
    else:
        conclusion = "→ рынок честный, без ловушек"

    return f"{trap_str} {conclusion}"


# ---------------------------------------------------------------------------
# Main: build votes
# ---------------------------------------------------------------------------

def build_role_votes(event: str, zone: EdgeZone) -> list[dict[str, Any]]:
    acc = zone.scen_accumulation
    bo = zone.breakout_up
    fstress = zone.funding_stress
    liq = zone.liq_spike
    false_bo = zone.scen_breakout_suspicious
    momentum = zone.momentum_up
    oi = zone.oi_strength
    trend_flow = zone.scen_trend_flow

    smart_money_prob = _clamp01(0.45 + 0.35 * acc + 0.20 * oi + 0.10 * trend_flow - 0.10 * fstress)
    breakout_prob = _clamp01(0.40 + 0.45 * bo + 0.20 * momentum - 0.10 * false_bo)
    risk_prob = _clamp01(0.50 - 0.30 * fstress - 0.25 * liq - 0.20 * false_bo + 0.10 * acc)
    contrarian_prob = _clamp01(0.55 - 0.45 * bo + 0.30 * false_bo + 0.20 * fstress)

    role_weights = load_role_weights()
    role_stats = load_role_stats()

    def _historical_weight(role: str, fallback: float) -> float:
        st = role_stats.get(role, {})
        if isinstance(st, dict) and "accuracy" in st:
            return max(0.1, min(1.0, float(st.get("accuracy", 0.5))))
        return max(0.1, float(fallback))

    w_smart = _historical_weight("smart_money", float(role_weights.get("smart_money", 1.30)))
    w_breakout = _historical_weight("breakout_trader", float(role_weights.get("breakout_trader", 1.20)))
    w_risk = _historical_weight("risk_manager", float(role_weights.get("risk_manager", 1.40)))
    w_contra = _historical_weight("contrarian", float(role_weights.get("contrarian", 1.10)))

    return [
        {
            "agent": "Smart Whale",
            "role": "smart_money",
            "weight": w_smart,
            "probability": smart_money_prob,
            "reasoning": _smart_whale_reasoning(acc, oi, fstress, trend_flow),
        },
        {
            "agent": "Breakout Trader",
            "role": "breakout_trader",
            "weight": w_breakout,
            "probability": breakout_prob,
            "reasoning": _breakout_trader_reasoning(bo, momentum, false_bo),
        },
        {
            "agent": "Risk Manager",
            "role": "risk_manager",
            "weight": w_risk,
            "probability": risk_prob,
            "reasoning": _risk_manager_reasoning(fstress, liq, false_bo, acc),
        },
        {
            "agent": "Contrarian",
            "role": "contrarian",
            "weight": w_contra,
            "probability": contrarian_prob,
            "reasoning": _contrarian_reasoning(bo, false_bo, fstress, acc),
        },
    ]


def weighted_vote(agent_votes: list[dict[str, Any]], disagreement_threshold: float = 0.30) -> dict[str, Any]:
    if not agent_votes:
        return {"final_probability": 0.5, "confidence": 0.0, "disagreement": 1.0, "agents_disagree": True}

    weighted_sum = 0.0
    total_weight = 0.0
    probs: list[float] = []
    for v in agent_votes:
        p = _clamp01(float(v.get("probability", 0.5)))
        w = max(0.0, float(v.get("weight", 1.0)))
        weighted_sum += p * w
        total_weight += w
        probs.append(p)

    final_prob = (weighted_sum / total_weight) if total_weight > 1e-12 else 0.5
    disagreement = max(probs) - min(probs)
    agents_disagree = disagreement > disagreement_threshold
    confidence = _clamp01(1.0 - disagreement)
    if agents_disagree:
        confidence = _clamp01(confidence * 0.75)

    return {
        "final_probability": round(final_prob, 3),
        "confidence": round(confidence, 3),
        "disagreement": round(disagreement, 3),
        "agents_disagree": agents_disagree,
    }

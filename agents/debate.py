"""
Local Debate Engine — rule-based дебаты агентов без LLM.

Философия:
  Агенты слышат аргументы друг друга и корректируют свои вероятности.
  Это не голосование — это интерпретация: каждый агент может изменить
  позицию на основе аргументов коллег.

  Не требует LLM. Детерминирован. Быстр.

Правила дебатов (один раунд):
  Smart Whale   → слушает Contrarian: если тот сигналит ловушку, осторожнее
  Breakout      → слушает Risk Manager: опасный рынок → снижает conviction
  Contrarian    → слушает Smart Whale: если накопление реально, смягчает скепсис
  Risk Manager  → усиливает дисконт при максимальном риске в группе

Возвращает обновлённый список голосов + debate_log.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from markets.market_engine import EdgeZone

_MAX_DELTA = 0.12  # максимальный сдвиг за один раунд дебатов


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def run_local_debate(
    votes: list[dict[str, Any]],
    zone: "EdgeZone",
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Один раунд rule-based дебатов.

    Parameters
    ----------
    votes : list[dict]
        Список голосов из build_role_votes.
    zone : EdgeZone
        Текущий рыночный контекст.

    Returns
    -------
    updated_votes : list[dict]
        Голоса с обновлёнными probability и reasoning.
    debate_log : list[str]
        Лог событий дебатов (для observability).
    """
    probs: dict[str, float] = {v["role"]: float(v.get("probability", 0.5)) for v in votes}
    log: list[str] = []

    updated: list[dict[str, Any]] = []
    for v in votes:
        role = v["role"]
        p = float(v.get("probability", 0.5))
        note = ""

        if role == "smart_money":
            contra_p = probs.get("contrarian", 0.5)
            # Contrarian видит ловушку (низкая вероятность → медвежий сигнал)
            if contra_p < 0.38:
                strength = (0.38 - contra_p) / 0.38  # [0, 1]
                delta = -_MAX_DELTA * 0.5 * strength
                p = _clamp01(p + delta)
                note = f"Contrarian предупреждает о ловушке ({contra_p:.2f}) → -{ abs(delta):.3f}"
                log.append(f"[debate] Smart Whale услышал Contrarian: {note}")

        elif role == "breakout_trader":
            risk_p = probs.get("risk_manager", 0.5)
            # Risk Manager сигналит опасность
            if risk_p < 0.42:
                strength = (0.42 - risk_p) / 0.42
                delta = -_MAX_DELTA * 0.6 * strength
                p = _clamp01(p + delta)
                note = f"Risk Manager видит опасность ({risk_p:.2f}) → -{abs(delta):.3f}"
                log.append(f"[debate] Breakout услышал Risk Manager: {note}")

        elif role == "contrarian":
            smart_p = probs.get("smart_money", 0.5)
            # Smart Whale видит накопление → контрарий смягчает скепсис
            if smart_p > 0.62 and zone.scen_accumulation > 0.35:
                strength = min(1.0, (smart_p - 0.62) / 0.38)
                delta = _MAX_DELTA * 0.4 * strength
                p = _clamp01(p + delta)
                note = (
                    f"Smart Whale видит накопление ({zone.scen_accumulation:.2f},"
                    f" p={smart_p:.2f}) → +{delta:.3f}"
                )
                log.append(f"[debate] Contrarian смягчил позицию: {note}")

        elif role == "risk_manager":
            # Risk Manager учитывает максимальный риск всей группы
            group_max_risk = max(zone.funding_stress, zone.liq_spike, zone.scen_breakout_suspicious)
            if group_max_risk > 0.5:
                delta = -_MAX_DELTA * 0.35 * (group_max_risk - 0.5) / 0.5
                p = _clamp01(p + delta)
                note = f"max_risk в группе={group_max_risk:.2f} → -{abs(delta):.3f}"
                log.append(f"[debate] Risk Manager усилил дисконт: {note}")

        reasoning = v.get("reasoning", "")
        if note:
            reasoning = f"{reasoning} || [дебаты] {note}"

        updated.append({**v, "probability": round(p, 3), "reasoning": reasoning})

    if not log:
        log.append("[debate] Все агенты остались при своих позициях (консенсус)")

    return updated, log

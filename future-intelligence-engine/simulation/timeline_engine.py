"""Timeline Engine — builds a full event timeline: scenarios → causal chain → forecast steps."""

from __future__ import annotations

from datetime import datetime, timedelta

import ollama

from simulation.scenario_engine import generate_scenarios
from simulation.causal_engine import generate_causal_chain

MODEL = "llama3"


def _generate_timeline_llm(event: str, scenario: str) -> list[dict]:
    """
    Asks LLM to produce a sequence of dated future developments
    for a given event and scenario.
    """
    prompt = f"""
You are a financial and geopolitical forecasting AI.

Event: {event}
Scenario: {scenario}

Generate a timeline of 4 key developments that would unfold over the next 30 days.
Each development should follow logically from the previous one.

Format each line exactly as:
Day N: <development>

Only output the 4 lines. No extra text.
"""
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response["message"]["content"]
        steps = []
        base = datetime.utcnow()
        for line in text.split("\n"):
            line = line.strip()
            if not line or not line.lower().startswith("day"):
                continue
            try:
                day_part, desc = line.split(":", 1)
                day_num = int("".join(c for c in day_part if c.isdigit()) or "7")
            except ValueError:
                day_num = len(steps) * 7 + 7
                desc = line
            steps.append({
                "day":         day_num,
                "date":        (base + timedelta(days=day_num)).strftime("%Y-%m-%d"),
                "development": desc.strip(),
            })
        return steps[:4]
    except Exception as exc:
        print(f"[timeline_engine] LLM error: {exc}")
        return []


def _fallback_steps(event: str, scenario: str) -> list[dict]:
    """Deterministic timeline used when LLM is unavailable."""
    base = datetime.utcnow()
    templates = [
        (1,  "Initial market reaction to {event}."),
        (7,  "Analysts publish first reports on scenario: {scenario}."),
        (14, "Institutional investors adjust positions based on new data."),
        (30, "Long-term impact of {event} becomes visible in market structure."),
    ]
    return [
        {
            "day":         day,
            "date":        (base + timedelta(days=day)).strftime("%Y-%m-%d"),
            "development": text.format(event=event, scenario=scenario),
        }
        for day, text in templates
    ]


def build_timeline(event: str, use_llm: bool = True) -> dict:
    """
    Builds a complete event timeline.

    Returns:
        {
            "event":         str,
            "generated_at":  str,
            "scenarios":     list[str],
            "causal_chain":  list[str],
            "timelines": [
                {
                    "scenario": str,
                    "steps": [{"day": int, "date": str, "development": str}, ...]
                },
                ...
            ]
        }
    """
    scenarios   = generate_scenarios(event)
    causal_chain = generate_causal_chain(event)

    timelines = []
    for scenario in scenarios:
        if use_llm:
            steps = _generate_timeline_llm(event, scenario)
        else:
            steps = []

        if not steps:
            steps = _fallback_steps(event, scenario)

        timelines.append({"scenario": scenario, "steps": steps})

    return {
        "event":        event,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "scenarios":    scenarios,
        "causal_chain": causal_chain,
        "timelines":    timelines,
    }


def generate_timeline(scenario: str, use_llm: bool = True) -> list[str]:
    """
    Simplified interface: takes a scenario string, returns a flat list
    of 4 step descriptions suitable for per-step agent simulation.
    """
    base = datetime.utcnow()
    if use_llm:
        steps = _generate_timeline_llm(scenario, scenario)
    else:
        steps = []

    if not steps:
        steps = _fallback_steps(scenario, scenario)

    return [
        f"Step {i + 1} [Day {s['day']}, {s['date']}]: {s['development']}"
        for i, s in enumerate(steps)
    ]


def print_timeline(timeline: dict) -> None:
    """Pretty-prints a timeline dict to stdout."""
    print(f"\n{'='*60}")
    print(f"EVENT: {timeline['event']}")
    print(f"Generated: {timeline['generated_at']}")

    if timeline["causal_chain"]:
        print(f"\n── Causal Chain ──")
        for step in timeline["causal_chain"]:
            print(f"  {step}")

    for block in timeline["timelines"]:
        print(f"\n── {block['scenario']} ──")
        for step in block["steps"]:
            print(f"  [{step['date']}] Day {step['day']:>2}: {step['development']}")

    print(f"{'='*60}\n")

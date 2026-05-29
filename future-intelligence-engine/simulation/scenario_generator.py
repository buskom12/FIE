import requests


def generate_scenarios_simple(event: str) -> list[str]:
    """MVP fallback — returns three deterministic scenarios without LLM."""
    return [
        f"{event} leads to strong positive outcome",
        f"{event} leads to negative market reaction",
        f"{event} has neutral impact",
    ]


def generate_scenarios(event: str) -> list[str]:
    prompt = f"""
Generate 3 possible future scenarios for this event:

Event:
{event}

Return short descriptions.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        text = response.json()["response"]
    except requests.exceptions.ConnectionError:
        print("[scenario_generator] Ollama not available — using simple scenarios.")
        return generate_scenarios_simple(event)
    except Exception as e:
        print(f"[scenario_generator] Error: {e}")
        return []

    scenarios = [s.strip() for s in text.split("\n") if s.strip()]
    return scenarios[:3]

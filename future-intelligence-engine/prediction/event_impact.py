import requests


def score_event_impact(event: str) -> float:
    prompt = f"""
Rate the impact of this event on global crypto markets.

Event:
{event}

Return a number between 0 and 1.
Only return the number.
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
        text = response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        print("[event_impact] Ollama not available — returning default score 0.5")
        return 0.5
    except Exception as e:
        print(f"[event_impact] Error: {e}")
        return 0.5

    try:
        score = float(text)
    except ValueError:
        score = 0.5

    return max(0.0, min(1.0, score))

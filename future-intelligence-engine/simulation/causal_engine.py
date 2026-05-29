import requests


def generate_causal_chain(event: str) -> list[str]:
    prompt = f"""
Create a causal chain of consequences for this event.

Event:
{event}

Return 4 steps:

event
↓
effect 1
↓
effect 2
↓
effect 3
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
        print("[causal_engine] Ollama not available — skipping causal chain.")
        return []
    except Exception as e:
        print(f"[causal_engine] Error: {e}")
        return []

    steps = [s.strip() for s in text.split("\n") if s.strip()]
    return steps

import re
import requests
import json
from agents.personas import AgentPersona
from memory.society_memory import SocietyMemory


class LLMAgent:

    def __init__(self, persona: AgentPersona, memory: SocietyMemory = None):
        self.persona = persona
        self.memory = memory
        self.last_prediction: float | None = None

    def evaluate_event(self, event: str):

        prompt = f"""
You are a {self.persona.role}.

Event:
{event}

Estimate probability (0-1) that this event happens.

Return JSON format:

{{
"probability": number,
"reasoning": "short explanation"
}}
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )

        output = response.json()["response"]

        try:
            match = re.search(r'\{.*?"probability".*?\}', output, re.DOTALL)
            data = json.loads(match.group()) if match else {}
            probability = float(data.get("probability", 0.5))
            reasoning = str(data.get("reasoning", output))
        except Exception:
            probability = 0.5
            reasoning = output

        self.last_prediction = probability

        return {
            "agent": self.persona.name,
            "role": self.persona.role,
            "probability": probability,
            "reasoning": reasoning,
            "weight": self.persona.weight,
        }

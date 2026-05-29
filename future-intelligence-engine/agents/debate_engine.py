import re
import json
import requests


def run_debate(agent_results: list, event: str) -> list:
    """
    Each agent sees the opinions of others and revises their estimate.
    Returns updated list of results with post-debate probabilities.
    """

    updated_results = []

    for i, agent in enumerate(agent_results):

        other_opinions = [
            f"- {r['agent']} ({r['role']}): probability {r['probability']}, reasoning: {r['reasoning'][:100]}"
            for j, r in enumerate(agent_results)
            if j != i
        ]

        debate_prompt = f"""
You are {agent['agent']}, a {agent['role']}.

Event: {event}

Your initial estimate was: probability {agent['probability']}
Your reasoning: {agent['reasoning'][:150]}

Other agents have shared their opinions:
{chr(10).join(other_opinions)}

After considering their perspectives, revise your probability estimate.

Return JSON:
{{
"probability": number between 0 and 1,
"reasoning": "updated short explanation"
}}
"""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": debate_prompt,
                    "stream": False
                }
            )

            output = response.json()["response"]
            match = re.search(r'\{.*?"probability".*?\}', output, re.DOTALL)
            data = json.loads(match.group()) if match else {}
            probability = float(data.get("probability", agent["probability"]))
            reasoning = str(data.get("reasoning", output))

        except Exception:
            probability = agent["probability"]
            reasoning = agent["reasoning"]

        updated_results.append({
            "agent": agent["agent"],
            "role": agent["role"],
            "probability": probability,
            "reasoning": reasoning,
        })

    return updated_results

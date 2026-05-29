"""Agent persona definitions and persona registry."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class AgentPersona:
    name: str
    role: str
    description: str
    risk_tolerance: float  # 0.0 (консервативный) — 1.0 (агрессивный)
    weight: float = 1.0
    accuracy: float = 0.5
    predictions_count: int = 0
    traits: list[str] = field(default_factory=list)

    def system_prompt(self) -> str:
        """Генерирует system prompt для LLM на основе персоны."""
        traits_text = ", ".join(self.traits) if self.traits else "no special traits"
        return (
            f"You are {self.name}, a {self.role}.\n"
            f"{self.description}\n"
            f"Your risk tolerance is {self.risk_tolerance:.0%} "
            f"(0% = very conservative, 100% = very aggressive).\n"
            f"Key traits: {traits_text}."
        )


# === Готовые персоны ===

CRYPTO_WHALE = AgentPersona(
    name="Whale",
    role="crypto whale",
    description=(
        "A large-scale market participant who moves significant capital. "
        "Focuses on on-chain data, liquidity pools, and market microstructure. "
        "Prioritizes asymmetric risk/reward opportunities."
    ),
    risk_tolerance=0.75,
    traits=["contrarian", "on-chain analyst", "liquidity hunter", "patient accumulator"],
)

MACRO_ECONOMIST = AgentPersona(
    name="Macro",
    role="macro economist",
    description=(
        "A top-down analyst who evaluates macroeconomic cycles, central bank policy, "
        "inflation, and global capital flows. Connects macro trends to crypto markets."
    ),
    risk_tolerance=0.40,
    traits=["data-driven", "long-horizon thinker", "central bank watcher", "cycle analyst"],
)

RETAIL_TRADER = AgentPersona(
    name="Retail",
    role="retail trader",
    description=(
        "A sentiment-driven market participant who follows trends, social media signals, "
        "and technical patterns. Represents the collective behavior of the retail crowd."
    ),
    risk_tolerance=0.85,
    traits=["trend follower", "FOMO-sensitive", "social sentiment reader", "momentum trader"],
)


# === Реестр всех персон ===

PERSONA_REGISTRY: Dict[str, AgentPersona] = {
    "crypto_whale": CRYPTO_WHALE,
    "macro_economist": MACRO_ECONOMIST,
    "retail_trader": RETAIL_TRADER,
}


def get_persona(key: str) -> AgentPersona:
    """Возвращает персону по ключу из реестра."""
    if key not in PERSONA_REGISTRY:
        available = ", ".join(PERSONA_REGISTRY.keys())
        raise ValueError(f"Unknown persona '{key}'. Available: {available}")
    return PERSONA_REGISTRY[key]

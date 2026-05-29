"""SwarmFactory — фабрика для создания конфигурированных роёв агентов."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from agents.llm_agent import LLMAgent
from agents.personas import AgentPersona, PERSONA_REGISTRY
from agents.persona_generator import generate_swarm  # noqa: F401 — re-export
from memory.society_memory import SocietyMemory


# ---------------------------------------------------------------------------
# Swarm presets
# ---------------------------------------------------------------------------

@dataclass
class SwarmConfig:
    """Описание состава роя: сколько агентов каждой роли и с какими весами."""

    name: str
    description: str
    # role -> (count, weight_range)
    composition: dict[str, tuple[int, tuple[float, float]]] = field(default_factory=dict)

    def total_agents(self) -> int:
        return sum(count for count, _ in self.composition.values())


# Встроенные конфигурации роёв
_SWARM_PRESETS: dict[str, SwarmConfig] = {
    "crypto_experts": SwarmConfig(
        name="crypto_experts",
        description="Рой крипто-участников: киты, ритейл, маркет-мейкеры, геймблеры",
        composition={
            "crypto whale":  (10, (1.2, 2.0)),
            "retail trader": (20, (0.5, 1.0)),
            "market maker":  (5,  (1.0, 1.5)),
            "gambler":       (5,  (0.3, 0.8)),
            "developer":     (10, (0.8, 1.4)),
        },
    ),
    "macro_analysts": SwarmConfig(
        name="macro_analysts",
        description="Рой макроаналитиков: экономисты, политики, банкиры, венчур",
        composition={
            "macro economist":   (10, (1.2, 1.8)),
            "central banker":    (5,  (1.3, 2.0)),
            "political analyst": (5,  (0.9, 1.4)),
            "venture capitalist":(10, (0.9, 1.5)),
            "hedge fund manager":(5,  (1.1, 1.8)),
            "journalist":        (5,  (0.5, 1.0)),
            "risk analyst":      (10, (1.0, 1.6)),
        },
    ),
    "tech_innovators": SwarmConfig(
        name="tech_innovators",
        description="Рой технологических новаторов: AI-исследователи, стартаперы, девелоперы",
        composition={
            "AI researcher":    (15, (1.0, 1.8)),
            "startup founder":  (10, (0.9, 1.5)),
            "developer":        (15, (0.7, 1.3)),
            "venture capitalist":(10, (0.9, 1.5)),
        },
    ),
    "diverse": SwarmConfig(
        name="diverse",
        description="Смешанный рой из всех 14 ролей с равными весами",
        composition={
            "crypto whale":      (5,  (0.8, 1.5)),
            "retail trader":     (5,  (0.5, 1.0)),
            "macro economist":   (5,  (0.9, 1.4)),
            "journalist":        (5,  (0.5, 1.0)),
            "hedge fund manager":(5,  (1.1, 1.8)),
            "venture capitalist":(5,  (0.9, 1.5)),
            "political analyst": (5,  (0.8, 1.3)),
            "central banker":    (5,  (1.2, 2.0)),
            "startup founder":   (5,  (0.8, 1.4)),
            "developer":         (5,  (0.7, 1.3)),
            "AI researcher":     (5,  (1.0, 1.8)),
            "gambler":           (5,  (0.3, 0.8)),
            "market maker":      (5,  (0.9, 1.5)),
            "risk analyst":      (5,  (1.0, 1.6)),
        },
    ),
}


# ---------------------------------------------------------------------------
# SwarmFactory
# ---------------------------------------------------------------------------

class SwarmFactory:
    """
    Фабрика роёв агентов.

    Создаёт списки :class:`LLMAgent` по заданной конфигурации или пресету.
    Поддерживает кастомные композиции и произвольные функции-трансформеры персон.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self.memory = SocietyMemory()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def list_presets(cls) -> list[str]:
        """Возвращает названия всех встроенных конфигураций."""
        return list(_SWARM_PRESETS.keys())

    def build(
        self,
        preset: str = "diverse",
        *,
        persona_transformer: Callable[[AgentPersona], AgentPersona] | None = None,
    ) -> list[LLMAgent]:
        """
        Собирает рой по имени пресета.

        Args:
            preset: Ключ встроенной конфигурации (см. :meth:`list_presets`).
            persona_transformer: Опциональный коллбэк для постобработки каждой персоны
                перед созданием агента (например, для буста весов или изменения трейтов).

        Returns:
            Список готовых :class:`LLMAgent`.
        """
        if preset not in _SWARM_PRESETS:
            available = ", ".join(_SWARM_PRESETS)
            raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

        config = _SWARM_PRESETS[preset]
        return self._build_from_config(config, persona_transformer)

    def build_custom(
        self,
        composition: dict[str, int | tuple[int, tuple[float, float]]],
        *,
        name: str = "custom",
        persona_transformer: Callable[[AgentPersona], AgentPersona] | None = None,
    ) -> list[LLMAgent]:
        """
        Собирает рой по произвольной композиции.

        Args:
            composition: Словарь ``role -> count`` или ``role -> (count, (w_min, w_max))``.
            name: Имя для конфигурации (используется в логах).
            persona_transformer: Опциональный постпроцессор персон.

        Returns:
            Список готовых :class:`LLMAgent`.

        Example::

            factory = SwarmFactory(seed=42)
            agents = factory.build_custom({
                "quant trader": (8, (1.0, 2.0)),
                "macro economist": 5,
            })
        """
        normalized: dict[str, tuple[int, tuple[float, float]]] = {}
        for role, spec in composition.items():
            if isinstance(spec, int):
                normalized[role] = (spec, (0.8, 1.2))
            else:
                normalized[role] = spec

        config = SwarmConfig(name=name, description="custom swarm", composition=normalized)
        return self._build_from_config(config, persona_transformer)

    def build_from_registry(
        self,
        keys: list[str],
        *,
        persona_transformer: Callable[[AgentPersona], AgentPersona] | None = None,
    ) -> list[LLMAgent]:
        """
        Создаёт по одному агенту для каждого ключа из :data:`PERSONA_REGISTRY`.

        Args:
            keys: Список ключей реестра (например, ``["crypto_whale", "macro_economist"]``).
            persona_transformer: Опциональный постпроцессор персон.

        Returns:
            Список готовых :class:`LLMAgent`.
        """
        agents: list[LLMAgent] = []
        for key in keys:
            if key not in PERSONA_REGISTRY:
                available = ", ".join(PERSONA_REGISTRY)
                raise ValueError(f"Unknown persona key '{key}'. Available: {available}")
            persona = PERSONA_REGISTRY[key]
            if persona_transformer:
                persona = persona_transformer(persona)
            agents.append(LLMAgent(persona, self.memory))
        return agents

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_from_config(
        self,
        config: SwarmConfig,
        persona_transformer: Callable[[AgentPersona], AgentPersona] | None,
    ) -> list[LLMAgent]:
        agents: list[LLMAgent] = []
        agent_idx = 0

        for role, (count, (w_min, w_max)) in config.composition.items():
            for _ in range(count):
                persona = AgentPersona(
                    name=f"{role.replace(' ', '_')}_{agent_idx}",
                    role=role,
                    description=f"{role} analyzing markets",
                    risk_tolerance=self._rng.uniform(0.2, 0.9),
                    weight=self._rng.uniform(w_min, w_max),
                )
                if persona_transformer:
                    persona = persona_transformer(persona)
                agents.append(LLMAgent(persona, self.memory))
                agent_idx += 1

        return agents

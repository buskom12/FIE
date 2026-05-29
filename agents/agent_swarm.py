from __future__ import annotations

import sys
import os
import importlib.util
from pathlib import Path

# В этом репозитории есть ДВА пакета с именем `agents`:
# - ./agents (текущий, wrapper слой)
# - ./future-intelligence-engine/agents (ядро FIE)
# Обычный `import agents.agent_manager` конфликтует.
# Поэтому загружаем AgentManager напрямую из файла ядра.
_FIE_ROOT = Path(__file__).resolve().parent.parent / "future-intelligence-engine"
_AGENT_MANAGER_PATH = _FIE_ROOT / "agents" / "agent_manager.py"
if not _AGENT_MANAGER_PATH.is_file():
    raise FileNotFoundError(f"Не найден AgentManager: {_AGENT_MANAGER_PATH}")

_spec = importlib.util.spec_from_file_location("fie_core_agent_manager", str(_AGENT_MANAGER_PATH))
if _spec is None or _spec.loader is None:
    raise ImportError(f"Не удалось загрузить модуль: {_AGENT_MANAGER_PATH}")

_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
AgentManager = getattr(_mod, "AgentManager")

_manager: AgentManager | None = None


def _get_manager() -> AgentManager:
    global _manager
    if _manager is None:
        _manager = AgentManager()
    return _manager


def run_swarm(event: str) -> list[dict]:
    """Запускает рой агентов и возвращает список оценок по событию."""
    return _get_manager().evaluate_event(event)

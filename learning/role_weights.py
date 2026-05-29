from __future__ import annotations

import json
from pathlib import Path
from typing import Any

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "data" / "logs" / "role_weights.json"
ROLE_STATS_PATH = Path(__file__).resolve().parent.parent / "data" / "logs" / "role_stats.json"

DEFAULT_ROLE_WEIGHTS: dict[str, float] = {
    "smart_money": 1.30,
    "breakout_trader": 1.20,
    "risk_manager": 1.40,
    "contrarian": 1.10,
}


def load_role_weights() -> dict[str, float]:
    if not WEIGHTS_PATH.is_file():
        return dict(DEFAULT_ROLE_WEIGHTS)
    try:
        data = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_ROLE_WEIGHTS)
    out = dict(DEFAULT_ROLE_WEIGHTS)
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                out[str(k)] = max(0.1, min(5.0, float(v)))
            except Exception:
                continue
    return out


def save_role_weights(weights: dict[str, Any]) -> None:
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean: dict[str, float] = {}
    for k, v in weights.items():
        try:
            clean[str(k)] = max(0.1, min(5.0, float(v)))
        except Exception:
            continue
    WEIGHTS_PATH.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")


def load_role_stats() -> dict[str, dict[str, float]]:
    if not ROLE_STATS_PATH.is_file():
        return {}
    try:
        data = json.loads(ROLE_STATS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, dict[str, float]] = {}
    if not isinstance(data, dict):
        return out
    for role, stats in data.items():
        if not isinstance(stats, dict):
            continue
        try:
            accuracy = max(0.0, min(1.0, float(stats.get("accuracy", 0.5))))
            count = max(0.0, float(stats.get("count", 0.0)))
        except Exception:
            continue
        out[str(role)] = {"accuracy": accuracy, "count": count}
    return out


def save_role_stats(stats: dict[str, Any]) -> None:
    ROLE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean: dict[str, dict[str, float]] = {}
    for role, raw in stats.items():
        if not isinstance(raw, dict):
            continue
        try:
            accuracy = max(0.0, min(1.0, float(raw.get("accuracy", 0.5))))
            count = max(0.0, float(raw.get("count", 0.0)))
        except Exception:
            continue
        clean[str(role)] = {"accuracy": accuracy, "count": count}
    ROLE_STATS_PATH.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")

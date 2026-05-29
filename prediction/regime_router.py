"""
Regime Router — диспетчер моделей/стратегий по режиму рынка.

Философия:
  Мы предсказываем ТОЛЬКО там, где есть доказанный edge.
  Для каждого режима — своя модель/стратегия.
  Всё остальное → skip (не угадывать).

Поддерживаемые режимы:
  trend_low  → trend_low_vol_model  (PatternEngine + role agents — максимальный edge)
  (остальные → None → skip)

Когда появятся новые стратегии — добавить сюда запись в REGIME_MODELS,
не трогая autonomous_loop.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from markets.market_engine import EdgeZone

# Таблица routing: regime_key → model_id
# trend_low = trend + low volatility (единственная zone с edge)
REGIME_MODELS: dict[str, str] = {
    "trend_low": "trend_low_vol_model",
}


def route_model(zone: "EdgeZone") -> str | None:
    """
    Возвращает model_id для данного EdgeZone.
    None → skip (нет edge, не предсказываем).

    Логика:
      1. allow_prediction должен быть True (пройден gate в market_engine)
      2. regime_key должен быть в REGIME_MODELS
    """
    if not zone.allow_prediction:
        return None

    # regime_key вида "trend_low" или "trend_high" и т.д.
    key = f"{zone.regime}_{zone.volatility}"
    return REGIME_MODELS.get(key)


def describe_regime(zone: "EdgeZone") -> str:
    """Человекочитаемое описание текущего режима для логов."""
    model = route_model(zone)
    if model is None:
        return f"no_edge | regime={zone.regime_key} | allow={zone.allow_prediction} | reason={zone.reason}"
    return (
        f"edge_zone | model={model} | regime={zone.regime_key}"
        f" | state={zone.state} | gate={zone.gate_score:.3f}"
        f" | acc={zone.scen_accumulation:.3f} | trend_flow={zone.scen_trend_flow:.3f}"
    )

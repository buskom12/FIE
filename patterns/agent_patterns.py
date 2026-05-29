"""
Агентные архетипы для PatternEngine.

Каждый архетип — отдельная модель мышления с уникальным набором
сигналов-приоритетов и методом вычисления вероятности.

Архетипы:
    SmartWhaleAgent  — следит за крупными игроками (on-chain)
    QuantModelAgent  — статистика + паттерны, взвешивает по силе сигнала
    MacroAnalyst     — экономические/макро-сигналы
    GamblerAgent     — случайный шум (baseline / контрольная группа)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Базовый класс
# ---------------------------------------------------------------------------

class BaseAgentPattern(ABC):
    """Базовый архетип агента для PatternEngine."""

    name: str = "base"
    weight: float = 1.0

    @abstractmethod
    def score(self, signals: list[dict]) -> float:
        """
        Возвращает вероятность бычьего исхода [0.0, 1.0]
        на основе входных сигналов.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weight={self.weight})"


# ---------------------------------------------------------------------------
# Smart Whale — крупные игроки и on-chain активность
# ---------------------------------------------------------------------------

class SmartWhaleAgent(BaseAgentPattern):
    """
    Следит за движениями крупного капитала.
    Высокий вес whale_buy / whale_sell, exchange flows.
    """

    name = "smart_whale"
    weight = 1.5  # исторически выше среднего edge

    _BULLISH = {"whale_buy", "exchange_outflow", "spot_premium", "open_interest_spike"}
    _BEARISH = {"whale_sell", "exchange_inflow", "futures_premium", "open_interest_drop"}
    _SIGNAL_WEIGHT = 2.0  # on-chain сигналы считаются вдвойне

    def score(self, signals: list[dict]) -> float:
        types = {s["type"] for s in signals}

        bull = sum(self._SIGNAL_WEIGHT for t in types if t in self._BULLISH)
        bear = sum(self._SIGNAL_WEIGHT for t in types if t in self._BEARISH)

        # Остальные сигналы с обычным весом
        generic_bull = sum(1.0 for t in types
                           if t not in self._BULLISH and t not in self._BEARISH
                           and "buy" in t or "oversold" in t or "support" in t)
        generic_bear = sum(1.0 for t in types
                           if t not in self._BULLISH and t not in self._BEARISH
                           and "sell" in t or "overbought" in t or "resist" in t)

        total = bull + bear + generic_bull + generic_bear
        if total == 0:
            return 0.5

        return (bull + generic_bull) / total


# ---------------------------------------------------------------------------
# Quant Model — статистика и паттерны
# ---------------------------------------------------------------------------

class QuantModelAgent(BaseAgentPattern):
    """
    Работает с техническими паттернами и статистикой.
    Учитывает RSI, MACD, volume. Взвешивает по количеству совпадений.
    """

    name = "quant_model"
    weight = 1.3

    _SIGNAL_SCORES = {
        # Бычьи технические сигналы (положительный вклад)
        "rsi_oversold": +0.8,
        "macd_crossover_up": +0.9,
        "support_bounce": +0.7,
        "high_volume": +0.5,
        "funding_negative": +0.6,
        # Медвежьи технические сигналы (отрицательный вклад)
        "rsi_overbought": -0.8,
        "macd_crossover_down": -0.9,
        "resistance_reject": -0.7,
        "low_volume": -0.3,
        "funding_positive": -0.6,
        # Нейтральные
        "rsi_neutral": 0.0,
        "sideways_volume": 0.0,
        "consolidation": 0.0,
    }

    def score(self, signals: list[dict]) -> float:
        raw = sum(self._SIGNAL_SCORES.get(s["type"], 0.0) for s in signals)
        # Нормализуем в [0, 1] через сигмоид
        import math
        return 1 / (1 + math.exp(-raw))


# ---------------------------------------------------------------------------
# Macro Analyst — макроэкономика
# ---------------------------------------------------------------------------

class MacroAnalystAgent(BaseAgentPattern):
    """
    Оценивает макро-контекст: страх/жадность, ставки, глобальный ликвидность.
    Консервативен — даёт сдержанные вероятности.
    """

    name = "macro_analyst"
    weight = 1.2

    _RISK_ON = {"fear_greed_low", "open_interest_spike", "spot_premium"}
    _RISK_OFF = {"fear_greed_high", "open_interest_drop", "futures_premium"}

    def score(self, signals: list[dict]) -> float:
        types = {s["type"] for s in signals}

        risk_on = sum(1 for t in types if t in self._RISK_ON)
        risk_off = sum(1 for t in types if t in self._RISK_OFF)

        # Добавляем вклад остальных сигналов с пониженным весом
        for s in signals:
            t = s["type"]
            if t in ("whale_buy", "high_volume", "rsi_oversold"):
                risk_on += 0.5
            elif t in ("whale_sell", "low_volume", "rsi_overbought"):
                risk_off += 0.5

        total = risk_on + risk_off
        if total == 0:
            return 0.5

        # Macro-агент консерватор: сжимаем к центру
        raw = risk_on / total
        return 0.5 + (raw - 0.5) * 0.7  # сжатие к 0.5


# ---------------------------------------------------------------------------
# Gambler — случайный шум (baseline)
# ---------------------------------------------------------------------------

class GamblerAgent(BaseAgentPattern):
    """
    Случайный агент — шум / baseline.
    Нужен, чтобы отделить signal от noise в агрегации.
    Вес намеренно низкий.
    """

    name = "gambler"
    weight = 0.3  # минимальный вклад

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def score(self, signals: list[dict]) -> float:
        return self._rng.uniform(0.1, 0.9)


# ---------------------------------------------------------------------------
# Фабрика архетипов
# ---------------------------------------------------------------------------

AGENT_ARCHETYPES: list[BaseAgentPattern] = [
    SmartWhaleAgent(),
    QuantModelAgent(),
    MacroAnalystAgent(),
    GamblerAgent(seed=42),
]


def get_archetype_score(signals: list[dict]) -> float:
    """
    Взвешенное голосование всех 4 архетипов.
    Используется как дополнительный feature для PatternEngine.
    """
    total_weight = sum(a.weight for a in AGENT_ARCHETYPES)
    weighted_sum = sum(a.score(signals) * a.weight for a in AGENT_ARCHETYPES)
    return weighted_sum / total_weight

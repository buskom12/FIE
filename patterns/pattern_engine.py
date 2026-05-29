"""
PatternEngine — сопоставление сигналов с историческими паттернами.

Философия: система знает, когда НЕ знает.
    - мало данных → None, а не 0.5
    - слабые кейсы → прижимаются к prior
    - сильные кейсы → остаются сильными

Три уровня регуляризации:
    1. Minimum support  — меньше MIN_MATCHES совпадений → None (нет ответа)
    2. Laplace smoothing — (pos + 1) / (n + 2), убирает экстремумы 0% и 100%
    3. Uncertainty penalty — чем меньше данных, тем ближе результат к prior
"""

from __future__ import annotations

from dataclasses import dataclass

from signals.interactions import InteractionEngine
from signals.weights import (
    CANONICAL_SIGNAL_KEYS,
    compute_signal_weights,
    normalize_signals_to_canonical,
    signal_weights as default_signal_weights,
)

@dataclass
class PatternEngineConfig:
    # Порог Жаккара для совпадения
    similarity_threshold: float = 0.3

    # Уровень 1: минимум совпадений для выдачи ответа
    # Меньше → возвращаем None (система "не знает")
    min_matches: int = 5

    # Уровень 2: Laplace smoothing параметр
    # (positives + laplace_alpha) / (n + laplace_alpha * 2)
    laplace_alpha: float = 1.0

    # Уровень 3: порог уверенности
    # confidence = min(1.0, n / confidence_threshold)
    # При n < confidence_threshold вероятность сжимается к prior
    confidence_threshold: int = 20

    # top_k: ограничение числа совпадений (None = без ограничения)
    top_k: int | None = None

    # Ограничение top-N сигналов для interaction-термов.
    # Критично для защиты от комбинаторного взрыва пар.
    interaction_top_n: int = 6


class PatternEngine:

    def __init__(
        self,
        similarity_threshold: float = 0.3,
        config: PatternEngineConfig | None = None,
    ) -> None:
        if config is not None:
            self._cfg = config
        else:
            self._cfg = PatternEngineConfig(similarity_threshold=similarity_threshold)

        self._cases: list[dict] = []
        self._positive_count: int = 0
        self.signal_weights: dict[str, float] = dict(default_signal_weights)

        # Накопленная статистика для online-обновления весов (Шаг 6)
        # { signal: {"pos": float, "total": float} }
        self._weight_stats: dict[str, dict[str, float]] = {}
        self.interaction_engine = InteractionEngine()

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------

    def add_case(
        self,
        signals: list[dict] | dict[str, float],
        outcome: int,
        market_context: dict | None = None,
    ) -> None:
        canonical_signals = normalize_signals_to_canonical(signals)
        context_features = self._extract_market_regime_features(market_context)
        regime_label = self._get_regime_label(market_context)

        top_signals = self._top_n_signals(canonical_signals)
        canonical = {**canonical_signals, **context_features}

        if isinstance(signals, list):
            base_set = frozenset(s["type"] for s in signals)
        else:
            base_set = frozenset(canonical_signals.keys())
        context_set = frozenset(context_features.keys())
        signal_set = base_set | context_set

        self._cases.append({
            "signal_set": signal_set,
            "canonical_signals": canonical,
            "outcome": outcome,
        })
        self.interaction_engine.update(
            top_signals,
            outcome,
            regime_label=regime_label,
            conditional_signals=canonical_signals,
        )
        if outcome == 1:
            self._positive_count += 1

    def fit_signal_weights(self, dataset: list[dict]) -> dict[str, float]:
        """
        Обучает веса сигналов на train-сете.
        Не использовать test/validation для этого шага (избегаем leakage).
        """
        train_dataset = []
        for case in dataset:
            train_dataset.append({
                # Контекст — фильтр, веса тренируем только по trigger-сигналам.
                "signals": normalize_signals_to_canonical(case.get("signals")),
                "outcome": case.get("outcome", 0),
            })
        self.signal_weights = compute_signal_weights(train_dataset)
        return dict(self.signal_weights)

    def update_signal_weights_ema(self, dataset: list[dict], alpha: float = 0.1) -> dict[str, float]:
        """
        Batch EMA-обновление весов по всему датасету сразу:
            weights = (1 - alpha) * old + alpha * new

        Используй когда нужно переобучить веса по накопленной истории разом.
        Для обновления после каждого кейса — используй update_weights_online.
        """
        old_weights = dict(self.signal_weights)
        new_weights = compute_signal_weights([
            {
                "signals": normalize_signals_to_canonical(case.get("signals")),
                "outcome": case.get("outcome", 0),
            }
            for case in dataset
        ])
        for signal, old_weight in old_weights.items():
            self.signal_weights[signal] = (1.0 - alpha) * old_weight + alpha * new_weights.get(signal, old_weight)
        return dict(self.signal_weights)

    def update_weights_online(self, case: dict, alpha: float = 0.1) -> dict[str, float]:
        """
        Шаг 6 — online EMA обновление весов после одного нового кейса.

        Принцип:
            weights[s] = (1 - alpha) * old + alpha * new_w(s)

        new_w(s) вычисляется из накопленной статистики (_weight_stats),
        которая растёт с каждым вызовом — веса уточняются по мере поступления данных.

        ВАЖНО: вызывать ТОЛЬКО на train-данных, никогда на test/validation.

        Args:
            case:  {"signals": list|dict, "outcome": 0|1}
            alpha: скорость адаптации [0..1]. 0.1 = медленно и стабильно.

        Returns:
            Обновлённый словарь весов (копия).
        """
        try:
            outcome = float(case["outcome"])
        except (KeyError, TypeError, ValueError):
            return dict(self.signal_weights)

        # Контекст — фильтр, для online EMA веса тренируем только по trigger-сигналам.
        canonical = normalize_signals_to_canonical(case.get("signals", {}))

        # Обновляем накопленную статистику
        for s, v in canonical.items():
            if s not in CANONICAL_SIGNAL_KEYS or v <= 0.0:
                continue
            if s not in self._weight_stats:
                self._weight_stats[s] = {"pos": 0.0, "total": 0.0}
            self._weight_stats[s]["total"] += 1.0
            self._weight_stats[s]["pos"] += outcome

        # Пересчитываем веса из накопленной статистики
        # weight = P(outcome=1 | signal) — та же семантика, что в compute_signal_weights
        for s, st in self._weight_stats.items():
            prob = (st["pos"] + 1.0) / (st["total"] + 2.0)   # Laplace smoothing
            old_w = self.signal_weights.get(s, 0.5)           # 0.5 = нейтральный prior
            self.signal_weights[s] = (1.0 - alpha) * old_w + alpha * prob

        return dict(self.signal_weights)

    @property
    def prior(self) -> float:
        """Базовая частота исхода=1 по тренировочным данным."""
        if not self._cases:
            return 0.5
        return self._positive_count / len(self._cases)

    # ------------------------------------------------------------------
    # Поиск паттернов
    # ------------------------------------------------------------------

    def find_patterns(
        self,
        signals: list[dict] | dict[str, float],
        market_context: dict | None = None,
    ) -> list[dict]:
        """Возвращает совпадения по Жаккару, отсортированные по убыванию сходства."""
        if isinstance(signals, list):
            base_set = frozenset(s["type"] for s in signals)
        else:
            base_set = frozenset(normalize_signals_to_canonical(signals).keys())
        context_set = frozenset(self._extract_market_regime_features(market_context).keys())
        query_set = base_set | context_set
        matches = []

        for case in self._cases:
            sim = self._jaccard(query_set, case["signal_set"])
            if sim >= self._cfg.similarity_threshold:
                matches.append({
                    "signal_set": case["signal_set"],
                    "outcome": case["outcome"],
                    "similarity": sim,
                })

        matches.sort(key=lambda m: m["similarity"], reverse=True)

        if self._cfg.top_k is not None:
            matches = matches[: self._cfg.top_k]

        return matches

    # ------------------------------------------------------------------
    # Вероятность + уверенность
    # ------------------------------------------------------------------

    def compute_probability(self, matches: list[dict]) -> float | None:
        """
        Возвращает вероятность исхода=1 или None если данных недостаточно.

        None — явный сигнал системы "я не знаю, лучше пропустить этот кейс".
        Это принципиально отличается от 0.5 ("50/50 с уверенностью").
        """
        n = len(matches)

        # Уровень 1: минимальная поддержка
        if n < self._cfg.min_matches:
            return None

        positives = sum(m["outcome"] for m in matches)
        alpha = self._cfg.laplace_alpha

        # Уровень 2: Laplace smoothing
        # Убирает экстремумы: 1/1 → не 100%, а 66%; 0/1 → не 0%, а 33%
        prob = (positives + alpha) / (n + alpha * 2)

        # Уровень 3: uncertainty penalty
        # Сжимает к prior пропорционально недостатку данных
        prob = self._apply_uncertainty(prob, n)

        return round(prob, 4)

    def compute_result(self, matches: list[dict]) -> dict:
        """
        Возвращает полный результат: вероятность + уверенность + метаданные.

        Использование вместо compute_probability когда нужен confidence score.

        Returns:
            probability : float | None  — вероятность исхода=1 (None = нет данных)
            confidence  : float         — уверенность [0.0, 1.0]
            matches     : int           — количество совпадений
            has_signal  : bool          — есть ли достаточно данных для ответа
        """
        n = len(matches)
        confidence = self.compute_confidence(n)
        prob = self.compute_probability(matches)

        return {
            "probability": prob,
            "confidence": round(confidence, 4),
            "matches": n,
            "has_signal": prob is not None,
        }

    def score_case(
        self,
        signals: list[dict] | dict[str, float],
        weights: dict[str, float] | None = None,
        market_context: dict | None = None,
    ) -> float | None:
        """
        Шаг 4 — оцениваем набор сигналов через обученные веса.

        Семантика весов (Шаг 6):
            weight[s] = P(outcome=1 | signal s присутствует)  ∈ [0.0, 1.0]
            default    = 0.5 (нейтральный prior, нет обучения)

        Формула:
            score = sum(w_s * v_s) / sum(v_s)

        Это взвешенное среднее вероятностей по присутствующим сигналам,
        где v_s = сила сигнала (1.0 для бинарного присутствия).

        Результат:
            0.5  — все сигналы нейтральны (необученные веса или нет сигналов)
            > 0.5 — набор сигналов предсказывает outcome=1
            < 0.5 — набор сигналов предсказывает outcome=0
        """
        canonical_signals = normalize_signals_to_canonical(signals)
        wmap = weights if weights is not None else self.signal_weights

        regime_label = self._get_regime_label(market_context)
        context_confidence = self._context_confidence(market_context)
        context_weight = 0.4 if context_confidence > 0.7 else 0.2

        score = 0.0
        total_strength = 0.0
        for s, v in canonical_signals.items():
            w_uncond = wmap.get(s, 0.5)  # 0.5 = нейтральный prior
            w = w_uncond

            if regime_label is not None:
                w_cond = self.interaction_engine.get_conditional_weight(s, regime_label)
                if w_cond is not None:
                    # Контекст = фильтр: условный вес домешивается с коэффициентом.
                    w = (1.0 - context_weight) * w_uncond + context_weight * w_cond

            score += w * v
            total_strength += v    # делим на сумму сил, а не на сумму весов

        if total_strength == 0:
            return None

        return score / total_strength

    def compute_final_probability(
        self,
        signals: list[dict] | dict[str, float],
        matches: list[dict] | None = None,
        market_context: dict | None = None,
    ) -> dict:
        """
        Шаг 5 — гибрид паттернов и весов.

        Ключевая идея:
            patterns = "память"    → pattern_prob = compute_probability(matches)
            weights  = "интеллект" → signal_score = score_case(signals, weights)

        Комбинация по правилу:
            if pattern_prob is None:
                final_prob = signal_score        # нет исторических данных → только интеллект
            else:
                final_prob = (pattern_prob + signal_score) / 2

        Поле ``source`` фиксирует, что именно вошло в итог:
            "hybrid"       — оба компонента
            "signal_only"  — паттернов нет (pattern_prob is None)
            "pattern_only" — signal_score недоступен
            "none"         — данных нет совсем
        """
        effective_matches = matches if matches is not None else self.find_patterns(signals, market_context=market_context)

        pattern_prob = self.compute_probability(effective_matches)
        signal_score = self.score_case(signals, self.signal_weights, market_context=market_context)

        if pattern_prob is None and signal_score is None:
            final_prob: float | None = None
            source = "none"
        elif pattern_prob is None:
            final_prob = signal_score
            source = "signal_only"
        elif signal_score is None:
            final_prob = pattern_prob
            source = "pattern_only"
        else:
            final_prob = (pattern_prob + signal_score) / 2
            source = "hybrid"

        interaction_prob = self._compute_interaction_probability(signals, market_context=market_context)
        if final_prob is None and interaction_prob is not None:
            final_prob = interaction_prob
            source = "interaction_only"
        elif final_prob is not None and interaction_prob is not None:
            final_prob = (final_prob + interaction_prob) / 2
            source = "hybrid_with_interactions"

        return {
            "pattern_probability": pattern_prob,
            "signal_score": signal_score,
            "interaction_probability": interaction_prob,
            "final_probability": round(final_prob, 4) if final_prob is not None else None,
            "source": source,
            "matches_count": len(effective_matches),
        }

    def compute_confidence(self, n: int) -> float:
        """
        Уверенность системы: min(1.0, n / confidence_threshold).

        n=0   → 0.00 (нет данных)
        n=10  → 0.50 (порог=20)
        n=20+ → 1.00 (полная уверенность)
        """
        return min(1.0, n / self._cfg.confidence_threshold)

    def _apply_uncertainty(self, prob: float, n: int) -> float:
        """
        Уровень 3: uncertainty penalty.
        Чем меньше данных — тем ближе предсказание к prior.

        При n → ∞  : prob остаётся без изменений
        При n = 0  : prob = prior (теоретически, но min_matches это блокирует)
        """
        confidence = self.compute_confidence(n)
        return self.prior + (prob - self.prior) * confidence

    def _top_n_signals(self, canonical: dict[str, float]) -> dict[str, float]:
        """Берём только top-N по силе для расчёта interaction-пар."""
        top_n = max(2, self._cfg.interaction_top_n)
        sorted_items = sorted(
            ((k, v) for k, v in canonical.items() if v > 0.0),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return dict(sorted_items[:top_n])

    def _compute_interaction_probability(
        self,
        signals: list[dict] | dict[str, float],
        market_context: dict | None = None,
    ) -> float | None:
        # Interaction пары считаем только по trigger-сигналам, чтобы не усиливать шум контекста.
        canonical_signals = normalize_signals_to_canonical(signals)
        top_signals = self._top_n_signals(canonical_signals)
        keys = list(top_signals.keys())

        if len(keys) < 2:
            return None

        interaction_probs: list[float] = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                pair = tuple(sorted([keys[i], keys[j]]))
                interaction_prob = self.interaction_engine.get_weight(pair)
                if interaction_prob is not None:
                    interaction_probs.append(interaction_prob)

        if not interaction_probs:
            return None
        return sum(interaction_probs) / len(interaction_probs)

    @staticmethod
    def _extract_market_regime_features(market_context: dict | None) -> dict[str, float]:
        """
        One-hot контекст режима рынка.
        Поддерживаем:
          volatility: high | low
          regime: trend | range
          market_type: trend | range
        """
        if not isinstance(market_context, dict):
            return {}

        out: dict[str, float] = {}

        volatility = market_context.get("volatility")
        if isinstance(volatility, str):
            v = volatility.strip().lower()
            if v == "high":
                out["regime_high_volatility"] = 1.0
            elif v == "low":
                out["regime_low_volatility"] = 1.0

        market_type = market_context.get("regime", market_context.get("market_type"))
        if isinstance(market_type, str):
            r = market_type.strip().lower()
            if r == "trend":
                out["regime_trend"] = 1.0
            elif r == "range":
                out["regime_range"] = 1.0

        return out

    def _build_features(
        self,
        signals: list[dict] | dict[str, float],
        market_context: dict | None = None,
    ) -> dict[str, float]:
        canonical = normalize_signals_to_canonical(signals)
        canonical.update(self._extract_market_regime_features(market_context))
        return canonical

    @staticmethod
    def _context_confidence(market_context: dict | None) -> float:
        """
        Оценка уверенности контекста.
        Пример использования:
            if context_confidence > 0.7: context_weight = 0.4
            else:                          context_weight = 0.2
        """
        if not isinstance(market_context, dict):
            return 0.0

        dims = 0

        volatility = market_context.get("volatility")
        if isinstance(volatility, str) and volatility.strip().lower() in ("high", "low"):
            dims += 1

        market_type = market_context.get("regime", market_context.get("market_type"))
        if isinstance(market_type, str) and market_type.strip().lower() in ("trend", "range"):
            dims += 1

        if dims >= 2:
            return 0.95
        if dims == 1:
            return 0.8
        return 0.0

    def _get_regime_label(self, market_context: dict | None) -> str | None:
        """Сводим market_context в дискретный лейбл для conditional stats."""
        if not isinstance(market_context, dict):
            return None

        volatility = market_context.get("volatility")
        vol_bin = None
        if isinstance(volatility, str):
            v = volatility.strip().lower()
            if v == "high":
                vol_bin = "high_volatility"
            elif v == "low":
                vol_bin = "low_volatility"

        market_type = market_context.get("regime", market_context.get("market_type"))
        reg_bin = None
        if isinstance(market_type, str):
            r = market_type.strip().lower()
            if r == "trend":
                reg_bin = "trend"
            elif r == "range":
                reg_bin = "range"

        # Минимально (как в примере): conditioning делаем по "price regime" (trend/range),
        # а если его нет — по volatility (high/low).
        if reg_bin is not None:
            return reg_bin
        if vol_bin is not None:
            return vol_bin
        return None

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------

    @staticmethod
    def _jaccard(a: frozenset, b: frozenset) -> float:
        union = len(a | b)
        return len(a & b) / union if union > 0 else 0.0

    def __len__(self) -> int:
        return len(self._cases)

    def __repr__(self) -> str:
        return (
            f"PatternEngine("
            f"cases={len(self._cases)}, "
            f"threshold={self._cfg.similarity_threshold}, "
            f"min_matches={self._cfg.min_matches}, "
            f"alpha={self._cfg.laplace_alpha}, "
            f"conf_threshold={self._cfg.confidence_threshold})"
        )

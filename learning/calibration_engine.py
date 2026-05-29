"""
CalibrationEngine — выравнивание предсказанных вероятностей с реальными частотами.

Проблема: модель говорит 80% → на самом деле случается только 60%.
Решение: обучить калибратор на train, применить на test.

Методы (без внешних зависимостей):
    PlattScaler      — логистическая регрессия поверх сырых вероятностей
    IsotonicScaler   — монотонная кусочно-постоянная функция (PAVA)
    BinningScaler    — простейший: заменяет prob на реальную частоту в бине

После калибровки:
    ECE снижается  → вероятности честнее
    Brier Score снижается → предсказания точнее
    Overfitting gap уменьшается → train/test ближе
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Базовый класс
# ---------------------------------------------------------------------------

class BaseCalibrator(ABC):
    """Базовый калибратор вероятностей."""

    _fitted: bool = False

    @abstractmethod
    def fit(self, probs: list[float], outcomes: list[int]) -> "BaseCalibrator":
        """Обучает калибратор на train-данных."""

    @abstractmethod
    def transform(self, probs: list[float]) -> list[float]:
        """Применяет калибровку к новым вероятностям."""

    def fit_transform(self, probs: list[float], outcomes: list[int]) -> list[float]:
        return self.fit(probs, outcomes).transform(probs)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Калибратор не обучен. Вызови fit() сначала.")

    @staticmethod
    def _clip(p: float) -> float:
        return max(1e-6, min(1 - 1e-6, p))


# ---------------------------------------------------------------------------
# Platt Scaling — логистическая регрессия поверх сырых prob
# ---------------------------------------------------------------------------

class PlattScaler(BaseCalibrator):
    """
    Platt Scaling: p_cal = sigmoid(a * p + b)

    Обучается градиентным спуском (без numpy).
    Хорошо работает, когда сырые вероятности монотонны, но смещены.
    """

    def __init__(self, lr: float = 0.01, epochs: int = 1000) -> None:
        self._lr = lr
        self._epochs = epochs
        self._a: float = 1.0
        self._b: float = 0.0

    def fit(self, probs: list[float], outcomes: list[int]) -> "PlattScaler":
        a, b = self._a, self._b
        n = len(probs)

        for _ in range(self._epochs):
            grad_a = grad_b = 0.0
            for p, y in zip(probs, outcomes):
                p_cal = self._sigmoid(a * p + b)
                err = p_cal - y
                grad_a += err * p
                grad_b += err
            a -= self._lr * grad_a / n
            b -= self._lr * grad_b / n

        self._a, self._b = a, b
        self._fitted = True
        return self

    def transform(self, probs: list[float]) -> list[float]:
        self._check_fitted()
        return [self._clip(self._sigmoid(self._a * p + self._b)) for p in probs]

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-max(-500, min(500, x))))

    def __repr__(self) -> str:
        return f"PlattScaler(a={self._a:.4f}, b={self._b:.4f})"


# ---------------------------------------------------------------------------
# Isotonic Regression (PAVA) — монотонная калибровка
# ---------------------------------------------------------------------------

class IsotonicScaler(BaseCalibrator):
    """
    Isotonic Regression методом Pool Adjacent Violators (PAVA).

    Обучает монотонно неубывающую функцию p → p_cal.
    Более гибкая, чем Platt, но требует больше данных (риск переобучения на малых выборках).
    """

    def __init__(self) -> None:
        self._breakpoints: list[tuple[float, float]] = []  # (prob_threshold, calibrated_prob)

    def fit(self, probs: list[float], outcomes: list[int]) -> "IsotonicScaler":
        pairs = sorted(zip(probs, outcomes), key=lambda x: x[0])
        sorted_probs = [p for p, _ in pairs]
        sorted_outcomes = [float(o) for _, o in pairs]

        # PAVA
        blocks: list[list[float]] = [[v] for v in sorted_outcomes]
        while True:
            merged = False
            i = 0
            new_blocks = []
            while i < len(blocks):
                if i + 1 < len(blocks) and _mean(blocks[i]) > _mean(blocks[i + 1]):
                    new_blocks.append(blocks[i] + blocks[i + 1])
                    i += 2
                    merged = True
                else:
                    new_blocks.append(blocks[i])
                    i += 1
            blocks = new_blocks
            if not merged:
                break

        # Разворачиваем блоки → массив откалиброванных значений
        calibrated = []
        for block in blocks:
            m = _mean(block)
            calibrated.extend([m] * len(block))

        # Строим таблицу: sorted_prob → calibrated
        self._breakpoints = list(zip(sorted_probs, calibrated))
        self._fitted = True
        return self

    def transform(self, probs: list[float]) -> list[float]:
        self._check_fitted()
        result = []
        for p in probs:
            # Линейная интерполяция между ближайшими точками
            result.append(self._clip(self._interpolate(p)))
        return result

    def _interpolate(self, p: float) -> float:
        if not self._breakpoints:
            return p
        xs = [bp[0] for bp in self._breakpoints]
        ys = [bp[1] for bp in self._breakpoints]
        if p <= xs[0]:
            return ys[0]
        if p >= xs[-1]:
            return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= p <= xs[i + 1]:
                if xs[i + 1] == xs[i]:
                    return ys[i]
                t = (p - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] + t * (ys[i + 1] - ys[i])
        return p


# ---------------------------------------------------------------------------
# Binning Scaler — простейший метод (надёжный при малых данных)
# ---------------------------------------------------------------------------

class BinningScaler(BaseCalibrator):
    """
    Заменяет вероятность на реальную частоту исходов в соответствующем бине.

    Самый простой метод. Надёжен при малых выборках.
    Не интерполирует — может давать ступенчатые вероятности.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self._n_bins = n_bins
        self._bin_means: dict[int, float] = {}

    def fit(self, probs: list[float], outcomes: list[int]) -> "BinningScaler":
        from collections import defaultdict
        bins: dict[int, list[int]] = defaultdict(list)

        for p, o in zip(probs, outcomes):
            idx = min(int(p * self._n_bins), self._n_bins - 1)
            bins[idx].append(o)

        self._bin_means = {
            idx: _mean(vals) for idx, vals in bins.items()
        }
        self._fitted = True
        return self

    def transform(self, probs: list[float]) -> list[float]:
        self._check_fitted()
        result = []
        for p in probs:
            idx = min(int(p * self._n_bins), self._n_bins - 1)
            if idx in self._bin_means:
                result.append(self._clip(self._bin_means[idx]))
            else:
                # Бин не встречался в train → ближайший
                nearest = min(self._bin_means.keys(), key=lambda k: abs(k - idx))
                result.append(self._clip(self._bin_means[nearest]))
        return result


# ---------------------------------------------------------------------------
# CalibrationEngine — фасад для выбора метода
# ---------------------------------------------------------------------------

class CalibrationEngine:
    """
    Высокоуровневый API для калибровки.

    Использование:
        engine = CalibrationEngine(method="platt")
        engine.fit(train_probs, train_outcomes)
        calibrated = engine.transform(test_probs)
        report = engine.calibration_report(test_probs, test_outcomes)
    """

    METHODS = {"platt": PlattScaler, "isotonic": IsotonicScaler, "binning": BinningScaler}

    def __init__(self, method: str = "platt") -> None:
        if method not in self.METHODS:
            raise ValueError(f"Неизвестный метод '{method}'. Доступны: {list(self.METHODS)}")
        self._method = method
        self._calibrator: BaseCalibrator = self.METHODS[method]()

    def fit(self, probs: list[float], outcomes: list[int]) -> "CalibrationEngine":
        self._calibrator.fit(probs, outcomes)
        return self

    def transform(self, probs: list[float]) -> list[float]:
        return self._calibrator.transform(probs)

    def calibration_report(
        self,
        raw_probs: list[float],
        outcomes: list[int],
        n_bins: int = 5,
    ) -> dict:
        """
        Сравнивает сырые vs откалиброванные вероятности.

        Returns:
            before / after метрики + таблица бинов.
        """
        from backtesting.metrics import brier_score, expected_calibration_error, calibration

        calibrated = self.transform(raw_probs)

        before_bs = brier_score(raw_probs, outcomes)
        after_bs = brier_score(calibrated, outcomes)

        before_ece = expected_calibration_error(raw_probs, outcomes, n_bins=n_bins)
        after_ece = expected_calibration_error(calibrated, outcomes, n_bins=n_bins)

        return {
            "method": self._method,
            "before": {
                "brier_score": round(before_bs, 4),
                "ece": round(before_ece, 4),
            },
            "after": {
                "brier_score": round(after_bs, 4),
                "ece": round(after_ece, 4),
            },
            "improvement": {
                "brier_delta": round(before_bs - after_bs, 4),
                "ece_delta": round(before_ece - after_ece, 4),
            },
            "bins_before": calibration(raw_probs, outcomes, n_bins=n_bins),
            "bins_after": calibration(calibrated, outcomes, n_bins=n_bins),
        }


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

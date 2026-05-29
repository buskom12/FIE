"""
Стратегии разбивки датасета на train/test.

walk_forward_split  — хронологический порядок (ближе к реальному рынку)
train_test_split    — случайное перемешивание (для i.i.d. данных)
k_fold_split        — k фолдов для оценки дисперсии результатов
"""

from __future__ import annotations

import random


def sort_temporal(data: list[dict]) -> list[dict]:
    """
    Сортирует кейсы по времени для честной temporal-валидации.
    Приоритет ключей: timestamp -> id.
    """
    if not data:
        return []

    sample = data[0]
    if isinstance(sample, dict) and "timestamp" in sample:
        return sorted(data, key=lambda x: x.get("timestamp"))
    if isinstance(sample, dict) and "id" in sample:
        return sorted(data, key=lambda x: x.get("id", 0))
    return list(data)


def walk_forward_split(
    data: list,
    split: float = 0.8,
) -> tuple[list, list]:
    """
    Хронологический split: прошлое → будущее.

    Не перемешивает данные — наиболее честная оценка для временных рядов.
    Модель обучается только на том, что «уже случилось».
    """
    if not 0 < split < 1:
        raise ValueError(f"split должен быть в диапазоне (0, 1), получено: {split}")

    temporal = sort_temporal(list(data))
    split_idx = int(len(temporal) * split)
    return temporal[:split_idx], temporal[split_idx:]


def train_test_split(
    data: list,
    test_size: float = 0.2,
    seed: int | None = 42,
) -> tuple[list, list]:
    """
    Случайное перемешивание + split.

    Используй для данных без временной зависимости.
    seed фиксирует воспроизводимость.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size должен быть в диапазоне (0, 1), получено: {test_size}")

    data = list(data)  # не мутируем оригинал
    random.seed(seed)
    random.shuffle(data)

    split_idx = int(len(data) * (1 - test_size))
    return data[:split_idx], data[split_idx:]


def k_fold_split(data: list, k: int = 5) -> list[tuple[list, list]]:
    """
    K-fold cross-validation: k пар (train, test).

    Позволяет оценить дисперсию accuracy — если результаты
    сильно скачут между фолдами, модель нестабильна.
    """
    if k < 2:
        raise ValueError("k должно быть >= 2")

    data = list(data)
    fold_size = len(data) // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(data)
        test = data[start:end]
        train = data[:start] + data[end:]
        folds.append((train, test))

    return folds


def temporal_k_fold_split(data: list[dict], k: int = 5) -> list[tuple[list, list]]:
    """
    Temporal CV без утечки: test всегда в будущем относительно train.
    Каждая итерация использует expanding-window:
        fold i: train=[0:boundary_i], test=[boundary_i:boundary_{i+1}]
    """
    if k < 2:
        raise ValueError("k должно быть >= 2")

    temporal = sort_temporal(list(data))
    n = len(temporal)
    if n < k + 1:
        raise ValueError("Недостаточно данных для temporal k-fold.")

    block = n // (k + 1)
    if block == 0:
        raise ValueError("Размер temporal-блока равен 0.")

    folds: list[tuple[list, list]] = []
    for i in range(1, k + 1):
        train_end = i * block
        test_end = n if i == k else (i + 1) * block
        train = temporal[:train_end]
        test = temporal[train_end:test_end]
        if train and test:
            folds.append((train, test))

    return folds


def random_baseline(n: int, seed: int | None = None) -> list[float]:
    """
    Случайный baseline — имитация «монеты».
    Используется для сравнения: FIE должен быть лучше этого.
    """
    rng = random.Random(seed)
    return [rng.random() for _ in range(n)]

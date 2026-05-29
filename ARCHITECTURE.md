# Архитектура Pattern Weighting

Пайплайн сопоставляет текущий набор признаков с историей, сглаживает неопределённость и комбинирует «память паттернов» с взвешенным скорингом сигналов.

## Слои

```
Данные / события
       ↓
Signal Intelligence Layer   ← нормализация и структура сигналов
       ↓
Pattern matching (Жаккар, Laplace, uncertainty)
       ↓
Pattern Weighting            ← гибрид вероятности паттерна и весов сигналов
```

### Signal Intelligence Layer

Слой отвечает за единый канонический вид входа перед Pattern Engine и обучением весов.

#### Шаг 1 — структура сигнала

Каждый кейс описывается словарём **канонических имён** → **сила присутствия** (обычно `1`, если сигнал активен; отсутствующие ключи или `0` означают «нет сигнала»).

```python
signals = {
    "low_volume": 1,
    "whale_absence": 1,
    "volatility_compression": 1,
}
```

Список ключей фиксирован; совместимость со старым форматом `list[{"type": ..., "strength": ...}]` сохраняется через `normalize_signals_to_canonical` в `signals/weights.py`.

#### Шаг 2 — хранилище весов (`signals/weights.py`)

Словарь `signal_weights` — дефолтные веса до обучения. Обновляется через `fit_signal_weights` (пересчёт по train-сету) или EMA.

#### Шаг 3 — вес из истории (`compute_signal_weights`)

Для каждого сигнала по кейсам датасета:

```
prob  = (pos + 1) / (total + 2)   # Laplace
edge  = abs(prob - 0.5)
weight = edge * 2                  # 0.0 мусор → 0.2 слабый → 0.5 хороший → 0.8+ сильный
```

#### Шаг 4 — применение весов (`PatternEngine.score_case`)

Взвешенное среднее присутствия сигналов:

```
score = sum(w * v for s, v in signals) / sum(w)
w = weights.get(s, 0.1)
```

Возвращает `None`, если ни одного сигнала не удалось взвесить.

#### Шаг 5 — гибрид паттернов и весов (`PatternEngine.compute_final_probability`)

```
patterns = "память"    → pattern_prob = compute_probability(matches)
weights  = "интеллект" → signal_score = score_case(signals, weights)

if pattern_prob is None:
    final_prob = signal_score        # нет истории — только интеллект
else:
    final_prob = (pattern_prob + signal_score) / 2
```

Поле `source` в результате: `"hybrid"` | `"signal_only"` | `"pattern_only"` | `"none"`.

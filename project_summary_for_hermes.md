# FIE Project Summary for Hermes

## 1. Что такое FIE

FIE — Python-проект для торговли на prediction-рынках (например, Polymarket).
Цель: найти и реализовать edge через автоматические входы/выходы на основе рыночных сигналов, паттернов и агентных оценок.

Проект сочетает:
- production trading loop (`services/prod_loop.py`)
- persistent SQLite схему сделок (`db/prod/models.py`)
- agentную архитектуру и рольную агрегацию
- паттерн-движок и обучение на исторических данных
- мониторинг экспериментальных cohort-метрик

---

## 2. Структура проекта

### `agents/`
- `agent_swarm.py` — запускает FIE-рій через `future-intelligence-engine.AgentManager`
- `debate.py` — rule-based дебаты ролей без LLM
- `role_based_decision.py` — строит role votes, веса ролей, итоговую вероятность

### `patterns/`
- `agent_patterns.py` — архетипы паттерн-агентов: `SmartWhale`, `QuantModel`, `MacroAnalyst`, `Gambler`
- `pattern_engine.py` — PatternEngine: поиск похожих кейсов, Laplace smoothing, confidence, `None` при недостатке данных

### `learning/`
- `evolution_engine.py` — эволюция весов агентов по фактическим исходам
- `role_weights.py` — загрузка/сохранение весов/статистики ролей
- `collective_learning_loop.py` — общий цикл коллективного обучения (обновление после исходов)

### `memory/`
- `society_memory.py` — простая память событий и поиск похожих по тексту

### `services/`
- `prod_loop.py` — production loop, вход/выход, risk gating, запись в БД

### `db/prod/`
- `models.py` — схема `Trade`, `PortfolioSnapshot`, `RegimeStat`, `SignalLatest`

### `data/`
- `collectors/news.py` — RSS + Polymarket events + placeholder X API
- `collectors/market.py` — BTC OHLCV: Binance API / csv / кэш
- `loader.py` — загрузка `data/historical_events.json`
- `historical_events.json` — исторические данные для бэктестов

### `api/`
- `api/main.py` — HTTP endpoint, который запускает `run_swarm` и агрегирует предсказания

---

## 3. Как устроены агенты

### Агентная архитектура
- `future-intelligence-engine/agents/agent_manager.py` — создает рой через `generate_swarm(100)` и оценивает событие
- `future-intelligence-engine/agents/swarm_factory.py` — строит рой из `LLMAgent` по пресетам или кастомной композиции
- `future-intelligence-engine/agents/llm_agent.py` — обращается к локальному LLM-серверу `http://localhost:11434/api/generate`
  и возвращает `{probability, reasoning}`
- `future-intelligence-engine/agents/debate_engine.py` — post-hoc дебаты: каждый агент видит мнения других и пересчитывает вероятность
- `future-intelligence-engine/agents/personas.py` — описания персон агентов, `risk_tolerance`, `weight`, `accuracy`

### Основные роли / пресеты
- `crypto_experts` — киты, ритейл, маркет-мейкеры, геймблеры
- `macro_analysts` — экономисты, политики, банкиры, риск-аналитики
- `tech_innovators` — AI-исследователи, стартаперы, devs
- `diverse` — смешанный рой всех типов

---

## 4. Как работает `patterns/` и `learning/`

### `patterns/agent_patterns.py`
- `SmartWhaleAgent` — on-chain и крупные деньги
- `QuantModelAgent` — технические сигналы, RSI/MACD/volume
- `MacroAnalystAgent` — макро контекст, риск-on/off
- `GamblerAgent` — рандомный baseline

### `patterns/pattern_engine.py`
- хранит кейсы и находит похожие по Жаккару
- `add_case(signals, outcome, market_context)` — сохраняет кейс
- `find_patterns(signals, market_context)` — ищет совпадения
- `compute_probability(matches)` — возвращает вероятность или `None` при недостатке данных
- использует:
  - минимальную поддержку (`min_matches`)
  - Laplace smoothing (`alpha`)
  - penalty к доверительной оценке при низком `n`

### `learning/evolution_engine.py`
- усиливает агента, если он оказался прав
- ослабляет агентa, если он ошибся
- веса ограничены `weight_min` / `weight_max`
- обновляет `persona.accuracy` EMA

### `learning/role_weights.py`
- загружает веса ролей из `data/logs/role_weights.json`
- fallback к `DEFAULT_ROLE_WEIGHTS`
- используется в `patterns/agent_patterns.py`

### `memory/society_memory.py`
- хранит простую историю событий
- возвращает похожие события по совпадению слов

---

## 5. Пайплайн: от новости до вероятности

1. `data/collectors/news.py` собирает события:
   - RSS фиды Cointelegraph и Coindesk
   - текущие Polymarket events через `gamma-api.polymarket.com`
   - placeholder для X API

2. `data/collectors/market.py` получает BTC OHLCV:
   - priority: CSV / кэш / Binance API
   - результат: свечи с `timestamp, open, high, low, price, volume`

3. `data/historical_events.json` и `data/loader.py` — исторический dataset для backtest

4. `patterns/pattern_engine.py` обучает / находит паттерны и считает вероятность

5. `future-intelligence-engine/agents/llm_agent.py` строит LLM-прогноз для каждой персоны

6. `agents/agent_swarm.py` агрегирует рой через `AgentManager.evaluate_event`

7. `prediction.aggregation.aggregate_predictions` собирает финальную вероятность и confidence

8. `services/prod_loop.py` использует эти метрики для торговли и записывает задачу в БД

---

## 6. Схема `db/prod/models.py`

### `Trade`
- `id`, `timestamp`, `strategy`, `entry_type`
- `regime_key`, `confidence`, `gate_score`, `score_C`, `d_score`
- `alpha_multiplier`, `c_size_multiplier`, `strong_multiplier`
- `edge`, `edge_score`, `p_model`, `p_market`, `edge_real`
- `size`, `variance`, `kelly_fraction`
- `tail_hit`
- `funding_stress_48`, `liq_spike`, `scen_breakout_suspicious`, `momentum_up`, `oi_strength`
- `hour_utc`, `p_bucket`, `hour_weight`, `hour_n`, `is_prior`, `market_active_min`
- `entry_price`, `exit_price`, `side`, `holding_steps`, `hold_min`
- `exit_reason`, `exit_signal`, `timeout_hit`, `reverse_hit`
- `mfe`, `mae`, `pnl`

### `PortfolioSnapshot`
- `equity`, `drawdown`, `sharpe_rolling`, `capital_A`, `capital_B`, `cap_hits`, `tail_hits`

### `RegimeStat`
- `regime_key`, `trades`, `pnl_sum`, `wins`, `ema_exp`, `ema_var`, `last_update_ts`

### `SignalLatest`
- последний сигнал/режим/volatility по стратегии

---

## 7. Текущий эксперимент и метрики

### Текущий статус
- эксперимент frozen
- изменён только timeout/hold logic для gate score в `0.1–0.3`
- `cut_id = 8682`
- текущий cohort: `n = 39`

### Что мониторим
- главные метрики:
  - `avg_pnl`
  - `capture_ratio`
  - `timeout_rate`
  - `tp_rate`
  - `total_pnl`
- вторичные:
  - `avg_mfe`
  - `avg_mae`
  - `avg_hold`

### Как считаем, что эксперимент сломался
- `capture_ratio <= 0`
- или `timeout_rate` возвращается к `~0.8–0.9`
- или `avg_pnl` становится отрицательным

### Когда принимать решение
- `n ≈ 50` — sanity check
- `n ≥ 100` — принимать решение оставлять/откатывать
- `n ≥ 200` — проверка устойчивости

---

## 8. Ключевые файлы для чтения Hermes

- `ARCHITECTURE.md`
- `README.md`
- `services/prod_loop.py`
- `db/prod/models.py`
- `patterns/pattern_engine.py`
- `agents/agent_swarm.py`
- `future-intelligence-engine/agents/agent_manager.py`
- `future-intelligence-engine/agents/llm_agent.py`
- `data/collectors/news.py`
- `data/collectors/market.py`
- `run_backtest.py`
- `run_state_backtest.py`

---

## 9. Что важно донести Hermes

- FIE — гибрид: паттерны + агентные оценки + evolution weights + production loop
- Сейчас это не эксперимент по новой модели, а замороженный timeout/hold эксперимент
- Самое ценное: найден механизм разрушения edge, а не только tweak p_model
- Если эксперимент подтвердится, следующий шаг — адаптивный timeout multiplier

---

## 10. Ссылки на основные модули

- `agents/agent_swarm.py`
- `future-intelligence-engine/agents/agent_manager.py`
- `future-intelligence-engine/agents/swarm_factory.py`
- `future-intelligence-engine/agents/llm_agent.py`
- `future-intelligence-engine/agents/debate_engine.py`
- `patterns/pattern_engine.py`
- `db/prod/models.py`
- `services/prod_loop.py`
- `data/collectors/news.py`
- `data/collectors/market.py`
- `data/historical_events.json`
- `run_backtest.py`

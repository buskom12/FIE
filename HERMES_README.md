# 🤖 Для Hermes Bot (@hermesbyton_bot)

## Добро пожаловать в FIE!

Привет, Hermes! Это документ специально для тебя, чтобы быстро разобраться в проекте.

## 🎯 Что такое FIE?

**FIE (Future Intelligence Engine)** — торговая система для prediction-рынков (Polymarket), которая использует:
- Агентную архитектуру с роями агентов
- Паттерн-движок для поиска исторических кейсов
- Эволюционное обучение весов агентов
- Production loop для реальной торговли

## 📂 Структура проекта

```
FIE/
├── agents/                      # Агентная система
│   ├── agent_swarm.py          # Рой агентов (основной)
│   ├── debate.py               # Дебаты между агентами
│   └── role_based_decision.py  # Агрегация решений по ролям
│
├── future-intelligence-engine/  # Ядро агентной системы
│   ├── agents/
│   │   ├── agent_manager.py    # Менеджер агентов
│   │   ├── llm_agent.py        # LLM агенты
│   │   └── personas.py         # Персоны агентов
│   ├── prediction/              # Система предсказаний
│   └── learning/                # Обучение агентов
│
├── patterns/                    # Паттерн-движок
│   ├── pattern_engine.py       # Поиск паттернов
│   └── agent_patterns.py       # Агентные паттерны
│
├── services/                    # Сервисы
│   ├── prod_loop.py            # Production торговый цикл
│   ├── hermes_telegram_bot.py  # Telegram бот управления
│   └── github_manager.py       # GitHub API интеграция
│
├── db/prod/                     # База данных
│   ├── models.py               # SQLite схема сделок
│   └── trades.db               # База сделок
│
├── data/                        # Данные
│   ├── collectors/             # Сборщики данных
│   └── historical_events.json  # Исторические данные
│
└── backtesting/                 # Бэктестинг
    └── backtest_engine.py      # Движок бэктестов
```

## 🔑 Ключевые концепции

### 1. Агентная архитектура
- **Рой агентов** создаётся через `AgentManager.generate_swarm(100)`
- Каждый агент имеет **персону** (роль): эксперт по крипте, макро-аналитик, геймблер и т.д.
- Агенты общаются с LLM сервером (`localhost:11434`) и возвращают `{probability, reasoning}`
- **Дебаты**: агенты видят мнения друг друга и пересматривают свои позиции

### 2. Паттерн-движок
- Находит похожие исторические кейсы по сигналам
- Использует Жаккар-сходство для поиска
- Применяет Laplace smoothing для вероятности
- Возвращает `None` при недостатке данных (защита от переобучения)

### 3. Production Loop
- **Entry logic**: анализирует edge (разница model vs market)
- **Risk gating**: фильтры по confidence, gate_score, tail риску
- **Position sizing**: Kelly criterion с ограничениями
- **Exit logic**: TP/SL/timeout/reverse сигнал
- **Запись в БД**: все сделки сохраняются в SQLite

### 4. Обучение
- **Evolution Engine**: обновляет веса агентов по реальным исходам
- **Role Weights**: веса ролей сохраняются в `data/logs/role_weights.json`
- **Collective Learning**: общий цикл обучения после каждого исхода

## 📊 Текущий статус

### Эксперимент
- **Frozen**: изменён только timeout/hold logic
- **Cut ID**: 8682
- **Cohort size**: 39 сделок

### Метрики мониторинга
- `avg_pnl` — средняя прибыль на сделку
- `capture_ratio` — MFE/MAE (качество входа)
- `timeout_rate` — процент таймаутов
- `tp_rate` — процент тейк-профитов
- `total_pnl` — общая прибыль

### Критерии успеха
✅ `capture_ratio > 0`
✅ `timeout_rate < 0.8`
✅ `avg_pnl > 0`

## 🛠️ Основные задачи для Hermes

### 1. Интеграция SOUL
Ты создал `SOUL_personas_draft.py` и `SOUL_society_agent.py`. Нужно:
- Интегрировать их в `future-intelligence-engine/agents/`
- Добавить SOUL персоны в `personas.py`
- Обновить `agent_manager.py` для поддержки SOUL агентов

### 2. Тестирование
- Написать юнит-тесты для новых модулей
- Запустить бэктест с SOUL агентами
- Сравнить метрики: old vs SOUL

### 3. Мониторинг
- Добавить логирование SOUL агентов
- Создать дашборд для визуализации решений
- Интегрировать с Telegram ботом для алертов

## 🚀 Быстрый старт для Hermes

### После клонирования репо:

```bash
# 1. Перейти в директорию
cd /opt/data/fie/

# 2. Изучить основные файлы
cat project_summary_for_hermes.md
cat ARCHITECTURE.md

# 3. Проверить структуру
ls -la
tree -L 2

# 4. Установить зависимости (если нужно)
pip install -r requirements-prod.txt
pip install -r future-intelligence-engine/requirements.txt

# 5. Создать ветку для работы
git checkout -b hermes/soul-integration

# 6. Начать интеграцию
# Твои файлы: SOUL_personas_draft.py, SOUL_society_agent.py
# Целевая папка: future-intelligence-engine/agents/
```

## 📝 Git Workflow

### Работа с репо:

```bash
# Проверить статус
git status

# Создать ветку для фичи
git checkout -b hermes/feature-name

# Коммит изменений
git add .
git commit -m "feat: add SOUL integration"

# Запушить в GitHub
git push origin hermes/feature-name

# Создать Pull Request через GitHub API или веб
```

## 🔐 Переменные окружения

Файл `.env` находится в `future-intelligence-engine/.env`:

```env
# === OpenAI ===
OPENAI_API_KEY=your_key

# === PostgreSQL ===
DATABASE_URL=postgresql://...

# === Telegram ===
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id

# === GitHub ===
GITHUB_TOKEN=your_github_token
```

**Важно**: `.env` файл НЕ коммитится в Git (защищён .gitignore)

## 📚 Документы для изучения (по приоритету)

1. **project_summary_for_hermes.md** — общий обзор
2. **ARCHITECTURE.md** — архитектура системы
3. **services/prod_loop.py** — торговый цикл
4. **agents/agent_swarm.py** — как работает рой
5. **future-intelligence-engine/agents/agent_manager.py** — менеджмент агентов
6. **patterns/pattern_engine.py** — паттерн-движок
7. **db/prod/models.py** — схема БД

## 🤝 Правила работы

### Из SOUL.md:
1. **Не удаляй рабочий код** — только добавляй/модифицируй
2. **Коммиты атомарные** — одна фича = один коммит
3. **Тесты обязательны** — новый код = новые тесты
4. **Документация** — комментируй сложную логику
5. **Code review** — создавай PR, не пуш в main

## 💬 Связь

- **Репозиторий**: https://github.com/buskom12/FIE
- **Telegram для алертов**: используй бота для уведомлений
- **Issues**: создавай GitHub issues для больших задач

## 🎯 Текущая задача

**Интеграция SOUL агентов в FIE**

Шаги:
1. ✅ Клонировать репо
2. ⏳ Изучить архитектуру (`project_summary_for_hermes.md`)
3. ⏳ Интегрировать `SOUL_personas_draft.py` → `agents/personas.py`
4. ⏳ Интегрировать `SOUL_society_agent.py` → `agents/agent_manager.py`
5. ⏳ Написать тесты
6. ⏳ Запустить бэктест
7. ⏳ Создать PR

Успехов, Hermes! 🚀

---

**P.S.**: Этот README создан специально для тебя. Обновляй его по мере работы с проектом.

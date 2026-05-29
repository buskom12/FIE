# 🤖 Hermes Telegram Bot

Telegram бот для управления FIE проектом удалённо через мессенджер.

## 🎯 Возможности

### Мониторинг
- ✅ Статус системы (процессы, БД, Git)
- ✅ Торговые метрики в реальном времени
- ✅ Просмотр логов
- ✅ Информация о GitHub репозитории

### Управление
- ✅ Запуск/остановка FIE
- ✅ Запуск бэктестов
- ✅ Git операции (pull, push, log)
- ✅ Создание GitHub issues

### GitHub интеграция
- ✅ Статус репозитория
- ✅ История коммитов
- ✅ Открытые issues и PR
- ✅ Статистика активности

## 📦 Установка

### 1. Установите зависимости

```bash
cd /Users/dimonbodrov/FIE/services
pip install -r requirements_bot.txt
```

### 2. Настройте .env файл

Добавьте в `future-intelligence-engine/.env`:

```env
# === Telegram ===
TELEGRAM_BOT_TOKEN=ваш_новый_токен_бота
TELEGRAM_CHAT_ID=ваш_chat_id

# === GitHub (опционально) ===
GITHUB_TOKEN=ваш_github_token
```

**Как получить CHAT_ID:**
1. Напишите боту любое сообщение
2. Откройте: `https://api.telegram.org/bot<ВАШ_ТОКЕН>/getUpdates`
3. Найдите поле `"chat":{"id":123456789}`
4. Скопируйте это число в `TELEGRAM_CHAT_ID`

**GitHub Token (опционально):**
- Нужен только для приватных репозиториев
- Создать: GitHub → Settings → Developer settings → Personal access tokens → Generate new token
- Права: `repo` (Full control of private repositories)

### 3. Запустите бота

```bash
cd /Users/dimonbodrov/FIE
python services/hermes_telegram_bot.py
```

**Или запустить в фоне:**

```bash
nohup python services/hermes_telegram_bot.py > logs/hermes_bot.log 2>&1 &
```

## 📱 Команды бота

### Основные команды

```
/start              - Главное меню с кнопками
/help               - Справка по всем командам
/status             - Статус системы (FIE, Git, БД)
```

### Мониторинг

```
/metrics            - Торговые метрики (PnL, Win Rate, и т.д.)
/logs               - Последние 20 строк логов
/logs 50            - Последние 50 строк логов
```

### Управление FIE

```
/run                - Запустить FIE
/stop               - Остановить FIE
/backtest           - Запустить state backtest (по умолчанию)
/backtest state     - Запустить state backtest
/backtest robust    - Запустить robust backtest
```

### Git операции

```
/git                - Статус репозитория
/git pull           - Подтянуть изменения с GitHub
/git push           - Запушить локальные изменения
/git log            - Последние 5 коммитов
/git log 10         - Последние 10 коммитов
```

### GitHub API

Для расширенных функций через GitHub API, добавьте команды в бота:

```python
# В hermes_telegram_bot.py добавьте:
from services.github_manager import GitHubManager

async def github_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """GitHub статус через API"""
    gh = GitHubManager()
    status = gh.format_repo_status()
    await update.message.reply_text(status, parse_mode="Markdown")

async def github_issues(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Открытые issues"""
    gh = GitHubManager()
    issues = gh.format_issues()
    await update.message.reply_text(issues, parse_mode="Markdown")
```

## 🎨 Интерфейс

При команде `/start` появляется меню с кнопками:

```
┌─────────────┬─────────────┐
│ 📊 Статус   │ 📈 Метрики  │
├─────────────┼─────────────┤
│ 🔄 Git      │ 📝 Логи     │
├─────────────┼─────────────┤
│ ▶️ Запустить│ ⏹️ Остановить│
└─────────────┴─────────────┘
```

## 📊 Примеры вывода

### Статус системы
```
📊 Статус системы

🔴 FIE: 🟢 Запущен
🌿 Git ветка: `main`
💾 БД: `2.45 MB`

Время: `2026-05-29 23:15:30`
```

### Метрики
```
📈 Торговые метрики

💰 Total PnL: `123.45`
📊 Avg PnL: `0.0234`
🎯 Win Rate: `55.5%`
📉 Capture Ratio: `0.75`
⏱️ Timeout Rate: `35.0%`
📈 TP Rate: `45.0%`

📊 Trades: `100`
⏰ Обновлено: `23:15:30`
```

### GitHub статус
```
🗂 GitHub Репозиторий

📦 Название: `FIE`
⭐️ Stars: 0
🍴 Forks: 0
🐛 Open Issues: 0
🌿 Default Branch: `main`
💾 Размер: 1024 KB
🔤 Язык: Python

📝 Последние коммиты:

`f8a2916` - Add root .gitignore for security
└ dimonbodrov • 29.05 22:45

`068cc26` - include future-intelligence-engine contents
└ dimonbodrov • 29.05 22:16
```

## 🔐 Безопасность

### ✅ Что безопасно
- Токен бота НЕ попадает в GitHub (защищён .gitignore)
- Только ваш CHAT_ID может управлять ботом
- GitHub token опционален (нужен только для приватных репо)

### ⚠️ Рекомендации
1. **Не давайте боту админских прав** в группах
2. **Регулярно обновляйте токен** (раз в 3-6 месяцев)
3. **GitHub token** создавайте с минимальными правами
4. **Логи бота** проверяйте на наличие ошибок

### 🔒 Ограничение доступа

В коде бота можно добавить проверку CHAT_ID:

```python
async def check_permission(self, update: Update) -> bool:
    """Проверка доступа"""
    allowed_chat_id = int(CHAT_ID)
    if update.effective_chat.id != allowed_chat_id:
        await update.message.reply_text("❌ Доступ запрещён")
        return False
    return True
```

## 🚀 Автозапуск бота

### macOS (через launchd)

Создайте файл `~/Library/LaunchAgents/com.hermes.bot.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hermes.bot</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/python3</string>
        <string>/Users/dimonbodrov/FIE/services/hermes_telegram_bot.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/dimonbodrov/FIE</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/dimonbodrov/FIE/logs/hermes_bot.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/dimonbodrov/FIE/logs/hermes_bot_error.log</string>
</dict>
</plist>
```

Активируйте:
```bash
launchctl load ~/Library/LaunchAgents/com.hermes.bot.plist
launchctl start com.hermes.bot
```

### Linux (через systemd)

Создайте `/etc/systemd/system/hermes-bot.service`:

```ini
[Unit]
Description=Hermes Telegram Bot
After=network.target

[Service]
Type=simple
User=yourusername
WorkingDirectory=/path/to/FIE
ExecStart=/usr/bin/python3 /path/to/FIE/services/hermes_telegram_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Активируйте:
```bash
sudo systemctl daemon-reload
sudo systemctl enable hermes-bot
sudo systemctl start hermes-bot
```

## 🔧 Расширение функционала

### Добавление новой команды

```python
# В hermes_telegram_bot.py

# 1. Зарегистрируйте обработчик
def _register_handlers(self):
    # ... существующие handlers
    self.app.add_handler(CommandHandler("mycmd", self.my_command))

# 2. Создайте метод-обработчик
async def my_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Описание команды"""
    try:
        # Ваша логика
        result = "Результат"
        await update.message.reply_text(result, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")
```

### Интеграция с GitHub API

```python
from services.github_manager import GitHubManager

async def create_issue_from_telegram(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Создать issue через Telegram"""
    # /issue Заголовок | Описание
    if not context.args:
        await update.message.reply_text("Используйте: /issue Заголовок | Описание")
        return
    
    text = " ".join(context.args)
    parts = text.split("|")
    
    if len(parts) != 2:
        await update.message.reply_text("Формат: /issue Заголовок | Описание")
        return
    
    title, body = parts[0].strip(), parts[1].strip()
    
    gh = GitHubManager()
    result = gh.create_issue(title, body, labels=["from-telegram"])
    
    if "error" in result:
        await update.message.reply_text(f"❌ Ошибка: {result['error']}")
    else:
        await update.message.reply_text(
            f"✅ Issue создан: #{result['number']}\n{result['url']}"
        )
```

## 📞 Поддержка

Если возникли проблемы:

1. Проверьте логи: `tail -f logs/hermes_bot.log`
2. Проверьте, что бот запущен: `pgrep -f hermes_telegram_bot`
3. Проверьте .env файл на наличие правильных токенов
4. Попробуйте перезапустить бота

## 📝 TODO

- [ ] Webhook режим вместо polling
- [ ] Поддержка нескольких пользователей
- [ ] Алерты при критических событиях
- [ ] Графики метрик
- [ ] Интерактивные кнопки для всех команд
- [ ] Создание issues/PR через бота
- [ ] Управление GitHub Actions

## 📄 Лицензия

Часть проекта FIE (Future Intelligence Engine)

"""
Hermes Telegram Bot - управление FIE проектом через Telegram
Функции:
- Мониторинг статуса проекта
- Управление GitHub репозиторием
- Запуск бэктестов
- Получение метрик и отчетов
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from dotenv import load_dotenv

# Загружаем переменные окружения
env_path = Path(__file__).resolve().parent.parent / "future-intelligence-engine" / ".env"
load_dotenv(dotenv_path=env_path)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # Опционально для приватных репо
REPO_PATH = Path(__file__).resolve().parent.parent


class HermesBot:
    """Главный класс бота Hermes"""

    def __init__(self):
        self.app = Application.builder().token(BOT_TOKEN).build()
        self._register_handlers()

    def _register_handlers(self):
        """Регистрация обработчиков команд"""
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("status", self.status))
        self.app.add_handler(CommandHandler("metrics", self.metrics))
        self.app.add_handler(CommandHandler("backtest", self.backtest))
        self.app.add_handler(CommandHandler("git", self.git_status))
        self.app.add_handler(CommandHandler("logs", self.get_logs))
        self.app.add_handler(CommandHandler("run", self.run_fie))
        self.app.add_handler(CommandHandler("stop", self.stop_fie))
        self.app.add_handler(CallbackQueryHandler(self.button_callback))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Приветственное сообщение"""
        keyboard = [
            [
                InlineKeyboardButton("📊 Статус", callback_data="status"),
                InlineKeyboardButton("📈 Метрики", callback_data="metrics"),
            ],
            [
                InlineKeyboardButton("🔄 Git статус", callback_data="git"),
                InlineKeyboardButton("📝 Логи", callback_data="logs"),
            ],
            [
                InlineKeyboardButton("▶️ Запустить", callback_data="run"),
                InlineKeyboardButton("⏹️ Остановить", callback_data="stop"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "🤖 *Hermes Bot активирован*\n\n"
            "Я помогу управлять FIE проектом через Telegram.\n\n"
            "Используйте /help для списка команд.",
            parse_mode="Markdown",
            reply_markup=reply_markup,
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Справка по командам"""
        help_text = """
🤖 *Команды Hermes Bot*

*Основные:*
/start - Главное меню
/help - Эта справка
/status - Статус системы

*Мониторинг:*
/metrics - Торговые метрики
/logs [N] - Последние N строк логов (по умолчанию 20)

*Управление:*
/run - Запустить FIE
/stop - Остановить FIE
/backtest [strategy] - Запустить бэктест

*GitHub:*
/git - Статус репозитория
/git pull - Подтянуть изменения
/git push - Запушить изменения
/git log - История коммитов

*Примеры:*
`/logs 50` - показать 50 строк логов
`/backtest state` - запустить state backtest
`/git log 5` - показать 5 последних коммитов
        """
        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Статус системы"""
        try:
            # Проверяем, запущен ли процесс FIE
            result = subprocess.run(
                ["pgrep", "-f", "run_fie.py"],
                capture_output=True,
                text=True,
                cwd=REPO_PATH,
            )
            is_running = bool(result.stdout.strip())

            # Проверяем последнюю запись в БД
            db_status = self._check_db_status()

            # Git статус
            git_branch = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                cwd=REPO_PATH,
            ).stdout.strip()

            status_text = f"""
📊 *Статус системы*

🔴 FIE: {'🟢 Запущен' if is_running else '🔴 Остановлен'}
🌿 Git ветка: `{git_branch}`
{db_status}

Время: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
            """
            await update.message.reply_text(status_text.strip(), parse_mode="Markdown")

        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {str(e)}")

    async def metrics(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Получение торговых метрик"""
        try:
            # Читаем метрики из БД
            metrics_data = self._get_metrics_from_db()

            if metrics_data:
                metrics_text = f"""
📈 *Торговые метрики*

💰 Total PnL: `{metrics_data.get('total_pnl', 'N/A')}`
📊 Avg PnL: `{metrics_data.get('avg_pnl', 'N/A')}`
🎯 Win Rate: `{metrics_data.get('win_rate', 'N/A')}%`
📉 Capture Ratio: `{metrics_data.get('capture_ratio', 'N/A')}`
⏱️ Timeout Rate: `{metrics_data.get('timeout_rate', 'N/A')}%`
📈 TP Rate: `{metrics_data.get('tp_rate', 'N/A')}%`

📊 Trades: `{metrics_data.get('n_trades', 'N/A')}`
⏰ Обновлено: `{datetime.now().strftime('%H:%M:%S')}`
                """
            else:
                metrics_text = "⚠️ Нет данных для отображения метрик"

            await update.message.reply_text(metrics_text.strip(), parse_mode="Markdown")

        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка получения метрик: {str(e)}")

    async def backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Запуск бэктеста"""
        try:
            strategy = context.args[0] if context.args else "state"
            
            await update.message.reply_text(
                f"🔄 Запускаю бэктест: `{strategy}`\nЭто может занять несколько минут...",
                parse_mode="Markdown"
            )

            backtest_script = f"run_{strategy}_backtest.py"
            result = subprocess.run(
                ["python", backtest_script],
                capture_output=True,
                text=True,
                cwd=REPO_PATH,
                timeout=300,
            )

            if result.returncode == 0:
                output = result.stdout[-1000:]  # Последние 1000 символов
                await update.message.reply_text(
                    f"✅ Бэктест завершен\n\n```\n{output}\n```",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text(
                    f"❌ Ошибка бэктеста:\n```\n{result.stderr[-500:]}\n```",
                    parse_mode="Markdown"
                )

        except subprocess.TimeoutExpired:
            await update.message.reply_text("⏰ Превышено время ожидания (5 минут)")
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {str(e)}")

    async def git_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Git операции"""
        try:
            if not context.args:
                # Просто git status
                result = subprocess.run(
                    ["git", "status", "--short"],
                    capture_output=True,
                    text=True,
                    cwd=REPO_PATH,
                )
                branch = subprocess.run(
                    ["git", "branch", "--show-current"],
                    capture_output=True,
                    text=True,
                    cwd=REPO_PATH,
                ).stdout.strip()

                status = result.stdout if result.stdout else "✅ Рабочее дерево чистое"
                
                await update.message.reply_text(
                    f"🌿 *Git статус*\n\nВетка: `{branch}`\n\n```\n{status}\n```",
                    parse_mode="Markdown"
                )
                return

            command = context.args[0]
            
            if command == "pull":
                result = subprocess.run(
                    ["git", "pull"],
                    capture_output=True,
                    text=True,
                    cwd=REPO_PATH,
                )
                await update.message.reply_text(
                    f"🔄 Git pull:\n```\n{result.stdout}\n```",
                    parse_mode="Markdown"
                )

            elif command == "push":
                result = subprocess.run(
                    ["git", "push"],
                    capture_output=True,
                    text=True,
                    cwd=REPO_PATH,
                )
                await update.message.reply_text(
                    f"⬆️ Git push:\n```\n{result.stdout or result.stderr}\n```",
                    parse_mode="Markdown"
                )

            elif command == "log":
                n = int(context.args[1]) if len(context.args) > 1 else 5
                result = subprocess.run(
                    ["git", "log", f"-{n}", "--oneline"],
                    capture_output=True,
                    text=True,
                    cwd=REPO_PATH,
                )
                await update.message.reply_text(
                    f"📜 История коммитов:\n```\n{result.stdout}\n```",
                    parse_mode="Markdown"
                )

        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка git: {str(e)}")

    async def get_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Получение логов"""
        try:
            n_lines = int(context.args[0]) if context.args else 20
            log_file = REPO_PATH / "logs" / "prod_loop.log"

            if not log_file.exists():
                await update.message.reply_text("⚠️ Файл логов не найден")
                return

            result = subprocess.run(
                ["tail", f"-{n_lines}", str(log_file)],
                capture_output=True,
                text=True,
            )

            await update.message.reply_text(
                f"📝 Последние {n_lines} строк логов:\n\n```\n{result.stdout}\n```",
                parse_mode="Markdown"
            )

        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {str(e)}")

    async def run_fie(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Запуск FIE"""
        try:
            # Проверяем, не запущен ли уже
            result = subprocess.run(
                ["pgrep", "-f", "run_fie.py"],
                capture_output=True,
                text=True,
            )
            
            if result.stdout.strip():
                await update.message.reply_text("⚠️ FIE уже запущен")
                return

            # Запускаем в фоне
            subprocess.Popen(
                ["python", "run_fie.py"],
                cwd=REPO_PATH,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            await update.message.reply_text("✅ FIE запущен")

        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка запуска: {str(e)}")

    async def stop_fie(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Остановка FIE"""
        try:
            result = subprocess.run(
                ["pkill", "-f", "run_fie.py"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                await update.message.reply_text("✅ FIE остановлен")
            else:
                await update.message.reply_text("⚠️ FIE не был запущен")

        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {str(e)}")

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка нажатий на кнопки"""
        query = update.callback_query
        await query.answer()

        # Создаём псевдо-сообщение для переиспользования команд
        update.message = query.message

        command_map = {
            "status": self.status,
            "metrics": self.metrics,
            "git": self.git_status,
            "logs": self.get_logs,
            "run": self.run_fie,
            "stop": self.stop_fie,
        }

        handler = command_map.get(query.data)
        if handler:
            await handler(update, context)

    def _check_db_status(self) -> str:
        """Проверка статуса БД"""
        try:
            db_path = REPO_PATH / "db" / "prod" / "trades.db"
            if db_path.exists():
                size_mb = db_path.stat().st_size / (1024 * 1024)
                return f"💾 БД: `{size_mb:.2f} MB`"
            return "💾 БД: не найдена"
        except:
            return "💾 БД: ошибка проверки"

    def _get_metrics_from_db(self) -> Optional[dict]:
        """Получение метрик из БД"""
        try:
            import sqlite3
            
            db_path = REPO_PATH / "db" / "prod" / "trades.db"
            if not db_path.exists():
                return None

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Последние 100 сделок
            cursor.execute("""
                SELECT 
                    COUNT(*) as n_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
                    AVG(mfe) as avg_mfe,
                    AVG(mae) as avg_mae,
                    AVG(mfe) / NULLIF(ABS(AVG(mae)), 0) as capture_ratio,
                    SUM(CASE WHEN timeout_hit = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as timeout_rate,
                    SUM(CASE WHEN exit_reason LIKE '%tp%' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as tp_rate
                FROM Trade
                ORDER BY timestamp DESC
                LIMIT 100
            """)

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    "n_trades": int(row[0]),
                    "total_pnl": f"{row[1]:.2f}" if row[1] else "0.00",
                    "avg_pnl": f"{row[2]:.4f}" if row[2] else "0.0000",
                    "win_rate": f"{row[3]:.1f}" if row[3] else "0.0",
                    "avg_mfe": f"{row[4]:.4f}" if row[4] else "0.0000",
                    "avg_mae": f"{row[5]:.4f}" if row[5] else "0.0000",
                    "capture_ratio": f"{row[6]:.2f}" if row[6] else "0.00",
                    "timeout_rate": f"{row[7]:.1f}" if row[7] else "0.0",
                    "tp_rate": f"{row[8]:.1f}" if row[8] else "0.0",
                }

        except Exception as e:
            print(f"Error getting metrics: {e}")
            return None

    def run(self):
        """Запуск бота"""
        print("🤖 Hermes Bot запущен...")
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    if not BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN не найден в .env")
        sys.exit(1)

    bot = HermesBot()
    bot.run()


if __name__ == "__main__":
    main()

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / "future-intelligence-engine" / ".env")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_alert(message: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        raise EnvironmentError(
            "TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID должны быть заданы в .env"
        )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
    }

    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()

from alerts.formatter import format_alert
from alerts.telegram_bot import send_telegram_alert

EDGE_THRESHOLD = 0.15


def check_and_alert(event: str, market_data: dict) -> None:
    if not market_data:
        return

    edge = market_data["edge"]

    if abs(edge) > EDGE_THRESHOLD:
        message = format_alert(event, market_data)
        send_telegram_alert(message)

def format_alert(event: str, market_data: dict) -> str:
    return (
        f"⚡ FIE HIGH EDGE\n\n"
        f"Event: {event}\n\n"
        f"FIE: {round(market_data['fie_probability'] * 100, 1)}%\n"
        f"Market: {round(market_data['market_probability'] * 100, 1)}%\n\n"
        f"Edge: {market_data['edge']}\n\n"
        f"Signal: {market_data['signal']}"
    )

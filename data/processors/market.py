from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from data.collectors.market import get_btc_ohlcv
except ModuleNotFoundError:
    # Поддержка запуска файлом: `python data/processors/market.py`
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from data.collectors.market import get_btc_ohlcv


def _to_iso_utc(timestamp_ms: int | float) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def normalize_market_rows(rows: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for row in rows:
        ts = row["timestamp"]
        p = float(row["price"])
        out = {
            "timestamp": int(ts),
            "datetime_utc": _to_iso_utc(ts),
            "symbol": "BTC",
            "price_usd": p,
            "volume_usd": float(row["volume"]),
        }
        if "high" in row:
            out["high_usd"] = float(row["high"])
            out["low_usd"] = float(row["low"])
        normalized.append(out)
    return normalized


def save_market_dataset(rows: list[dict], path: str = "data/datasets/btc_market_30d.json") -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return output_path


def build_and_save_market_dataset(path: str = "data/datasets/btc_market_30d.json") -> Path:
    raw_rows = get_btc_ohlcv()
    normalized_rows = normalize_market_rows(raw_rows)
    return save_market_dataset(normalized_rows, path=path)


if __name__ == "__main__":
    dataset_path = build_and_save_market_dataset()
    print(f"Market dataset saved to: {dataset_path}")

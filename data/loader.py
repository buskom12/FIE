import json
from pathlib import Path


def load_historical_data(path: str = "data/historical_events.json") -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Файл с историческими данными не найден: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from learning.role_weights import load_role_stats, load_role_weights, save_role_stats, save_role_weights

DECISIONS_LOG = Path(__file__).resolve().parent.parent / "data" / "logs" / "collective_intelligence.jsonl"


def _read_rows() -> list[dict[str, Any]]:
    if not DECISIONS_LOG.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with DECISIONS_LOG.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _write_rows(rows: list[dict[str, Any]]) -> None:
    DECISIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DECISIONS_LOG.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def mark_latest_outcome(event: str, outcome: float) -> bool:
    """
    Проставляет real outcome для последней записи события, где outcome ещё не задан.
    """
    rows = _read_rows()
    for i in range(len(rows) - 1, -1, -1):
        row = rows[i]
        if row.get("event") == event and row.get("outcome") is None:
            row["outcome"] = 1.0 if float(outcome) >= 0.5 else 0.0
            row["learned"] = False
            rows[i] = row
            _write_rows(rows)
            return True
    return False


def run_learning_update(
    *,
    reward_factor: float = 1.08,
    penalty_factor: float = 0.92,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Обновляет веса ролей по всем лог-записям с outcome != None и learned != True.
    """
    rows = _read_rows()
    weights = load_role_weights()
    role_stats = load_role_stats()
    learned_rows = 0
    role_updates: dict[str, int] = {}

    for idx, row in enumerate(rows):
        if row.get("learned") is True:
            continue
        outcome = row.get("outcome")
        if outcome is None:
            continue
        votes = row.get("agents_votes")
        if not isinstance(votes, list) or not votes:
            rows[idx]["learned"] = True
            learned_rows += 1
            continue

        actual = 1 if float(outcome) >= threshold else 0
        for v in votes:
            if not isinstance(v, dict):
                continue
            role = str(v.get("role", "")).strip()
            if not role:
                continue
            prob = float(v.get("probability", 0.5))
            pred = 1 if prob >= threshold else 0
            cur_w = float(weights.get(role, 1.0))
            if pred == actual:
                cur_w *= reward_factor
                is_correct = 1.0
            else:
                cur_w *= penalty_factor
                is_correct = 0.0
            weights[role] = max(0.1, min(5.0, cur_w))
            role_updates[role] = role_updates.get(role, 0) + 1

            st = role_stats.get(role, {"accuracy": 0.5, "count": 0.0})
            prev_acc = float(st.get("accuracy", 0.5))
            prev_count = float(st.get("count", 0.0))
            alpha = 0.1
            new_acc = (1.0 - alpha) * prev_acc + alpha * is_correct
            role_stats[role] = {"accuracy": max(0.0, min(1.0, new_acc)), "count": prev_count + 1.0}

        rows[idx]["learned"] = True
        rows[idx]["learning_applied"] = {
            "reward_factor": reward_factor,
            "penalty_factor": penalty_factor,
            "threshold": threshold,
        }
        learned_rows += 1

    save_role_weights(weights)
    save_role_stats(role_stats)
    _write_rows(rows)
    return {
        "processed_rows": learned_rows,
        "role_updates": role_updates,
        "weights": weights,
        "role_stats": role_stats,
        "log_path": str(DECISIONS_LOG),
    }

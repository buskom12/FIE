from __future__ import annotations

from collections import defaultdict


class InteractionEngine:
    def __init__(self):
        # stats for pair interactions (signal_i, signal_j)
        self.pair_stats = defaultdict(lambda: {"pos": 0.0, "total": 0.0})

        # stats for conditional weights: P(outcome=1 | signal present, regime)
        # key = (signal, regime_label)
        self.conditional_stats = defaultdict(lambda: {"pos": 0.0, "total": 0.0})

        # ВАЖНО: защита от шума
        self.min_pair_total = 10

    def update(
        self,
        pair_signals: dict[str, float],
        outcome: int,
        *,
        regime_label: str | None = None,
        conditional_signals: dict[str, float] | None = None,
    ) -> None:
        # Pair stats: only among provided keys.
        keys = list(pair_signals.keys())

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                pair = tuple(sorted([keys[i], keys[j]]))
                self.pair_stats[pair]["total"] += 1.0
                self.pair_stats[pair]["pos"] += float(outcome)

        # Conditional stats: key=(signal, regime_label)
        if regime_label is None or not conditional_signals:
            return

        for signal, v in conditional_signals.items():
            if v <= 0.0:
                continue
            key = (signal, regime_label)
            self.conditional_stats[key]["total"] += 1.0
            self.conditional_stats[key]["pos"] += float(outcome)

    def get_weight(self, pair: tuple[str, str]) -> float | None:
        s = self.pair_stats.get(pair)

        if not s or s["total"] < self.min_pair_total:
            return None

        return (s["pos"] + 1) / (s["total"] + 2)

    def get_conditional_weight(self, signal: str, regime_label: str) -> float | None:
        key = (signal, regime_label)
        s = self.conditional_stats.get(key)
        if not s or s["total"] < self.min_pair_total:
            return None
        return (s["pos"] + 1.0) / (s["total"] + 2.0)

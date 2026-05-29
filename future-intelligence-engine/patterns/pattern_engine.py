from dataclasses import dataclass, field


@dataclass
class HistoricalCase:
    signals: list[dict]
    outcome: float  # 1 = positive event, 0 = negative event
    label: str = ""


class PatternEngine:
    """
    Ищет исторические паттерны по сигналам и вычисляет вероятность исхода.

    Логика:
        data → signals → pattern match → probability
    """

    def __init__(self, overlap_threshold: float = 0.6):
        self.history: list[HistoricalCase] = []
        self.overlap_threshold = overlap_threshold

    # ------------------------------------------------------------------
    # Наполнение истории
    # ------------------------------------------------------------------

    def add_case(self, signals: list[dict], outcome: float, label: str = "") -> None:
        self.history.append(HistoricalCase(signals=signals, outcome=outcome, label=label))

    def add_cases(self, cases: list[dict]) -> None:
        for c in cases:
            self.add_case(
                signals=c["signals"],
                outcome=c["outcome"],
                label=c.get("label", ""),
            )

    # ------------------------------------------------------------------
    # Поиск совпадений
    # ------------------------------------------------------------------

    def find_patterns(self, current_signals: list[dict]) -> list[HistoricalCase]:
        return [
            case
            for case in self.history
            if self._signal_overlap(case.signals, current_signals) >= self.overlap_threshold
        ]

    def compute_probability(self, matches: list[HistoricalCase]) -> float:
        if not matches:
            return 0.5
        return sum(m.outcome for m in matches) / len(matches)

    def analyze(self, current_signals: list[dict]) -> dict:
        """Единый вход: сигналы → результат анализа."""
        matches = self.find_patterns(current_signals)
        probability = self.compute_probability(matches)
        return {
            "probability": round(probability, 4),
            "matched_cases": len(matches),
            "total_history": len(self.history),
            "confidence": self._confidence(matches),
        }

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _signal_overlap(self, s1: list[dict], s2: list[dict]) -> float:
        set1 = {s["type"] for s in s1}
        set2 = {s["type"] for s in s2}
        if not set1:
            return 0.0
        return len(set1 & set2) / max(len(set1), len(set2))

    def _confidence(self, matches: list[HistoricalCase]) -> str:
        n = len(matches)
        if n == 0:
            return "none"
        if n < 3:
            return "low"
        if n < 10:
            return "medium"
        return "high"

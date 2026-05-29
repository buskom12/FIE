from collections import Counter
import re


def detect_signals(texts: list[str]) -> list[dict]:
    words = []

    for t in texts:
        tokens = re.findall(r'\w+', t.lower())
        words.extend(tokens)

    counter = Counter(words)

    return [
        {"keyword": word, "frequency": count}
        for word, count in counter.items()
        if count > 3 and len(word) > 4
    ]

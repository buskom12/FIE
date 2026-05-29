from signals.data_sources import collect_text_stream
from signals.signal_engine import detect_signals


def run_signal_scan() -> None:
    texts = collect_text_stream()
    signals = detect_signals(texts)

    print("Detected signals:")
    for s in signals:
        print(s)


if __name__ == "__main__":
    run_signal_scan()

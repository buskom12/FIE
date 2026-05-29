"""Event discovery pipeline — fetches, scores and ranks top events."""

from __future__ import annotations

import feedparser
import requests

from prediction.event_impact import score_event_impact

FETCH_TIMEOUT = 8  # seconds per source


NEWS_SOURCES = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptoslate.com/feed/",
]

RSS_SOURCES = {
    "CoinTelegraph": NEWS_SOURCES[0],
    "CoinDesk":      NEWS_SOURCES[1],
    "CryptoSlate":   NEWS_SOURCES[2],
}

MAX_PER_SOURCE = 5


def _parse_feed(url: str) -> feedparser.FeedParserDict:
    """Fetches and parses an RSS feed with a hard timeout via requests."""
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": "FIE/1.0"})
        resp.raise_for_status()
        return feedparser.parse(resp.content)
    except Exception:
        return feedparser.FeedParserDict(entries=[])


def discover_events() -> list[str]:
    """Returns a flat list of event titles from all NEWS_SOURCES."""
    events = []
    for source in NEWS_SOURCES:
        feed = _parse_feed(source)
        for entry in feed.entries[:MAX_PER_SOURCE]:
            events.append(entry.get("title", "").strip())
    return [e for e in events if e]


def _fetch_from_source(name: str, url: str) -> list[dict]:
    """Fetches up to MAX_PER_SOURCE entries from a single RSS feed."""
    try:
        feed = _parse_feed(url)
        entries = []
        for entry in feed.entries[:MAX_PER_SOURCE]:
            entries.append({
                "title":     entry.get("title", "").strip(),
                "summary":   entry.get("summary", entry.get("description", "")).strip(),
                "source":    name,
                "link":      entry.get("link", ""),
                "published": entry.get("published", ""),
            })
        return entries
    except Exception as exc:
        print(f"[event_discovery] Failed to fetch '{name}': {exc}")
        return []


def fetch_all_events() -> list[dict]:
    """Fetches raw events from all RSS sources."""
    events = []
    for name, url in RSS_SOURCES.items():
        events.extend(_fetch_from_source(name, url))
    return events


def score_events(events: list[dict]) -> list[dict]:
    """
    Scores each event with an impact score (0.0–1.0).
    Uses event_impact module; falls back to 0.5 if unavailable.
    """
    scored = []
    for event in events:
        impact = score_event_impact(event["title"])
        scored.append({**event, "impact_score": round(impact, 3)})
    return scored


def discover_top_events(n: int = 5, score: bool = False) -> list[dict]:
    """
    Main entry point. Returns the top-n events sorted by impact score.

    Args:
        n:     Number of top events to return.
        score: If True, calls LLM to score each event (slower).
               If False, returns events without scoring (fast).
    """
    events = fetch_all_events()

    if not events:
        return []

    if score:
        events = score_events(events)
        events.sort(key=lambda e: e["impact_score"], reverse=True)
    
    return events[:n]


def get_top_event(score: bool = False) -> dict | None:
    """Returns the single most impactful event, or None if no events found."""
    results = discover_top_events(n=1, score=score)
    return results[0] if results else None

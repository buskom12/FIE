from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import requests


DEFAULT_RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
]


def _parse_rss_datetime(value: str | None) -> str | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def _safe_text(element: ET.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    text = element.text.strip()
    return text or None


def get_rss_news(
    feeds: list[str] | None = None,
    timeout: int = 20,
    limit_per_feed: int = 20,
) -> list[dict]:
    feed_urls = feeds or DEFAULT_RSS_FEEDS
    events: list[dict] = []

    for feed_url in feed_urls:
        response = requests.get(feed_url, timeout=timeout)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        items = root.findall(".//item")[:limit_per_feed]

        for item in items:
            title = _safe_text(item.find("title"))
            link = _safe_text(item.find("link"))
            published_raw = _safe_text(item.find("pubDate"))
            published_at = _parse_rss_datetime(published_raw)

            events.append(
                {
                    "source": "rss",
                    "feed": feed_url,
                    "title": title,
                    "url": link,
                    "published_at": published_at,
                    "published_raw": published_raw,
                }
            )

    events.sort(key=lambda x: x.get("published_at") or "", reverse=True)
    return events


def get_polymarket_events(limit: int = 20, timeout: int = 20) -> list[dict]:
    url = "https://gamma-api.polymarket.com/events"
    response = requests.get(url, params={"limit": limit, "closed": "false"}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    events: list[dict] = []
    for item in payload:
        title = item.get("title")
        slug = item.get("slug")
        events.append(
            {
                "source": "polymarket",
                "title": title,
                "url": f"https://polymarket.com/event/{slug}" if slug else None,
                "published_at": item.get("startDate"),
                "status": item.get("status"),
            }
        )

    return events


def get_x_posts(query: str = "bitcoin", limit: int = 20, bearer_token: str | None = None) -> list[dict]:
    # X API требует аутентификацию; здесь оставляем минимальную заглушку для pipeline.
    _ = (query, limit, bearer_token)
    return []


def collect_news_events() -> list[dict]:
    events: list[dict] = []
    events.extend(get_rss_news())

    try:
        events.extend(get_polymarket_events())
    except requests.RequestException:
        # Полимаркет может быть временно недоступен; не валим весь сбор.
        pass

    events.extend(get_x_posts())
    events.sort(key=lambda x: x.get("published_at") or "", reverse=True)
    return events


if __name__ == "__main__":
    snapshot = collect_news_events()
    print(f"Collected news/events: {len(snapshot)}")

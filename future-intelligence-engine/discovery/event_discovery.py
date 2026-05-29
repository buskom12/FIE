import feedparser

RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
]


def discover_events() -> list[dict]:
    events = []

    for feed_url in RSS_FEEDS:
        parsed = feedparser.parse(feed_url)

        for entry in parsed.entries[:5]:
            events.append({
                "title": entry.title,
                "link": entry.link,
            })

    return events

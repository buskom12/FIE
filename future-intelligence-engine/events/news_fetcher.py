import feedparser


def fetch_crypto_news():

    url = "https://cointelegraph.com/rss"

    feed = feedparser.parse(url)

    events = []

    for entry in feed.entries[:5]:

        events.append({
            "title": entry.title,
            "summary": entry.summary
        })

    return events

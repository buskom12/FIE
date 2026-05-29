from events.event_discovery import discover_events

events = discover_events()

for e in events:
    print(e)

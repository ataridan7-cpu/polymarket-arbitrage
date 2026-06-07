"""
Smarkets Exchange market data — public API, no authentication required.

Smarkets is a UK-based prediction/betting exchange. US-accessible for data
reads; trading from US may be restricted. Covers US politics but NOT Fed rates.

Fee model: 2% commission on net winnings of the winning leg.
  smarkets_fee(price) = 0.02 × (1 − price)

Prices are returned as integers in 1/10000 units (e.g., 7463 = 74.63%).
"""

import httpx

SMARKETS_URL = "https://api.smarkets.com/v3"

_US_KEYWORDS = [
    "trump", "biden", "harris", "senate", "house", "congress",
    "democratic", "republican", "president", "midterm", "election",
    "approval", "federal", "us ", "u.s.", "american", "america",
    "california", "new york", "texas", "florida", "governor",
]


def smarkets_fee(price: float) -> float:
    """2% commission on net winnings of the winning leg."""
    p = max(0.0, min(1.0, float(price)))
    return 0.02 * (1.0 - p)


def _is_us_related(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in _US_KEYWORDS)


def get_smarkets_markets(timeout: float = 15.0) -> list[dict]:
    """
    Fetch open Smarkets binary (YES/NO) markets for US politics events.

    API flow: events → markets → contracts + quotes (3 calls per event).
    Returns list of dicts: question, yes_ask, yes_bid, no_ask, no_bid, mid, url, platform.
    """
    results = []
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True,
                          headers={"User-Agent": "arb-scanner/1.0"}) as client:
            # 1. Fetch politics events
            r = client.get(f"{SMARKETS_URL}/events/",
                           params={"type": "politics", "limit": 200, "state": "upcoming"})
            if r.status_code != 200:
                return []
            events = r.json().get("events", [])

            # Filter to US-related events only
            us_events = [e for e in events if _is_us_related(e.get("name", ""))]

            for event in us_events:
                eid = event["id"]
                event_name = event.get("name", "")
                event_slug = event.get("slug", "")
                event_url = f"https://smarkets.com/event/{eid}/{event_slug}"

                # 2. Fetch markets for this event
                r2 = client.get(f"{SMARKETS_URL}/events/{eid}/markets/")
                if r2.status_code != 200:
                    continue
                markets = r2.json().get("markets", [])

                for market in markets:
                    if market.get("complete"):
                        continue
                    mid = market["id"]
                    market_name = market.get("name", "")

                    # 3a. Fetch contracts
                    r3 = client.get(f"{SMARKETS_URL}/markets/{mid}/contracts/")
                    if r3.status_code != 200:
                        continue
                    contracts = r3.json().get("contracts", [])

                    # Only process 2-contract markets (binary outcomes)
                    if len(contracts) != 2:
                        continue

                    names_lower = {c["name"].lower() for c in contracts}

                    # Identify YES/NO contracts. Accept:
                    #   (a) explicit "yes"/"no" contracts
                    #   (b) "democrat"/"republican" — treat Democrat as YES
                    if {"yes", "no"} <= names_lower:
                        yes_c = next(c for c in contracts if c["name"].lower() == "yes")
                        no_c  = next(c for c in contracts if c["name"].lower() == "no")
                        question = market_name if market_name.lower().strip() != event_name.lower().strip() else event_name
                    elif {"democrat", "republican"} <= names_lower or {"democratic", "republican"} <= names_lower:
                        yes_c = next(c for c in contracts if c["name"].lower().startswith("democrat"))
                        no_c  = next(c for c in contracts if c["name"].lower() == "republican")
                        # Phrase question so matchers understand it as "Democrat wins?"
                        question = f"{market_name} — Democrat"
                    else:
                        continue

                    # 3b. Fetch quotes (orderbook)
                    r4 = client.get(f"{SMARKETS_URL}/markets/{mid}/quotes/")
                    if r4.status_code != 200:
                        continue
                    quotes = r4.json()

                    def best_price(contract_id, side):
                        entries = quotes.get(str(contract_id), {}).get(side, [])
                        if not entries:
                            return None
                        if side == "bids":
                            return max(e["price"] for e in entries) / 10000
                        return min(e["price"] for e in entries) / 10000

                    yes_bid = best_price(yes_c["id"], "bids")
                    yes_ask = best_price(yes_c["id"], "offers")
                    no_bid  = best_price(no_c["id"],  "bids")
                    no_ask  = best_price(no_c["id"],  "offers")

                    if yes_ask is None and yes_bid is None:
                        continue

                    mid_price = ((yes_ask or 0) + (yes_bid or 0)) / 2 if (yes_ask and yes_bid) else (yes_ask or yes_bid or 0)

                    results.append({
                        "question": question,
                        "yes_ask":  round(yes_ask, 4) if yes_ask else None,
                        "yes_bid":  round(yes_bid, 4) if yes_bid else None,
                        "no_ask":   round(no_ask, 4)  if no_ask  else None,
                        "no_bid":   round(no_bid, 4)  if no_bid  else None,
                        "mid":      round(mid_price, 4),
                        "url":      event_url,
                        "platform": "smarkets",
                    })

    except Exception as e:
        print(f"  [Smarkets] fetch error: {e}")

    return results

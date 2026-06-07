"""
Betfair Exchange market data.

Requires account credentials in environment variables:
  BETFAIR_APP_KEY   — API app key (Account > API > My Application Keys)
  BETFAIR_USERNAME  — account email / username
  BETFAIR_PASSWORD  — account password

How to get credentials (takes ~5 minutes):
  1. Register at betfair.com (Israel is accepted)
  2. Account menu → My Account → Security & API → My Application Keys
  3. Click "Get a Free Application Key" → copy the "Delayed Data" key
  4. Set env vars:
       export BETFAIR_APP_KEY="YourKeyHere"
       export BETFAIR_USERNAME="your@email.com"
       export BETFAIR_PASSWORD="yourpassword"

Fee model: 5% commission on net market winnings.
  betfair_fee(price) = 0.05 × (1 − price)
  This is cheaper than PredictIt (<80¢) and Polymarket (>53¢).

Decimal odds ↔ probability: probability = 1 / decimal_odds
  e.g. odds 2.0 → 50¢, odds 1.5 → 66.7¢, odds 4.0 → 25¢
"""

import os
import httpx

BETFAIR_LOGIN_URL = "https://identitysso.betfair.com/api/login"
BETFAIR_API_URL   = "https://api.betfair.com/exchange/betting/rest/v1.0"

BETFAIR_POLITICS_TYPE_ID = "2378961"
BETFAIR_COMMISSION = 0.05   # default 5% on net winnings


def betfair_fee(price: float) -> float:
    """5% commission on net market winnings of the winning leg."""
    p = max(0.0, min(1.0, float(price)))
    return BETFAIR_COMMISSION * (1.0 - p)


def _creds():
    return (
        os.getenv("BETFAIR_APP_KEY"),
        os.getenv("BETFAIR_USERNAME"),
        os.getenv("BETFAIR_PASSWORD"),
    )


def _login(client: httpx.Client, app_key: str, username: str, password: str) -> str | None:
    try:
        r = client.post(
            BETFAIR_LOGIN_URL,
            data={"username": username, "password": password},
            headers={
                "X-Application": app_key,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        if r.status_code != 200:
            return None
        d = r.json()
        if d.get("status") != "SUCCESS":
            print(f"  [Betfair] login error: {d.get('error', 'unknown')}")
            return None
        return d.get("token")
    except Exception as e:
        print(f"  [Betfair] login exception: {e}")
        return None


def _api(client: httpx.Client, app_key: str, token: str, endpoint: str, body: dict) -> list | dict:
    r = client.post(
        f"{BETFAIR_API_URL}/{endpoint}/",
        json=body,
        headers={
            "X-Application":  app_key,
            "X-Authentication": token,
            "Content-Type":   "application/json",
            "Accept":         "application/json",
        },
    )
    r.raise_for_status()
    return r.json()


def get_betfair_markets(timeout: float = 25.0) -> list[dict]:
    """
    Fetch Betfair binary political markets with live best-available prices.
    Returns [] with a hint message if credentials are not configured.
    """
    app_key, username, password = _creds()
    if not all([app_key, username, password]):
        print(
            "  [Betfair] credentials not set. To enable:\n"
            "    export BETFAIR_APP_KEY='...'  BETFAIR_USERNAME='...'  BETFAIR_PASSWORD='...'\n"
            "  Get your free API key at: betfair.com → Account → My Application Keys"
        )
        return []

    results = []
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True,
                          headers={"User-Agent": "arb-scanner/1.0"}) as client:

            token = _login(client, app_key, username, password)
            if not token:
                return []

            # ── 1. List political events ───────────────────────────────────
            events_resp = _api(client, app_key, token, "listEvents", {
                "filter": {"eventTypeIds": [BETFAIR_POLITICS_TYPE_ID]},
            })
            event_ids = [e["event"]["id"] for e in events_resp]
            if not event_ids:
                return []

            # ── 2. Get market catalogue (names + runner descriptions) ──────
            catalogs = _api(client, app_key, token, "listMarketCatalogue", {
                "filter": {
                    "eventTypeIds": [BETFAIR_POLITICS_TYPE_ID],
                    "eventIds": event_ids,
                    "marketCountries": ["US", "GB"],     # US politics + global
                    "marketTypeCodes": ["WINNER", "MATCH_ODDS", "NEXT_MANAGER"],
                },
                "maxResults": "500",
                "marketProjection": ["MARKET_NAME", "RUNNER_DESCRIPTION", "EVENT"],
            })

            # ── 3. Fetch live prices in batches of 40 ─────────────────────
            market_ids   = [m["marketId"] for m in catalogs]
            catalog_map  = {m["marketId"]: m for m in catalogs}
            runner_names = {
                m["marketId"]: {
                    r["selectionId"]: r.get("runnerName", "")
                    for r in m.get("runners", [])
                }
                for m in catalogs
            }

            for i in range(0, len(market_ids), 40):
                batch = market_ids[i : i + 40]
                books = _api(client, app_key, token, "listMarketBook", {
                    "marketIds": batch,
                    "priceProjection": {"priceData": ["EX_BEST_OFFERS"]},
                })

                for book in books:
                    mid     = book["marketId"]
                    runners = book.get("runners", [])
                    cat     = catalog_map.get(mid, {})

                    active  = [r for r in runners if r.get("status") == "ACTIVE"]
                    if len(active) != 2:
                        continue   # only binary markets

                    names  = runner_names.get(mid, {})
                    r_yes  = active[0]
                    r_no   = active[1]
                    name_yes = names.get(r_yes["selectionId"], "Yes")
                    name_no  = names.get(r_no["selectionId"],  "No")

                    def best_back(runner):
                        backs = runner.get("ex", {}).get("availableToBack", [])
                        # Sorted descending by decimal odds (best for backer = highest odds first)
                        return backs[0]["price"] if backs else None

                    def best_lay(runner):
                        lays = runner.get("ex", {}).get("availableToLay", [])
                        return lays[0]["price"] if lays else None

                    def odds_to_prob(decimal_odds):
                        if not decimal_odds or decimal_odds <= 1.0:
                            return None
                        return round(1.0 / decimal_odds, 4)

                    yes_back_odds = best_back(r_yes)   # decimal odds to BUY YES
                    yes_lay_odds  = best_lay(r_yes)    # decimal odds to SELL YES
                    no_back_odds  = best_back(r_no)    # decimal odds to BUY NO

                    yes_ask = odds_to_prob(yes_back_odds)   # cost to buy YES
                    yes_bid = odds_to_prob(yes_lay_odds)    # price at which someone buys YES from you
                    no_ask  = odds_to_prob(no_back_odds)    # cost to buy NO

                    if yes_ask is None:
                        continue

                    mid_price = ((yes_ask + yes_bid) / 2) if yes_bid else yes_ask

                    market_name = cat.get("marketName", "")
                    event_name  = cat.get("event", {}).get("name", "")
                    question = f"{market_name} — {name_yes}" if name_yes.lower() not in ("yes", "true") else (market_name or event_name)

                    event_id = cat.get("event", {}).get("id", "")
                    url = f"https://www.betfair.com/exchange/plus/politics/market/{mid}"

                    results.append({
                        "question": question,
                        "yes_ask":  yes_ask,
                        "yes_bid":  yes_bid,
                        "no_ask":   no_ask,
                        "no_bid":   None,
                        "mid":      round(mid_price, 4),
                        "url":      url,
                        "platform": "betfair",
                        "market_id": mid,
                        "name_yes":  name_yes,
                        "name_no":   name_no,
                    })

    except Exception as e:
        print(f"  [Betfair] fetch error: {e}")

    return results

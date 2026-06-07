"""
PredictIt market data — public API, no authentication required.

PredictIt fee model: 10% of profits on any winning position.
Per-contract fee = 0.10 × (1 − entry_price)  [for a YES buy that resolves YES]
"""

import httpx

PREDICTIT_URL = "https://www.predictit.org/api/marketdata/all/"


def predictit_fee(price: float) -> float:
    """
    Fee on a YES contract bought at `price` if it resolves YES.
    PredictIt takes 10% of the $1 − price profit.
    """
    p = max(0.0, min(1.0, float(price)))
    return 0.10 * (1.0 - p)


def get_predictit_markets(timeout: float = 15.0) -> list[dict]:
    """
    Fetch all open PredictIt binary-outcome contracts.

    Returns a list of dicts with keys:
      question, yes_ask, yes_bid, no_ask, no_bid, mid, url, platform
    """
    try:
        r = httpx.get(
            PREDICTIT_URL,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; arb-scanner/1.0)"},
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [PredictIt] fetch error: {e}")
        return []

    results = []
    for mkt in data.get("markets", []):
        name    = mkt.get("name") or ""
        mkt_url = (
            mkt.get("url")
            or f"https://www.predictit.org/markets/detail/{mkt.get('id', 0)}"
        )
        for contract in mkt.get("contracts") or []:
            if contract.get("status") != "Open":
                continue
            ya = contract.get("bestBuyYesCost")    # best ask to buy YES
            yb = contract.get("bestSellYesCost")   # best bid for YES sellers
            na = contract.get("bestBuyNoCost")     # best ask to buy NO
            nb = contract.get("bestSellNoCost")    # best bid for NO sellers
            if not ya or not na:
                continue
            cname    = (contract.get("name") or "").strip()
            question = (
                f"{name} — {cname}"
                if cname and cname.lower() not in name.lower()
                else name
            )
            mid = (float(ya) + float(yb)) / 2 if yb else float(ya)
            results.append({
                "question": question,
                "yes_ask":  round(float(ya), 4),
                "yes_bid":  round(float(yb), 4) if yb else None,
                "no_ask":   round(float(na), 4),
                "no_bid":   round(float(nb), 4) if nb else None,
                "mid":      round(mid, 4),
                "url":      mkt_url,
                "platform": "predictit",
            })
    return results

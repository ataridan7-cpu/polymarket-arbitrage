"""
Manifold Markets data — public API, no authentication required.

IMPORTANT: Manifold uses play money (mana), NOT real USD.
Prices here are crowd probability estimates, not real-money bets.
Use for price-signal comparison only — divergences from Polymarket/Kalshi
may indicate genuine mispricings worth researching.

Fee model: 0% (free to trade on Manifold, no real money at stake).
"""

import httpx

MANIFOLD_URL = "https://api.manifold.markets/v0"


def get_manifold_markets(timeout: float = 20.0, limit: int = 500) -> list[dict]:
    """
    Fetch open Manifold binary markets sorted by liquidity.

    Returns a list of dicts with keys:
      question, yes_ask, yes_bid, no_ask, no_bid, mid, url, platform
    Probability from the AMM is used as both bid and ask (no spread model).
    """
    try:
        r = httpx.get(
            f"{MANIFOLD_URL}/markets",
            params={"limit": limit},
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "arb-scanner/1.0"},
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [Manifold] fetch error: {e}")
        return []

    # Filter to open binary markets, sort by liquidity descending
    open_binary = [
        m for m in data
        if m.get("outcomeType") == "BINARY"
        and not m.get("isResolved")
        and m.get("probability") is not None
    ]
    open_binary.sort(key=lambda m: float(m.get("totalLiquidity") or 0), reverse=True)

    results = []
    for m in open_binary:
        p = float(m["probability"])
        results.append({
            "question": (m.get("question") or "").strip(),
            "yes_ask":  round(p, 4),
            "yes_bid":  round(p, 4),
            "no_ask":   round(1.0 - p, 4),
            "no_bid":   round(1.0 - p, 4),
            "mid":      round(p, 4),
            "url":      m.get("url") or "",
            "platform": "manifold",
        })
    return results

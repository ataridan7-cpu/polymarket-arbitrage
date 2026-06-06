#!/usr/bin/env python3
"""
Fast Arbitrage Scanner
======================
Scans Polymarket for:
1. Bundle arb: YES ask + NO ask < $1.00 (guaranteed profit)
2. Bundle arb: YES bid + NO bid > $1.00
3. Wide spreads worth market-making
Focuses on contested markets (YES mid-price between 10c–90c).
"""

import asyncio
import httpx
import json
from dataclasses import dataclass, field
from typing import Optional


GAMMA_URL  = "https://gamma-api.polymarket.com"
CLOB_URL   = "https://clob.polymarket.com"
TAKER_FEE  = 0.015     # 1.5% taker fee per leg
MIN_EDGE   = 0.005     # 0.5% net edge after fees
MM_SPREAD  = 0.02      # flag spreads ≥ 2¢ (strong ≥ 4¢, marginal 2–4¢)
FETCH_N    = 500       # fetch this many markets from Gamma
SCAN_CAP   = 200       # scan at most this many after filtering
MIN_VOLUME = 500       # skip markets with < $500 daily volume
CONTESTED  = (0.08, 0.92)   # only contested markets (YES mid in this range)


@dataclass
class BookSide:
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread: Optional[float] = None
    mid: Optional[float] = None

    def from_book(self, book: dict) -> "BookSide":
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        bb = max((float(x["price"]) for x in bids), default=None) if bids else None
        ba = min((float(x["price"]) for x in asks), default=None) if asks else None
        self.best_bid = bb
        self.best_ask = ba
        if bb and ba:
            self.spread = round(ba - bb, 4)
            self.mid    = round((ba + bb) / 2, 4)
        elif bb:
            self.mid = bb
        elif ba:
            self.mid = ba
        return self


async def fetch_markets(client: httpx.AsyncClient, n: int) -> list[dict]:
    all_markets = []
    offset = 0
    per_page = 100
    while len(all_markets) < n:
        r = await client.get(f"{GAMMA_URL}/markets", params={
            "closed": "false", "active": "true",
            "order": "volume24hr", "ascending": "false",
            "limit": per_page, "offset": offset,
        }, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        all_markets.extend(batch)
        if len(batch) < per_page:
            break
        offset += per_page
        await asyncio.sleep(0.1)
    return all_markets[:n]


async def fetch_book(client: httpx.AsyncClient, token_id: str) -> Optional[dict]:
    try:
        r = await client.get(f"{CLOB_URL}/book", params={"token_id": token_id}, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def top3_depth(book: dict, side: str) -> float:
    """Total size at the top 3 price levels on the given side ('bids' or 'asks')."""
    levels = book.get(side, [])
    reverse = (side == "bids")
    levels_sorted = sorted(levels, key=lambda x: float(x["price"]), reverse=reverse)
    return sum(float(x.get("size", 0)) for x in levels_sorted[:3])


async def scan_one(client: httpx.AsyncClient, m: dict) -> Optional[dict]:
    try:
        tok_raw = m.get("clobTokenIds", "")
        if not tok_raw:
            return None
        toks = json.loads(tok_raw)
        if len(toks) < 2:
            return None
        yes_id, no_id = str(toks[0]), str(toks[1])
    except Exception:
        return None

    yb_raw, nb_raw = await asyncio.gather(
        fetch_book(client, yes_id),
        fetch_book(client, no_id),
    )
    if not yb_raw or not nb_raw:
        return None

    yes = BookSide().from_book(yb_raw)
    no  = BookSide().from_book(nb_raw)

    if yes.mid is None or no.mid is None:
        return None

    # Skip non-contested markets
    if not (CONTESTED[0] <= yes.mid <= CONTESTED[1]):
        return None

    vol = float(m.get("volume24hr") or 0)
    if vol < MIN_VOLUME:
        return None

    # Bundle BUY arb: pay (yes_ask + no_ask), collect $1 at resolution
    edge_buy = None
    if yes.best_ask and no.best_ask:
        cost      = yes.best_ask + no.best_ask
        fees      = cost * TAKER_FEE
        edge_buy  = round(1.0 - cost - fees, 4)

    # Bundle SELL arb: collect (yes_bid + no_bid), owe $1 at resolution
    edge_sell = None
    if yes.best_bid and no.best_bid:
        revenue   = yes.best_bid + no.best_bid
        fees      = revenue * TAKER_FEE
        edge_sell = round(revenue - fees - 1.0, 4)

    slug = m.get("slug") or m.get("market_slug") or ""
    yes_bid_depth = top3_depth(yb_raw, "bids")
    yes_ask_depth = top3_depth(yb_raw, "asks")

    return {
        "id":            str(m.get("id", "")),
        "question":      (m.get("question") or "?")[:90],
        "vol24h":        vol,
        "yes":           yes,
        "no":            no,
        "edge_buy":      edge_buy,
        "edge_sell":     edge_sell,
        "slug":          slug,
        "yes_bid_depth": yes_bid_depth,
        "yes_ask_depth": yes_ask_depth,
    }


async def main():
    print("=" * 72)
    print("  Polymarket Arbitrage Scanner  —  live contested-market scan")
    print("=" * 72)

    async with httpx.AsyncClient() as client:
        print(f"\n[1/3] Fetching up to {FETCH_N} markets by 24h volume...")
        raw = await fetch_markets(client, FETCH_N)
        print(f"      Retrieved {len(raw)} markets from Gamma API")

        # Pre-filter: only keep markets with volume
        raw = [m for m in raw if float(m.get("volume24hr") or 0) >= MIN_VOLUME]
        print(f"      {len(raw)} markets with ≥ ${MIN_VOLUME:,} daily volume")

        to_scan = raw[:SCAN_CAP]
        print(f"\n[2/3] Scanning {len(to_scan)} markets for arb & MM gaps...")
        results = []
        batch_size = 25

        for i in range(0, len(to_scan), batch_size):
            batch = to_scan[i : i + batch_size]
            batch_results = await asyncio.gather(*[scan_one(client, m) for m in batch])
            for r in batch_results:
                if r:
                    results.append(r)
            done = min(i + batch_size, len(to_scan))
            contested = len(results)
            print(f"      {done}/{len(to_scan)} scanned  |  contested markets found: {contested}", end="\r")
            await asyncio.sleep(0.05)

        print()

    print(f"\n[3/3] Results  ({len(results)} contested markets with real order books)\n")

    # ---- BUNDLE BUY ARB ----
    buy_arbs = sorted(
        [r for r in results if r["edge_buy"] and r["edge_buy"] > MIN_EDGE],
        key=lambda x: -x["edge_buy"]
    )
    if buy_arbs:
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  BUNDLE BUY ARB  —  buy YES + NO together for < $1.00              ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        for r in buy_arbs[:10]:
            cost = r["yes"].best_ask + r["no"].best_ask
            print(f"  Edge: {r['edge_buy']*100:+.2f}%   Cost: ${cost:.4f}   Vol24h: ${r['vol24h']:,.0f}")
            print(f"  YES ask: {r['yes'].best_ask:.4f}   NO ask: {r['no'].best_ask:.4f}")
            print(f"  {r['question']}")
            print()
    else:
        print("  ✗ No bundle BUY arb found in contested markets\n")

    # ---- BUNDLE SELL ARB ----
    sell_arbs = sorted(
        [r for r in results if r["edge_sell"] and r["edge_sell"] > MIN_EDGE],
        key=lambda x: -x["edge_sell"]
    )
    if sell_arbs:
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  BUNDLE SELL ARB  —  sell YES + NO together for > $1.00            ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        for r in sell_arbs[:10]:
            rev = r["yes"].best_bid + r["no"].best_bid
            print(f"  Edge: {r['edge_sell']*100:+.2f}%   Revenue: ${rev:.4f}   Vol24h: ${r['vol24h']:,.0f}")
            print(f"  YES bid: {r['yes'].best_bid:.4f}   NO bid: {r['no'].best_bid:.4f}")
            print(f"  {r['question']}")
            print()
    else:
        print("  ✗ No bundle SELL arb found\n")

    # ---- MARKET MAKING ----
    mm_all = sorted(
        [r for r in results if
         (r["yes"].spread and r["yes"].spread >= MM_SPREAD) or
         (r["no"].spread  and r["no"].spread  >= MM_SPREAD)],
        key=lambda x: -(max(x["yes"].spread or 0, x["no"].spread or 0))
    )
    mm_strong   = [r for r in mm_all if max(r["yes"].spread or 0, r["no"].spread or 0) >= 0.04]
    mm_marginal = [r for r in mm_all if max(r["yes"].spread or 0, r["no"].spread or 0) < 0.04]

    def print_mm_row(r):
        ys = r["yes"].spread or 0
        ns = r["no"].spread  or 0
        url = f"https://polymarket.com/event/{r['slug']}" if r["slug"] else "(no slug)"
        print(f"  YES spread: {ys*100:.1f}¢  (bid {r['yes'].best_bid or 0:.3f} / ask {r['yes'].best_ask or 0:.3f}  depth top-3: {r['yes_bid_depth']:.0f} / {r['yes_ask_depth']:.0f})")
        print(f"  NO  spread: {ns*100:.1f}¢  (bid {r['no'].best_bid  or 0:.3f} / ask {r['no'].best_ask  or 0:.3f})")
        print(f"  Vol24h: ${r['vol24h']:,.0f}   mid YES: {r['yes'].mid:.3f}")
        print(f"  {r['question']}")
        print(f"  {url}")
        print()

    if mm_strong:
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  MM LEADS — STRONG  (spread ≥ 4¢)                                  ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        for r in mm_strong[:10]:
            print_mm_row(r)
    else:
        print("  ✗ No strong MM leads (≥ 4¢) in contested markets\n")

    if mm_marginal:
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  MM LEADS — MARGINAL  (spread 2–4¢)                                ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        for r in mm_marginal[:10]:
            print_mm_row(r)
    else:
        print("  ✗ No marginal MM leads (2–4¢) in contested markets\n")

    # ---- SUMMARY STATS ----
    print("=" * 72)
    print(f"Scanned: {len(to_scan)} markets   Contested (price 8%–92%): {len(results)}")
    print(f"Bundle buy arb: {len(buy_arbs)}   Bundle sell arb: {len(sell_arbs)}   MM leads: {len(mm_all)} ({len(mm_strong)} strong, {len(mm_marginal)} marginal)")

    # Show YES/NO price distribution for contested markets
    if results:
        yes_mids = [r["yes"].mid for r in results if r["yes"].mid]
        spreads  = [r["yes"].spread for r in results if r["yes"].spread]
        if yes_mids:
            print(f"\nYES mid-price range: {min(yes_mids):.2f}–{max(yes_mids):.2f}  "
                  f"avg: {sum(yes_mids)/len(yes_mids):.2f}")
        if spreads:
            avg_sp = sum(spreads)/len(spreads)
            max_sp = max(spreads)
            print(f"YES spread range:    {min(spreads)*100:.1f}¢–{max_sp*100:.1f}¢  avg: {avg_sp*100:.1f}¢")

    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())

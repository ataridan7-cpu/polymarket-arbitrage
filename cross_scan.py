#!/usr/bin/env python3
"""
Cross-Platform Scanner: Polymarket vs Kalshi + PredictIt
=========================================================
Finds price gaps for the same prediction across platforms.
"""

import asyncio
import httpx
import json
import sys
from difflib import SequenceMatcher

from utils.kelly import kelly_contracts
from utils.predictit import get_predictit_markets, predictit_fee
from utils.manifold import get_manifold_markets
from utils.smarkets import get_smarkets_markets, smarkets_fee
from utils.betfair import get_betfair_markets, betfair_fee

GAMMA_URL  = "https://gamma-api.polymarket.com"
CLOB_URL   = "https://clob.polymarket.com"
KALSHI_URL = "https://api.elections.kalshi.com/trade-api/v2"

POLY_FEE    = 0.015   # 1.5% taker fee
KALSHI_FEE  = 0.01    # legacy flat-fee constant; real fee modeled in kalshi_fee()
PI_FEE_RATE = 0.10    # PredictIt 10% of profits fee
MIN_EDGE    = 0.02    # 2% minimum net edge to signal
SIM_THRESH  = 0.72    # minimum text similarity to consider a match
POLY_N      = 500     # number of Polymarket markets to fetch
DEMO_BANKROLL = 1000.0  # paper-trading bankroll for Kelly sizing

# Only scan these two niches — the only categories with meaningful cross-platform
# overlap after testing. Everything else (crypto brackets, elon, weather, CPI)
# is structurally incompatible or covered on only one platform.
ACTIVE_CATEGORIES = {"politics", "fed"}


# ── helpers ───────────────────────────────────────────────────────────────────

async def get_poly_markets(client, n=POLY_N):
    markets, offset = [], 0
    while len(markets) < n:
        r = await client.get(f"{GAMMA_URL}/markets", params={
            "closed": "false", "active": "true",
            "order": "volume24hr", "ascending": "false",
            "limit": 100, "offset": offset,
        }, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch: break
        markets.extend(batch)
        if len(batch) < 100: break
        offset += 100
        await asyncio.sleep(0.08)
    return markets[:n]


KALSHI_SERIES = [
    "KXBTC","KXETH","KXFED","KXCPI","KXPCE","KXGDP",
    "KXNQ","KXSP500","KXGOLD","KXOIL",
    "KXELONMARS","KXNEWPOPE","KXTRUMP",
    "KXHIGHNY","KXHIGHLA","KXRAIN",
]


# Max series we will actually fetch markets for. The matcher only keeps markets
# in known categories (see category()), so fetching the full active set —
# ~10,700 series, the vast majority unmatchable songs/films/foreign elections —
# is pure waste and takes >70 min serially. We cap to a relevant shortlist.
MAX_KALSHI_SERIES = 40


async def discover_kalshi_series(client) -> list[str]:
    """
    Build a shortlist of Kalshi series worth scanning.

    Starts from the curated KALSHI_SERIES (guaranteed coverage), then unions in
    any *active* series whose ticker maps to a known category() bucket
    (crypto / macro / politics / elon). Anything else can never match a
    Polymarket market, so it is dropped. The API ignores `limit` on /series
    (it returns the full ~10.7k set), so we filter client-side and cap the
    result to keep the markets fetch fast.
    """
    seen = set(KALSHI_SERIES)
    discovered: list[str] = []
    try:
        r = await client.get(f"{KALSHI_URL}/series",
                             params={"status": "active"},
                             timeout=15)
        if r.status_code == 200:
            for s in r.json().get("series", []):
                ticker = s.get("ticker")
                if not ticker or ticker in seen:
                    continue
                # Match against the same buckets the matcher uses downstream.
                text = f"{ticker} {s.get('title') or ''}"
                if category(text) != "other":
                    discovered.append(ticker)
                    seen.add(ticker)
    except Exception:
        pass
    # Sort discovered tickers deterministically so the cap is reproducible
    # run-to-run (the API returns them in unstable order).
    series = list(KALSHI_SERIES) + sorted(discovered)
    return series[:MAX_KALSHI_SERIES]


async def _fetch_kalshi_series(client, sem, series):
    """Fetch open binary markets for one series, with 429 backoff."""
    async with sem:
        for attempt in range(3):
            try:
                r = await client.get(f"{KALSHI_URL}/markets",
                                     params={"status": "open", "limit": 100,
                                             "series_ticker": series},
                                     timeout=15)
                if r.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                batch = r.json().get("markets", [])
                return [m for m in batch if not m.get("mve_collection_ticker")]
            except Exception:
                await asyncio.sleep(2 ** attempt)
        return []


async def get_kalshi_markets(client):
    """
    Fetch simple binary Kalshi markets for the relevant series shortlist.

    Series are fetched concurrently (bounded by a semaphore) rather than in a
    serial loop with fixed sleeps — this takes the step from ~70 min to a few
    seconds. Multi-leg sports markets are excluded.
    """
    series_list = await discover_kalshi_series(client)
    sem = asyncio.Semaphore(6)
    results = await asyncio.gather(
        *(_fetch_kalshi_series(client, sem, s) for s in series_list)
    )
    markets = [m for batch in results for m in batch]
    return markets


async def get_poly_ob(client, tok_raw):
    try:
        toks = json.loads(tok_raw)
        yes_id, no_id = str(toks[0]), str(toks[1])
    except Exception:
        return None, None
    try:
        yr, nr = await asyncio.gather(
            client.get(f"{CLOB_URL}/book", params={"token_id": yes_id}, timeout=8),
            client.get(f"{CLOB_URL}/book", params={"token_id": no_id}, timeout=8),
        )
        yb, nb = yr.json(), nr.json()
        def best_bid(b): return max((float(x["price"]) for x in b.get("bids",[])), default=None)
        def best_ask(b): return min((float(x["price"]) for x in b.get("asks",[])), default=None)
        return {
            "yes_bid": best_bid(yb), "yes_ask": best_ask(yb),
            "no_bid":  best_bid(nb), "no_ask":  best_ask(nb),
        }, None
    except Exception as e:
        return None, str(e)


def poly_mid(ob):
    ba, bb = ob.get("yes_ask"), ob.get("yes_bid")
    if ba and bb: return (ba + bb) / 2
    return ba or bb


def kalshi_fee(price):
    """
    Kalshi per-contract trading fee as a fraction of $1.

    Kalshi charges ceil(0.07 * contracts * price * (1 - price)) cents per order;
    the per-contract fee is therefore 0.07 * price * (1 - price). It peaks near
    50¢ (~1.75¢) and shrinks toward the tails — unlike the old flat 1%, which
    overcharged the tails and undercharged the middle. We use the unrounded
    fraction so the edge math stays size-agnostic.
    """
    p = max(0.0, min(1.0, float(price)))
    return 0.07 * p * (1.0 - p)


# Kalshi market titles that are price-bracket or time-specific snapshots —
# structurally incompatible with Polymarket's threshold questions.
_KALSHI_INCOMPATIBLE = [
    "price range",      # narrow bracket e.g. "Bitcoin price range on Jun 6"
    "price at ",        # point-in-time snapshot e.g. "Ethereum price at Jun 6 at 4am"
    "high temp",        # weather bracket
    "temperature",
]

def kalshi_yes_mid(m):
    if m.get("mve_collection_ticker"):   # skip multi-leg sports markets
        return None
    title = (m.get("title") or "").lower()
    if any(pat in title for pat in _KALSHI_INCOMPATIBLE):
        return None                       # skip structurally incompatible markets
    ya = float(m.get("yes_ask_dollars") or 0)
    yb = float(m.get("yes_bid_dollars") or 0)
    if ya and yb:
        return (ya + yb) / 2
    return ya or yb or None


_STOPWORDS = {"will","the","a","an","be","to","in","on","by","at","what","who","which","when",
              "is","are","was","were","market","bet","odds","win","winner","prediction",
              "of","for","and","or","than","that","this","it","as","with","from","after",
              "before","above","below","over","under","reach","reaches","hit","hits","get",
              "gets","any"}

# Generic template words that appear in many same-category titles and carry no
# discriminating power. Stripped before token-overlap scoring so the match is
# driven by the *distinctive* tokens (names, thresholds), not the boilerplate.
_BOILERPLATE = {"election","presidential","president","nomination","nominee","primary",
                "republican","democratic","democrat","party","candidate","general","round",
                "advance","race","seat","governor","mayoral","mayor","senate","senator",
                "house","congress","congressional","vote","votes","second","first",
                "interest","rate","rates","federal","funds","fed","meeting","fomc",
                "year","years","month","price","value","level"}


def _tokens(text):
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [w for w in text.split() if w not in _STOPWORDS]


def normalize(text):
    return ' '.join(_tokens(text))


def content_tokens(text):
    """Distinctive tokens: drop stopwords AND generic template boilerplate."""
    return {w for w in _tokens(text) if w not in _BOILERPLATE}


def jaccard(t1, t2):
    a, b = content_tokens(t1), content_tokens(t2)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similarity(t1, t2):
    """Blend character-sequence ratio with content-token Jaccard.

    SequenceMatcher alone scores on shared boilerplate (it rated
    'Eric Trump … nomination' vs 'Trump family member … nominee' at 0.95).
    Weighting toward token-set overlap rejects that while still rewarding
    genuinely equivalent questions that share their distinctive words.
    """
    n1, n2 = normalize(t1), normalize(t2)
    seq = SequenceMatcher(None, n1, n2).ratio()
    jac = jaccard(t1, t2)
    return 0.35 * seq + 0.65 * jac


def discriminator_penalty(t1, t2):
    """Penalise pairs whose distinctive tokens point at different subjects.

    Fed markets are exempt: Kalshi frames questions as absolute rate levels
    ("above 5.25%") while Polymarket uses rate-change direction ("cut 25bps").
    The vocabulary divergence is structural, not a sign of different events.
    """
    if category(t1) == "fed" and category(t2) == "fed":
        return 0.0
    a, b = content_tokens(t1), content_tokens(t2)
    only1 = {w for w in a - b if w.isalpha() and len(w) >= 4}
    only2 = {w for w in b - a if w.isalpha() and len(w) >= 4}
    return 0.30 if only1 and only2 else 0.0


def category(text: str) -> str:
    """Assign a broad category so we only compare like-for-like markets."""
    t = text.lower()
    if any(k in t for k in ["cpi","inflation","consumer price"]):
        return "cpi"
    if any(k in t for k in ["federal funds","fed rate","interest rate","fomc",
                              "basis point","bps","rate cut","rate hike",
                              "monetary policy","rate decision"]):
        # Exclude non-Fed central banks (ECB, BoE, BoJ, etc.)
        if any(k in t for k in ["ecb","european central","bank of england","boe",
                                  "bank of japan","boj","bank of canada","rba",
                                  "reserve bank"]):
            return "other"
        return "fed"
    if any(k in t for k in ["gdp","gross domestic"]):
        return "gdp"
    if any(k in t for k in ["bitcoin","btc"]):
        return "btc"
    if any(k in t for k in ["ethereum","eth"]):
        return "eth"
    if any(k in t for k in ["trump","harris","biden","president","election",
                              "senate","congress","governor","mayor","mayoral",
                              "referendum","primary","gubernatorial","ballot",
                              "representative","assembly","midterm","runoff"]):
        return "politics"
    if any(k in t for k in ["elon","musk","spacex","tesla"]):
        if any(k in t for k in ["tweet","post","x.com","twitter"]):
            return "elon_tweets"
        return "elon_events"
    return "other"


import re as _re

# FOMC meeting month abbreviations for normalisation
_FOMC_MONTHS = {
    "january":"jan","february":"feb","march":"mar","april":"apr",
    "may":"may","june":"jun","july":"jul","august":"aug",
    "september":"sep","october":"oct","november":"nov","december":"dec",
    "jan":"jan","feb":"feb","mar":"mar","apr":"apr",
    "jun":"jun","jul":"jul","aug":"aug","sep":"sep",
    "oct":"oct","nov":"nov","dec":"dec",
}

def extract_fomc_key(text: str):
    """Return 'mon<year>' (e.g. 'jun2026') if an FOMC meeting date is found.

    Handles both "June 2026" and "Jun 17, 2026" (Kalshi's day-included format).
    """
    t = text.lower()
    for name, abbr in _FOMC_MONTHS.items():
        # Optional day number between month and year: "Jun 17, 2026" or "June 2026"
        m = _re.search(rf'\b{name}\w*[\s,\-]+(?:\d{{1,2}}[,\s]+)?(\d{{4}})\b', t)
        if m:
            return f"{abbr}{m.group(1)}"
    return None


def fed_month_boost(t1: str, t2: str) -> float:
    """
    +0.35 when both Fed questions reference the same FOMC meeting month+year.

    This bridges the vocabulary gap between Kalshi's absolute-level framing
    ("above 5.25% following the June 2026 meeting") and Polymarket's direction
    framing ("cut 25bps after the June 2026 meeting"). Same meeting = same event.
    """
    k1 = extract_fomc_key(t1)
    k2 = extract_fomc_key(t2)
    if k1 and k2 and k1 == k2:
        return 0.35
    return 0.0


# US state names used for politics entity matching
_US_STATES = {
    "alabama","alaska","arizona","arkansas","california","colorado",
    "connecticut","delaware","florida","georgia","hawaii","idaho",
    "illinois","indiana","iowa","kansas","kentucky","louisiana","maine",
    "maryland","massachusetts","michigan","minnesota","mississippi",
    "missouri","montana","nebraska","nevada","hampshire","jersey",
    "mexico","york","carolina","dakota","ohio","oklahoma","oregon",
    "pennsylvania","rhode","tennessee","texas","utah","vermont",
    "virginia","washington","wisconsin","wyoming",
}

# Expanded US political figure list
_POLITICIANS = [
    "trump","harris","biden","obama","clinton","sanders","warren",
    "desantis","newsom","abbott","pence","rubio","cruz","paul",
    "mcconnell","pelosi","schumer","ocasio","cortez","gaetz","jordan",
    "boebert","greene","manchin","sinema","warnock","ossoff",
    "fetterman","oz","hochul","whitmer","pritzker","shapiro",
    "youngkin","kemp","stitt","kelly","cortez masto","hassan",
]


def entity_boost(t1, t2):
    # Must be same category or heavy penalty
    if category(t1) != category(t2):
        return -0.5

    cat = category(t1)

    if cat == "fed":
        # For Fed: shared year number (same meeting cycle) gives a small boost;
        # the big lift comes from fed_month_boost applied separately in scoring.
        nums1 = set(_re.findall(r'\d{4}', t1))
        nums2 = set(_re.findall(r'\d{4}', t2))
        return 0.10 if nums1 & nums2 else 0.0

    if cat == "politics":
        l1, l2 = t1.lower(), t2.lower()
        # Shared politician name → strong signal same subject
        pols_shared = {p for p in _POLITICIANS if p in l1 and p in l2}
        if pols_shared:
            return 0.25
        # Shared US state name → same race geography
        states_shared = {s for s in _US_STATES if s in l1 and s in l2}
        if states_shared:
            return 0.20
        # Shared election year
        years1 = set(_re.findall(r'20\d{2}', t1))
        years2 = set(_re.findall(r'20\d{2}', t2))
        if years1 & years2:
            return 0.15
        return 0.0

    # Crypto (kept for completeness even if not in ACTIVE_CATEGORIES)
    crypto = ["bitcoin","btc","ethereum","eth","solana","sol","xrp","doge"]
    c1 = {c for c in crypto if c in t1.lower()}
    c2 = {c for c in crypto if c in t2.lower()}
    if c1 & c2:
        return 0.25

    nums1 = set(_re.findall(r'\d+(?:\.\d+)?%?', t1))
    nums2 = set(_re.findall(r'\d+(?:\.\d+)?%?', t2))
    return 0.15 if nums1 & nums2 else 0.0


# ── main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("  Cross-Platform Scanner: Polymarket / Kalshi / PredictIt / Manifold")
    print("=" * 72)
    print()
    print("  PLATFORM FEE COMPARISON")
    print("  ──────────────────────────────────────────────────────────────────────")
    print("  Platform    Fee model                     At 10¢   At 50¢   At 90¢  Access")
    print("  ──────────  ────────────────────────────  ───────  ───────  ───────  ──────")
    print("  Polymarket  1.5% flat taker on entry      1.50%    1.50%    1.50%   Global (crypto)")
    print("  Kalshi      7%×p×(1−p) per contract       0.63%    1.75%    0.63%   US only   ← cheapest US at tails")
    print("  Betfair     5% of net winnings            4.50%    2.50%    0.50%   IL ✓ UK ✓ ← cheapest overall >75¢")
    print("  Smarkets    2% of net winnings            1.80%    1.00%    0.20%   UK only   ← cheapest overall always")
    print("  PredictIt   10% of profit on winning leg  90.0%    5.00%    1.11%   US only")
    print("  Manifold    0% (play money — mana)        —        —        —       Global (signal)")
    print()
    print("  For Israeli traders:")
    print("    Best fee for balanced markets (31-69¢): Betfair (2.5% at 50¢) < Polymarket (1.5%)")
    print("    Wait — Polymarket's 1.5% beats Betfair's 2.5% at 50¢. Betfair wins only at p>75¢.")
    print("    Polymarket (crypto, global) is your lowest-fee option for most markets.")
    print("    Betfair (register at betfair.com, accepts IL) wins at p > 75¢ or p < 25¢.")
    print("  ──────────────────────────────────────────────────────────────────────")
    print()

    async with httpx.AsyncClient() as client:
        print(f"\n[1/4] Fetching Polymarket top {POLY_N} markets…")
        poly_raw = await get_poly_markets(client, POLY_N)
        # keep only contested (YES mid 10%–90%) with some volume
        poly_liquid = [
            m for m in poly_raw
            if float(m.get("volume24hr") or 0) >= 500
        ]
        print(f"      {len(poly_liquid)} markets with ≥ $500 daily volume")

        print(f"\n[2/4] Fetching all Kalshi open markets…")
        kalshi_raw = await get_kalshi_markets(client)
        # only binary markets with a real YES price
        kalshi = [m for m in kalshi_raw if kalshi_yes_mid(m) is not None]
        print(f"      {len(kalshi)} Kalshi markets with prices")

        # Pre-compute Kalshi mids for price-proximity filter
        kalshi_mids = {id(km): kalshi_yes_mid(km) for km in kalshi}

        # Also grab Polymarket mid-prices cheaply from outcomePrices in Gamma data
        def poly_gamma_mid(pm):
            try:
                import json as _json
                prices = _json.loads(pm.get("outcomePrices") or "[]")
                if prices:
                    return float(prices[0])
            except Exception:
                pass
            return None

        print(f"\n[3/4] Matching markets by title similarity…")
        pairs = []
        for pm in poly_liquid:
            pq = pm.get("question") or ""
            if not pq:
                continue
            pm_gamma_mid = poly_gamma_mid(pm)

            best_score, best_km = 0.0, None
            pq_cat = category(pq)
            if pq_cat not in ACTIVE_CATEGORIES:
                continue
            for km in kalshi:
                kt = km.get("title") or ""
                if not kt:
                    continue

                # Skip pairs in different or uncategorised buckets before any scoring
                if pq_cat == "other" or category(kt) != pq_cat:
                    continue

                # Price-proximity guard: skip if prices differ by more than 30¢.
                # Disabled for fed pairs: Kalshi asks about absolute rate levels while
                # Polymarket asks about rate-change direction, so prices legitimately
                # differ by ~20¢ on the same event (complementary structure).
                km_mid = kalshi_mids.get(id(km))
                if pq_cat != "fed" and pm_gamma_mid is not None and km_mid is not None:
                    if abs(pm_gamma_mid - km_mid) > 0.30:
                        continue

                if pq_cat == "fed":
                    # Fed pairs: FOMC meeting date is the primary match key.
                    # Vocabulary is incompatible (absolute level vs direction),
                    # so similarity scoring alone will never reach threshold.
                    fomc_pm = extract_fomc_key(pq)
                    fomc_km = extract_fomc_key(kt)
                    if fomc_pm and fomc_km:
                        if fomc_pm != fomc_km:
                            continue  # Different meetings — skip
                        # Same meeting: guaranteed same event, force above threshold
                        s = min(0.76 + entity_boost(pq, kt), 1.0)
                    else:
                        # No meeting date extractable — fall back to similarity
                        s = min(similarity(pq, kt) + entity_boost(pq, kt) + fed_month_boost(pq, kt), 1.0)
                else:
                    s = similarity(pq, kt) + entity_boost(pq, kt) - discriminator_penalty(pq, kt)
                    s = min(s, 1.0)
                if s > best_score:
                    best_score = s
                    best_km = km
            if best_score >= SIM_THRESH and best_km:
                pairs.append((best_score, pm, best_km))

        pairs.sort(key=lambda x: -x[0])
        print(f"      {len(pairs)} potential matches above {SIM_THRESH:.0%} similarity")

        # Take top 60 highest-similarity pairs to check order books
        top_pairs = pairs[:60]
        print(f"\n[4/4] Fetching Polymarket order books for top {len(top_pairs)} pairs…")

        arbs = []
        price_diffs = []

        for i, (score, pm, km) in enumerate(top_pairs):
            tok_raw = pm.get("clobTokenIds", "")
            ob, err = await get_poly_ob(client, tok_raw)
            if not ob:
                continue

            pm_mid = poly_mid(ob)
            km_mid = kalshi_yes_mid(km)
            if pm_mid is None or km_mid is None:
                continue

            diff = abs(pm_mid - km_mid)
            price_diffs.append((score, pm, km, ob, pm_mid, km_mid, diff))

            # Cross-platform bundle arb: buy YES on one side + NO on the other.
            # One leg always pays $1 at resolution regardless of outcome.
            # NOTE: Fed pairs are excluded from bundle arb because Kalshi asks about
            # absolute rate levels ("above 5.25%") while Polymarket asks about rate
            # change direction ("cut 25bps") — they are NOT complementary binaries and
            # a "no change" scenario could make both legs lose.
            pq_cat_arb = category(pm.get("question") or "")
            if pq_cat_arb == "fed":
                continue

            km_yes_ask = float(km.get("yes_ask_dollars") or 0)
            km_no_ask  = float(km.get("no_ask_dollars")  or 0)

            poly_slug    = pm.get("slug") or pm.get("market_slug") or ""
            kalshi_tick  = km.get("ticker") or ""
            kalshi_event = km.get("event_ticker") or kalshi_tick
            close_time   = km.get("close_time") or ""
            vol24h       = float(pm.get("volume24hr") or 0)

            def _arb_base():
                return {
                    "poly_q":       (pm.get("question") or "")[:80],
                    "kalshi_t":     (km.get("title") or "")[:80],
                    "similarity":   score,
                    "vol24h":       vol24h,
                    "sell_price":   1.0,
                    "poly_url":     f"https://polymarket.com/event/{poly_slug}" if poly_slug else "",
                    "kalshi_url":   f"https://kalshi.com/markets/{kalshi_event}" if kalshi_event else "",
                    "close_time":   close_time[:10] if close_time else "",  # YYYY-MM-DD
                }

            # Case C: Buy Poly YES + Buy Kalshi NO → collect $1 either way
            if ob["yes_ask"] and km_no_ask:
                cost       = ob["yes_ask"] + km_no_ask
                fee_poly   = round(ob["yes_ask"] * POLY_FEE, 5)
                fee_kalshi = round(kalshi_fee(km_no_ask), 5)
                fees       = fee_poly + fee_kalshi
                gross      = round(1.0 - cost, 4)
                net        = round(gross - fees, 4)
                if net >= MIN_EDGE:
                    a = _arb_base()
                    a.update({
                        "type":       "Bundle: Buy Poly YES + Kalshi NO",
                        "net_edge":   net,
                        "gross_edge": gross,
                        "fees":       fees,
                        "fee_poly":   fee_poly,
                        "fee_kalshi": fee_kalshi,
                        "buy_price":  cost,
                        "leg1":       f"Buy YES on Polymarket  @ ${ob['yes_ask']:.3f}  (fee: ${fee_poly:.4f})",
                        "leg2":       f"Buy NO  on Kalshi      @ ${km_no_ask:.3f}  (fee: ${fee_kalshi:.4f})",
                    })
                    arbs.append(a)

            # Case D: Buy Kalshi YES + Buy Poly NO → collect $1 either way
            if km_yes_ask and ob["no_ask"]:
                cost       = km_yes_ask + ob["no_ask"]
                fee_kalshi = round(kalshi_fee(km_yes_ask), 5)
                fee_poly   = round(ob["no_ask"] * POLY_FEE, 5)
                fees       = fee_kalshi + fee_poly
                gross      = round(1.0 - cost, 4)
                net        = round(gross - fees, 4)
                if net >= MIN_EDGE:
                    a = _arb_base()
                    a.update({
                        "type":       "Bundle: Buy Kalshi YES + Poly NO",
                        "net_edge":   net,
                        "gross_edge": gross,
                        "fees":       fees,
                        "fee_poly":   fee_poly,
                        "fee_kalshi": fee_kalshi,
                        "buy_price":  cost,
                        "leg1":       f"Buy YES on Kalshi      @ ${km_yes_ask:.3f}  (fee: ${fee_kalshi:.4f})",
                        "leg2":       f"Buy NO  on Polymarket  @ ${ob['no_ask']:.3f}  (fee: ${fee_poly:.4f})",
                    })
                    arbs.append(a)

            if (i + 1) % 10 == 0:
                print(f"      {i+1}/{len(top_pairs)} checked  |  arb signals so far: {len(arbs)}", end="\r")
            await asyncio.sleep(0.05)

        print()

    # ── results ────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")

    # Only show arb signals where the price gap is ALSO confirmed by order books
    # AND the category genuinely matches
    genuine_arbs = [
        a for a in arbs
        if a["similarity"] >= SIM_THRESH
        and category(a["poly_q"]) == category(a["kalshi_t"])
        and category(a["poly_q"]) != "other"
    ]

    if genuine_arbs:
        genuine_arbs.sort(key=lambda x: -x["net_edge"])
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  CROSS-PLATFORM BUNDLE ARB  (YES one platform + NO the other)       ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print()
        print("  HOW TO EXECUTE A BUNDLE ARB TRADE")
        print("  ──────────────────────────────────")
        print("  1. Open both platform URLs below simultaneously.")
        print("  2. Place BOTH legs as close to simultaneously as possible;")
        print("     stale prices vanish fast — use limit orders at the shown ask.")
        print("  3. Each $1 contract pays $1 at resolution, regardless of outcome.")
        print("  4. Your profit = $1.00 − total cost − fees (shown as Net edge).")
        print("  5. Position sizing: start small (≤ $100/signal) until you have")
        print("     confirmed that both markets resolve on the same event/date.")
        print("     Scale up to 1–2% of daily volume only after first verification.")
        print("  6. Max position per leg is limited by the top-of-book liquidity;")
        print("     do not move the market — use limit orders, not market orders.")
        print("  7. RISK: If the two questions resolve differently (e.g., different")
        print("     thresholds or dates), you hold an unhedged directional position.")
        print("     Always read both question texts before trading.")
        print()
        for a in genuine_arbs[:10]:
            # Kelly-optimal sizing (half-Kelly, capped at 20% of demo bankroll)
            contracts, dollar_size = kelly_contracts(
                0.5, a["buy_price"], DEMO_BANKROLL
            )
            exp_profit = contracts * a["net_edge"]
            fees      = a.get("fees", 0)
            fee_poly  = a.get("fee_poly", 0)
            fee_kalshi= a.get("fee_kalshi", 0)
            print(f"  ┌─ {a['type']}")
            print(f"  │  Cost: ${a['buy_price']:.3f}  →  Gross: ${a['gross_edge']:.3f}  −  Fees: ${fees:.4f}  =  Net: ${a['net_edge']:.3f} ({a['net_edge']*100:+.2f}%)")
            print(f"  │  Fees: Poly ${fee_poly:.4f} ({POLY_FEE*100:.1f}%)  +  Kalshi ${fee_kalshi:.4f} (0.07·p·(1−p))")
            print(f"  │  Match score: {a['similarity']:.2f}   Vol24h: ${a['vol24h']:,.0f}")
            if a.get("close_time"):
                print(f"  │  Resolves: {a['close_time']}")
            print(f"  │")
            print(f"  │  Poly:  {a['poly_q']}")
            print(f"  │  Kalshi:{a['kalshi_t']}")
            print(f"  │")
            print(f"  │  LEG 1 — {a['leg1']}")
            print(f"  │  LEG 2 — {a['leg2']}")
            if a.get("poly_url"):
                print(f"  │  Polymarket: {a['poly_url']}")
            if a.get("kalshi_url"):
                print(f"  │  Kalshi:     {a['kalshi_url']}")
            print(f"  │")
            print(f"  │  Suggested size: {contracts} contracts (~${contracts * a['buy_price']:.0f} total cost)")
            print(f"  │  Expected profit at suggested size: ~${exp_profit:.2f}")
            print(f"  └─ ⚠  Verify both questions resolve on the same event before trading")
            print()
    else:
        print("  ✗ No cross-platform arb found.\n")
        print("  Note: Polymarket and Kalshi currently share very few equivalent")
        print("  markets. Polymarket's volume is in crypto/sports price brackets;")
        print("  Kalshi's simple binary markets cover CPI, GDP, Fed rate, and weather.")
        print("  Genuine cross-platform arb requires both platforms asking the exact")
        print("  same YES/NO question — which is rare today.\n")

    # Each entry: (label, poly_url, kalshi_url)
    link_pairs = []
    for a in genuine_arbs[:10]:
        link_pairs.append((a["type"], a.get("poly_url", ""), a.get("kalshi_url", "")))

    # Show closest candidate pairs for manual review
    if price_diffs:
        price_diffs.sort(key=lambda x: -x[0])   # sort by similarity score
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  CLOSEST MARKET MATCHES  (for manual review)                        ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        shown = 0
        for score, pm, km, ob, pm_mid, km_mid, diff in price_diffs:
            if category(pm.get("question","")) == category(km.get("title","")):
                direction = "Poly higher" if pm_mid > km_mid else "Kalshi higher"
                slug        = pm.get("slug") or pm.get("market_slug") or ""
                k_event     = km.get("event_ticker") or km.get("ticker") or ""
                poly_url    = f"https://polymarket.com/event/{slug}" if slug else ""
                kalshi_url  = f"https://kalshi.com/markets/{k_event}" if k_event else ""
                print(f"  Similarity: {score:.2f}  Gap: {diff*100:.1f}¢ ({direction})")
                print(f"  Poly   [{pm_mid:.3f}]: {(pm.get('question') or '')[:70]}")
                print(f"  Kalshi [{km_mid:.3f}]: {(km.get('title') or '')[:70]}")
                if poly_url:
                    print(f"  → Polymarket: {poly_url}")
                if kalshi_url:
                    print(f"  → Kalshi:     {kalshi_url}")
                print()
                label = f"Match {score:.2f} | {(pm.get('question') or '')[:45]}"
                link_pairs.append((label, poly_url, kalshi_url))
                shown += 1
                if shown >= 10:
                    break
        if shown == 0:
            print("  No same-category pairs found in top matches.\n")

    # ---- QUICK LINKS ----
    if link_pairs:
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  QUICK LINKS  —  open these markets                                 ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        for label, poly_url, kalshi_url in link_pairs:
            print(f"  {label}")
            if poly_url:
                print(f"  → Polymarket: {poly_url}")
            if kalshi_url:
                print(f"  → Kalshi:     {kalshi_url}")
            print()

    # ── PredictIt cross-scan ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Cross-Platform Scanner: Polymarket  ↔  PredictIt  —  live prices")
    print("=" * 72)
    print("\n[PI] Fetching PredictIt markets…")
    pi_markets = await asyncio.get_event_loop().run_in_executor(
        None, get_predictit_markets
    )
    print(f"     {len(pi_markets)} PredictIt contracts with prices")

    print("[PI] Matching Polymarket ↔ PredictIt…")
    pi_pairs = []
    for pm in poly_liquid:
        pq = pm.get("question") or ""
        if not pq:
            continue
        pq_cat = category(pq)
        if pq_cat not in ACTIVE_CATEGORIES:
            continue
        pm_mid = poly_gamma_mid(pm)
        best_score, best_pi = 0.0, None
        for pi in pi_markets:
            pt = pi.get("question") or ""
            if category(pt) != pq_cat:
                continue
            # Same price-guard exemption for fed: complementary question structures
            if pq_cat != "fed" and pm_mid is not None and pi.get("mid") is not None:
                if abs(pm_mid - pi["mid"]) > 0.30:
                    continue
            if pq_cat == "fed":
                fomc_pm = extract_fomc_key(pq)
                fomc_pt = extract_fomc_key(pt)
                if fomc_pm and fomc_pt:
                    if fomc_pm != fomc_pt:
                        continue
                    s = min(0.76 + entity_boost(pq, pt), 1.0)
                else:
                    s = min(similarity(pq, pt) + entity_boost(pq, pt) + fed_month_boost(pq, pt), 1.0)
            else:
                s = similarity(pq, pt) + entity_boost(pq, pt) - discriminator_penalty(pq, pt)
                s = min(s, 1.0)
            if s > best_score:
                best_score, best_pi = s, pi
        if best_score >= SIM_THRESH and best_pi:
            pi_pairs.append((best_score, pm, best_pi))

    pi_pairs.sort(key=lambda x: -x[0])
    print(f"     {len(pi_pairs)} PredictIt matches above {SIM_THRESH:.0%} similarity")

    pi_arbs = []
    for score, pm, pi in pi_pairs[:30]:
        tok_raw = pm.get("clobTokenIds", "")
        ob, _ = await get_poly_ob(client, tok_raw)
        if not ob:
            continue

        ya   = pi.get("yes_ask")
        na   = pi.get("no_ask")
        slug = pm.get("slug") or pm.get("market_slug") or ""
        poly_url = f"https://polymarket.com/event/{slug}" if slug else ""
        pi_url   = pi.get("url", "")

        # Case E: Buy Poly YES + PredictIt NO
        if ob["yes_ask"] and na:
            cost    = ob["yes_ask"] + na
            fp      = round(ob["yes_ask"] * POLY_FEE, 5)
            fpi     = round(predictit_fee(na), 5)       # 10% of (1−na) on PI NO win
            fees    = fp + fpi
            gross   = round(1.0 - cost, 4)
            net     = round(gross - fees, 4)
            if net >= MIN_EDGE:
                pi_arbs.append({
                    "type":       "Bundle: Buy Poly YES + PredictIt NO",
                    "net_edge":   net, "gross_edge": gross, "fees": fees,
                    "buy_price":  cost, "similarity": score,
                    "vol24h":     float(pm.get("volume24hr") or 0),
                    "poly_q":     pm.get("question") or "",
                    "pi_q":       pi.get("question") or "",
                    "leg1":       f"Buy YES on Polymarket  @ ${ob['yes_ask']:.3f}  (fee: ${fp:.4f})",
                    "leg2":       f"Buy NO  on PredictIt   @ ${na:.3f}  (fee: ${fpi:.4f})",
                    "poly_url":   poly_url, "pi_url": pi_url,
                })

        # Case F: Buy PredictIt YES + Poly NO
        if ya and ob["no_ask"]:
            cost    = ya + ob["no_ask"]
            fp      = round(ob["no_ask"] * POLY_FEE, 5)
            fpi     = round(predictit_fee(ya), 5)       # 10% of (1−ya) on PI YES win
            fees    = fp + fpi
            gross   = round(1.0 - cost, 4)
            net     = round(gross - fees, 4)
            if net >= MIN_EDGE:
                pi_arbs.append({
                    "type":       "Bundle: Buy PredictIt YES + Poly NO",
                    "net_edge":   net, "gross_edge": gross, "fees": fees,
                    "buy_price":  cost, "similarity": score,
                    "vol24h":     float(pm.get("volume24hr") or 0),
                    "poly_q":     pm.get("question") or "",
                    "pi_q":       pi.get("question") or "",
                    "leg1":       f"Buy YES on PredictIt   @ ${ya:.3f}  (fee: ${fpi:.4f})",
                    "leg2":       f"Buy NO  on Polymarket  @ ${ob['no_ask']:.3f}  (fee: ${fp:.4f})",
                    "poly_url":   poly_url, "pi_url": pi_url,
                })

        await asyncio.sleep(0.03)

    pi_arbs.sort(key=lambda x: -x["net_edge"])
    print()
    if pi_arbs:
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  POLY ↔ PREDICTIT BUNDLE ARB                                        ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        for a in pi_arbs[:10]:
            contracts, dollar_size = kelly_contracts(0.5, a["buy_price"], DEMO_BANKROLL)
            exp_profit = contracts * a["net_edge"]
            print(f"  ┌─ {a['type']}")
            print(f"  │  Cost ${a['buy_price']:.3f}  Gross ${a['gross_edge']:.3f}  Fees ${a['fees']:.4f}  Net ${a['net_edge']:.3f} ({a['net_edge']*100:+.2f}%)")
            print(f"  │  Match: {a['similarity']:.2f}   Vol24h: ${a['vol24h']:,.0f}")
            print(f"  │  Poly:       {a['poly_q'][:65]}")
            print(f"  │  PredictIt:  {a['pi_q'][:65]}")
            print(f"  │  LEG 1 — {a['leg1']}")
            print(f"  │  LEG 2 — {a['leg2']}")
            print(f"  │  Polymarket: {a['poly_url']}")
            print(f"  │  PredictIt:  {a['pi_url']}")
            print(f"  │  Kelly size: {contracts} contracts (~${dollar_size:.0f})  Expected profit: ~${exp_profit:.2f}")
            print(f"  └─ ⚠  PredictIt fees are 10% of profits — verify before trading")
            print()
    else:
        print("  ✗ No Poly ↔ PredictIt arb found.\n")

    # ── Betfair cross-scan ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Cross-Platform Scanner: Polymarket  ↔  Betfair  —  live prices")
    print("=" * 72)
    print("  Betfair accepts Israeli accounts. Set BETFAIR_APP_KEY + credentials to enable.")
    print("  Fee: 5% of net winnings (beats Polymarket at p > 75¢ or p < 25¢).")
    print()
    print("[BF] Fetching Betfair US politics markets…")
    bf_markets = await asyncio.get_event_loop().run_in_executor(
        None, get_betfair_markets
    )
    print(f"     {len(bf_markets)} Betfair binary markets with prices")

    bf_pairs = []
    for pm in poly_liquid:
        pq = pm.get("question") or ""
        if not pq:
            continue
        pq_cat = category(pq)
        if pq_cat not in ACTIVE_CATEGORIES:
            continue
        pm_mid = poly_gamma_mid(pm)
        best_score, best_bf = 0.0, None
        for bf in bf_markets:
            bt = bf.get("question") or ""
            if not bt or category(bt) != pq_cat:
                continue
            if pm_mid is not None and bf.get("mid") is not None:
                if abs(pm_mid - bf["mid"]) > 0.35:
                    continue
            s = similarity(pq, bt) + entity_boost(pq, bt) - discriminator_penalty(pq, bt)
            s = min(s, 1.0)
            if s > best_score:
                best_score, best_bf = s, bf
        if best_score >= SIM_THRESH and best_bf:
            bf_pairs.append((best_score, pm, best_bf))

    bf_pairs.sort(key=lambda x: -x[0])
    print(f"     {len(bf_pairs)} Betfair matches above {SIM_THRESH:.0%} similarity")

    bf_arbs = []
    for score, pm, bf in bf_pairs[:20]:
        tok_raw = pm.get("clobTokenIds", "")
        ob, _ = await get_poly_ob(client, tok_raw)
        if not ob:
            continue

        ya   = bf.get("yes_ask")
        na   = bf.get("no_ask")
        slug = pm.get("slug") or pm.get("market_slug") or ""
        poly_url = f"https://polymarket.com/event/{slug}" if slug else ""
        bf_url   = bf.get("url", "")

        # Case I: Buy Poly YES + Betfair NO
        if ob["yes_ask"] and na:
            cost  = ob["yes_ask"] + na
            fp    = round(ob["yes_ask"] * POLY_FEE, 5)
            fbf   = round(betfair_fee(na), 5)
            fees  = fp + fbf
            gross = round(1.0 - cost, 4)
            net   = round(gross - fees, 4)
            if net >= MIN_EDGE:
                bf_arbs.append({
                    "type":      "Bundle: Buy Poly YES + Betfair NO",
                    "net_edge":  net, "gross_edge": gross, "fees": fees,
                    "buy_price": cost, "similarity": score,
                    "vol24h":    float(pm.get("volume24hr") or 0),
                    "poly_q":    pm.get("question") or "",
                    "bf_q":      bf.get("question") or "",
                    "leg1":      f"Buy YES on Polymarket @ ${ob['yes_ask']:.3f}  (fee: ${fp:.4f})",
                    "leg2":      f"Buy NO  on Betfair    @ ${na:.3f}  (fee: ${fbf:.4f})",
                    "poly_url":  poly_url, "bf_url": bf_url,
                })

        # Case J: Buy Betfair YES + Poly NO
        if ya and ob["no_ask"]:
            cost  = ya + ob["no_ask"]
            fp    = round(ob["no_ask"] * POLY_FEE, 5)
            fbf   = round(betfair_fee(ya), 5)
            fees  = fp + fbf
            gross = round(1.0 - cost, 4)
            net   = round(gross - fees, 4)
            if net >= MIN_EDGE:
                bf_arbs.append({
                    "type":      "Bundle: Buy Betfair YES + Poly NO",
                    "net_edge":  net, "gross_edge": gross, "fees": fees,
                    "buy_price": cost, "similarity": score,
                    "vol24h":    float(pm.get("volume24hr") or 0),
                    "poly_q":    pm.get("question") or "",
                    "bf_q":      bf.get("question") or "",
                    "leg1":      f"Buy YES on Betfair    @ ${ya:.3f}  (fee: ${fbf:.4f})",
                    "leg2":      f"Buy NO  on Polymarket @ ${ob['no_ask']:.3f}  (fee: ${fp:.4f})",
                    "poly_url":  poly_url, "bf_url": bf_url,
                })

        await asyncio.sleep(0.03)

    bf_arbs.sort(key=lambda x: -x["net_edge"])
    print()
    if bf_arbs:
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  POLY ↔ BETFAIR BUNDLE ARB                                          ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        for a in bf_arbs[:10]:
            contracts, dollar_size = kelly_contracts(0.5, a["buy_price"], DEMO_BANKROLL)
            exp_profit = contracts * a["net_edge"]
            print(f"  ┌─ {a['type']}")
            print(f"  │  Cost ${a['buy_price']:.3f}  Gross ${a['gross_edge']:.3f}  Fees ${a['fees']:.4f}  Net ${a['net_edge']:.3f} ({a['net_edge']*100:+.2f}%)")
            print(f"  │  Match: {a['similarity']:.2f}   Vol24h: ${a['vol24h']:,.0f}")
            print(f"  │  Poly:    {a['poly_q'][:65]}")
            print(f"  │  Betfair: {a['bf_q'][:65]}")
            print(f"  │  LEG 1 — {a['leg1']}")
            print(f"  │  LEG 2 — {a['leg2']}")
            print(f"  │  Polymarket: {a['poly_url']}")
            print(f"  │  Betfair:    {a['bf_url']}")
            print(f"  │  Kelly size: {contracts} contracts (~${dollar_size:.0f})  Expected: ~${exp_profit:.2f}")
            print(f"  └─ ⚠  Verify questions resolve on the same event before trading")
            print()
    else:
        print("  ✗ No Poly ↔ Betfair arb found.\n")

    # ── Manifold price-signal comparison ─────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Price-Signal Comparison: Polymarket  ↔  Manifold  (play money)")
    print("=" * 72)
    print("  Manifold uses mana (play money). Divergences ≥ 5¢ from Polymarket")
    print("  may flag genuine mispricings worth investigating on real platforms.")
    print()
    print("[MF] Fetching Manifold markets…")
    mf_markets = await asyncio.get_event_loop().run_in_executor(
        None, get_manifold_markets
    )
    print(f"     {len(mf_markets)} Manifold binary markets fetched")

    mf_pairs = []
    for pm in poly_liquid:
        pq = pm.get("question") or ""
        if not pq:
            continue
        pq_cat = category(pq)
        if pq_cat not in ACTIVE_CATEGORIES:
            continue
        pm_mid = poly_gamma_mid(pm)
        best_score, best_mf = 0.0, None
        for mf in mf_markets:
            mt = mf.get("question") or ""
            if not mt or category(mt) != pq_cat:
                continue
            if pq_cat == "fed":
                fomc_pm = extract_fomc_key(pq)
                fomc_mf = extract_fomc_key(mt)
                if fomc_pm and fomc_mf:
                    if fomc_pm != fomc_mf:
                        continue
                    s = min(0.76 + entity_boost(pq, mt), 1.0)
                else:
                    s = min(similarity(pq, mt) + entity_boost(pq, mt) + fed_month_boost(pq, mt), 1.0)
            else:
                # Wider price guard for play-money signal (40¢ instead of 30¢)
                if pm_mid is not None and abs(pm_mid - mf["mid"]) > 0.40:
                    continue
                s = similarity(pq, mt) + entity_boost(pq, mt) - discriminator_penalty(pq, mt)
                s = min(s, 1.0)
            if s > best_score:
                best_score, best_mf = s, mf

        if best_score >= SIM_THRESH and best_mf:
            mf_pairs.append((best_score, pm, best_mf))

    mf_pairs.sort(key=lambda x: -x[0])
    print(f"[MF] {len(mf_pairs)} Manifold matches above {SIM_THRESH:.0%} similarity")

    if mf_pairs:
        print()
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  MANIFOLD ↔ POLYMARKET PRICE DIVERGENCES  (signal — not real arb)  ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        shown = 0
        for score, pm, mf in mf_pairs:
            pm_mid = poly_gamma_mid(pm)
            mf_mid = mf.get("mid")
            if pm_mid is None or mf_mid is None:
                continue
            diff = abs(pm_mid - mf_mid)
            if diff < 0.02:
                continue
            direction = "Poly higher" if pm_mid > mf_mid else "Manifold higher"
            slug      = pm.get("slug") or pm.get("market_slug") or ""
            poly_url  = f"https://polymarket.com/event/{slug}" if slug else ""
            print(f"  Match {score:.2f}  Gap: {diff*100:.1f}¢ ({direction})")
            print(f"  Poly     [{pm_mid:.3f}]: {(pm.get('question') or '')[:65]}")
            print(f"  Manifold [{mf_mid:.3f}]: {(mf.get('question') or '')[:65]}")
            if poly_url:
                print(f"  → Polymarket: {poly_url}")
            if mf.get("url"):
                print(f"  → Manifold:   {mf['url']}")
            print()
            shown += 1
            if shown >= 10:
                break
        if shown == 0:
            print("  No divergences ≥ 2¢ found — prices are aligned.\n")
    else:
        print("  No Manifold matches found.\n")

    # ── Smarkets cross-scan ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Cross-Platform Scanner: Polymarket  ↔  Smarkets  —  live prices")
    print("=" * 72)
    print("  NOTE: Smarkets is a UK exchange. US trading requires VPN/UK account.")
    print("  Fee: 2% of net winnings (cheaper than Poly at p>69¢, PredictIt always).")
    print()
    print("[SM] Fetching Smarkets US politics markets…")
    sm_markets = await asyncio.get_event_loop().run_in_executor(
        None, get_smarkets_markets
    )
    print(f"     {len(sm_markets)} Smarkets binary markets with prices")

    sm_pairs = []
    for pm in poly_liquid:
        pq = pm.get("question") or ""
        if not pq:
            continue
        pq_cat = category(pq)
        if pq_cat not in ACTIVE_CATEGORIES:
            continue
        pm_mid = poly_gamma_mid(pm)
        best_score, best_sm = 0.0, None
        for sm in sm_markets:
            st = sm.get("question") or ""
            if not st or category(st) != pq_cat:
                continue
            if pm_mid is not None and sm.get("mid") is not None:
                if abs(pm_mid - sm["mid"]) > 0.30:
                    continue
            s = similarity(pq, st) + entity_boost(pq, st) - discriminator_penalty(pq, st)
            s = min(s, 1.0)
            if s > best_score:
                best_score, best_sm = s, sm
        if best_score >= SIM_THRESH and best_sm:
            sm_pairs.append((best_score, pm, best_sm))

    sm_pairs.sort(key=lambda x: -x[0])
    print(f"     {len(sm_pairs)} Smarkets matches above {SIM_THRESH:.0%} similarity")

    sm_arbs = []
    for score, pm, sm in sm_pairs[:20]:
        tok_raw = pm.get("clobTokenIds", "")
        ob, _ = await get_poly_ob(client, tok_raw)
        if not ob:
            continue

        ya   = sm.get("yes_ask")
        na   = sm.get("no_ask")
        slug = pm.get("slug") or pm.get("market_slug") or ""
        poly_url = f"https://polymarket.com/event/{slug}" if slug else ""
        sm_url   = sm.get("url", "")

        # Case G: Buy Poly YES + Smarkets NO
        if ob["yes_ask"] and na:
            cost  = ob["yes_ask"] + na
            fp    = round(ob["yes_ask"] * POLY_FEE, 5)
            fsm   = round(smarkets_fee(na), 5)
            fees  = fp + fsm
            gross = round(1.0 - cost, 4)
            net   = round(gross - fees, 4)
            if net >= MIN_EDGE:
                sm_arbs.append({
                    "type":      "Bundle: Buy Poly YES + Smarkets NO",
                    "net_edge":  net, "gross_edge": gross, "fees": fees,
                    "buy_price": cost, "similarity": score,
                    "vol24h":    float(pm.get("volume24hr") or 0),
                    "poly_q":    pm.get("question") or "",
                    "sm_q":      sm.get("question") or "",
                    "leg1":      f"Buy YES on Polymarket @ ${ob['yes_ask']:.3f}  (fee: ${fp:.4f})",
                    "leg2":      f"Buy NO  on Smarkets   @ ${na:.3f}  (fee: ${fsm:.4f})",
                    "poly_url":  poly_url, "sm_url": sm_url,
                })

        # Case H: Buy Smarkets YES + Poly NO
        if ya and ob["no_ask"]:
            cost  = ya + ob["no_ask"]
            fp    = round(ob["no_ask"] * POLY_FEE, 5)
            fsm   = round(smarkets_fee(ya), 5)
            fees  = fp + fsm
            gross = round(1.0 - cost, 4)
            net   = round(gross - fees, 4)
            if net >= MIN_EDGE:
                sm_arbs.append({
                    "type":      "Bundle: Buy Smarkets YES + Poly NO",
                    "net_edge":  net, "gross_edge": gross, "fees": fees,
                    "buy_price": cost, "similarity": score,
                    "vol24h":    float(pm.get("volume24hr") or 0),
                    "poly_q":    pm.get("question") or "",
                    "sm_q":      sm.get("question") or "",
                    "leg1":      f"Buy YES on Smarkets   @ ${ya:.3f}  (fee: ${fsm:.4f})",
                    "leg2":      f"Buy NO  on Polymarket @ ${ob['no_ask']:.3f}  (fee: ${fp:.4f})",
                    "poly_url":  poly_url, "sm_url": sm_url,
                })

        await asyncio.sleep(0.03)

    sm_arbs.sort(key=lambda x: -x["net_edge"])
    print()
    if sm_arbs:
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  POLY ↔ SMARKETS BUNDLE ARB                                         ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        for a in sm_arbs[:10]:
            contracts, dollar_size = kelly_contracts(0.5, a["buy_price"], DEMO_BANKROLL)
            exp_profit = contracts * a["net_edge"]
            print(f"  ┌─ {a['type']}")
            print(f"  │  Cost ${a['buy_price']:.3f}  Gross ${a['gross_edge']:.3f}  Fees ${a['fees']:.4f}  Net ${a['net_edge']:.3f} ({a['net_edge']*100:+.2f}%)")
            print(f"  │  Match: {a['similarity']:.2f}   Vol24h: ${a['vol24h']:,.0f}")
            print(f"  │  Poly:     {a['poly_q'][:65]}")
            print(f"  │  Smarkets: {a['sm_q'][:65]}")
            print(f"  │  LEG 1 — {a['leg1']}")
            print(f"  │  LEG 2 — {a['leg2']}")
            print(f"  │  Polymarket: {a['poly_url']}")
            print(f"  │  Smarkets:   {a['sm_url']}")
            print(f"  │  Kelly size: {contracts} contracts (~${dollar_size:.0f})  Expected profit: ~${exp_profit:.2f}")
            print(f"  └─ ⚠  Smarkets requires UK account — verify access before trading")
            print()
    else:
        print("  ✗ No Poly ↔ Smarkets arb found.\n")

    print("=" * 72)
    print(f"  SUMMARY STATS")
    print(f"  Polymarket: {len(poly_liquid)}  Kalshi: {len(kalshi)}  PredictIt: {len(pi_markets)}  Betfair: {len(bf_markets)}  Smarkets: {len(sm_markets)}  Manifold: {len(mf_markets)}")
    print(f"  Kalshi pairs: {len(price_diffs)}   arbs: {len(genuine_arbs)}")
    print(f"  PredictIt pairs: {len(pi_pairs)}   arbs: {len(pi_arbs)}")
    print(f"  Betfair pairs: {len(bf_pairs)}   arbs: {len(bf_arbs)}")
    print(f"  Smarkets pairs: {len(sm_pairs)}   arbs: {len(sm_arbs)}")
    print(f"  Manifold signal pairs: {len(mf_pairs)}")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())

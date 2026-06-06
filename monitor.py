#!/usr/bin/env python3
"""
Continuous arbitrage monitor with paper trading.

Runs the Polymarket and cross-platform scanners in a loop, applies Kelly
position sizing, and logs demo trades. Only prints when signals change.

Usage:
  python monitor.py                 # scan every 90s, $1 000 demo bankroll
  python monitor.py --interval 60   # faster cycle
  python monitor.py --bankroll 500  # smaller bankroll
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime

import httpx

# scanner helpers
import scanner as poly_scanner
import cross_scan

from utils.kelly import kelly_contracts, kelly_mm
from utils.paper_trader import PaperTrader
from utils.predictit import get_predictit_markets, predictit_fee

DEFAULT_INTERVAL  = 90      # seconds between scans
DEFAULT_BANKROLL  = 1000.0  # demo bankroll
HALF_KELLY        = True
MAX_OPEN_TRADES   = 20      # don't log more than this many paper trades at once


# ── Polymarket scanner ────────────────────────────────────────────────────────

async def _poly_signals(client: httpx.AsyncClient, bankroll: float) -> list[dict]:
    """
    Run the single-venue Polymarket scanner and return signal dicts.
    Returns: list of {"type", "question", "url", "entry_cost", "edge_pct",
                       "num_contracts", "dollar_size", "kelly_frac"}
    """
    raw = await poly_scanner.fetch_markets(client, poly_scanner.FETCH_N)
    raw = [m for m in raw if float(m.get("volume24hr") or 0) >= poly_scanner.MIN_VOLUME]
    to_scan = raw[:poly_scanner.SCAN_CAP]

    results = []
    batch_size = 25
    for i in range(0, len(to_scan), batch_size):
        batch = to_scan[i : i + batch_size]
        batch_res = await asyncio.gather(*[poly_scanner.scan_one(client, m) for m in batch])
        results.extend(r for r in batch_res if r)
        await asyncio.sleep(0.05)

    signals = []

    # Bundle BUY arbs
    for r in results:
        if r.get("edge_buy") and r["edge_buy"] > poly_scanner.MIN_EDGE:
            cost = r["yes"].best_ask + r["no"].best_ask
            n, sz = kelly_contracts(0.5, cost, bankroll, half_kelly=HALF_KELLY)
            frac  = sz / bankroll if bankroll else 0
            url   = f"https://polymarket.com/event/{r['slug']}" if r.get("slug") else ""
            signals.append({
                "type": "bundle_buy", "question": r["question"], "url": url,
                "entry_cost": cost, "edge_pct": r["edge_buy"],
                "num_contracts": n, "dollar_size": sz, "kelly_frac": frac,
                "platform": "polymarket",
            })

    # Bundle SELL arbs
    for r in results:
        if r.get("edge_sell") and r["edge_sell"] > poly_scanner.MIN_EDGE:
            revenue = r["yes"].best_bid + r["no"].best_bid
            n, sz = kelly_contracts(0.5, 1.0 - revenue, bankroll, half_kelly=HALF_KELLY)
            frac  = sz / bankroll if bankroll else 0
            url   = f"https://polymarket.com/event/{r['slug']}" if r.get("slug") else ""
            signals.append({
                "type": "bundle_sell", "question": r["question"], "url": url,
                "entry_cost": revenue, "edge_pct": r["edge_sell"],
                "num_contracts": n, "dollar_size": sz, "kelly_frac": frac,
                "platform": "polymarket",
            })

    # MM leads (strong only: ≥ 4¢)
    for r in results:
        yes_sp = r["yes"].spread or 0
        no_sp  = r["no"].spread  or 0
        best_sp = max(yes_sp, no_sp)
        if best_sp >= 0.04:
            mid  = r["yes"].mid or 0.5
            n, sz = kelly_mm(best_sp, mid, bankroll, half_kelly=HALF_KELLY)
            frac  = sz / bankroll if bankroll else 0
            url   = f"https://polymarket.com/event/{r['slug']}" if r.get("slug") else ""
            signals.append({
                "type": "mm", "question": r["question"], "url": url,
                "entry_cost": mid, "edge_pct": best_sp / 2,
                "num_contracts": n, "dollar_size": sz, "kelly_frac": frac,
                "platform": "polymarket",
            })

    return signals


# ── Cross-platform scanner ────────────────────────────────────────────────────

async def _cross_signals(
    client: httpx.AsyncClient,
    poly_liquid: list[dict],
    bankroll: float,
) -> list[dict]:
    """
    Run Kalshi + PredictIt cross-scan on the pre-fetched Polymarket list.
    Returns signal dicts in the same format as _poly_signals.
    """
    signals = []

    # ── Kalshi ────────────────────────────────────────────────────────────────
    kalshi_raw = await cross_scan.get_kalshi_markets(client)
    kalshi = [m for m in kalshi_raw if cross_scan.kalshi_yes_mid(m) is not None]
    kalshi_mids = {id(km): cross_scan.kalshi_yes_mid(km) for km in kalshi}

    def poly_gamma_mid(pm):
        try:
            prices = json.loads(pm.get("outcomePrices") or "[]")
            if prices:
                return float(prices[0])
        except Exception:
            return None

    pairs = []
    for pm in poly_liquid:
        pq = pm.get("question") or ""
        if not pq:
            continue
        pm_mid = poly_gamma_mid(pm)
        pq_cat = cross_scan.category(pq)
        if pq_cat == "other":
            continue
        best_score, best_km = 0.0, None
        for km in kalshi:
            kt = km.get("title") or ""
            if cross_scan.category(kt) != pq_cat:
                continue
            km_mid = kalshi_mids.get(id(km))
            if pm_mid is not None and km_mid is not None:
                if abs(pm_mid - km_mid) > 0.30:
                    continue
            s = min(
                cross_scan.similarity(pq, kt)
                + cross_scan.entity_boost(pq, kt)
                - cross_scan.discriminator_penalty(pq, kt),
                1.0,
            )
            if s > best_score:
                best_score, best_km = s, km
        if best_score >= cross_scan.SIM_THRESH and best_km:
            pairs.append((best_score, pm, best_km))

    pairs.sort(key=lambda x: -x[0])
    for score, pm, km in pairs[:30]:
        ob, _ = await cross_scan.get_poly_ob(client, pm.get("clobTokenIds", ""))
        if not ob:
            continue
        km_yes = float(km.get("yes_ask_dollars") or 0)
        km_no  = float(km.get("no_ask_dollars")  or 0)
        slug   = pm.get("slug") or pm.get("market_slug") or ""
        ke     = km.get("event_ticker") or km.get("ticker") or ""

        for yes_price, no_price, label in [
            (ob["yes_ask"], km_no, "cross_kalshi"),
            (km_yes, ob["no_ask"], "cross_kalshi"),
        ]:
            if not yes_price or not no_price:
                continue
            cost  = yes_price + no_price
            fp    = round(yes_price * cross_scan.POLY_FEE, 5)
            fk    = round(cross_scan.kalshi_fee(no_price), 5)
            fees  = fp + fk
            net   = round(1.0 - cost - fees, 4)
            if net >= cross_scan.MIN_EDGE:
                n, sz = kelly_contracts(0.5, cost, bankroll, half_kelly=HALF_KELLY)
                frac  = sz / bankroll if bankroll else 0
                signals.append({
                    "type": label,
                    "question": (pm.get("question") or "")[:80],
                    "url": f"https://polymarket.com/event/{slug}" if slug else "",
                    "url2": f"https://kalshi.com/markets/{ke}" if ke else "",
                    "entry_cost": cost, "edge_pct": net,
                    "num_contracts": n, "dollar_size": sz, "kelly_frac": frac,
                    "platform": "cross_kalshi",
                })
        await asyncio.sleep(0.03)

    # ── PredictIt ─────────────────────────────────────────────────────────────
    pi_markets = await asyncio.get_event_loop().run_in_executor(
        None, get_predictit_markets
    )
    for pm in poly_liquid:
        pq = pm.get("question") or ""
        if not pq or cross_scan.category(pq) == "other":
            continue
        pm_mid = poly_gamma_mid(pm)
        pq_cat = cross_scan.category(pq)
        best_score, best_pi = 0.0, None
        for pi in pi_markets:
            if cross_scan.category(pi.get("question") or "") != pq_cat:
                continue
            if pm_mid is not None and pi.get("mid") is not None:
                if abs(pm_mid - pi["mid"]) > 0.30:
                    continue
            s = min(
                cross_scan.similarity(pq, pi.get("question") or "")
                + cross_scan.entity_boost(pq, pi.get("question") or "")
                - cross_scan.discriminator_penalty(pq, pi.get("question") or ""),
                1.0,
            )
            if s > best_score:
                best_score, best_pi = s, pi
        if best_score >= cross_scan.SIM_THRESH and best_pi:
            ob, _ = await cross_scan.get_poly_ob(client, pm.get("clobTokenIds", ""))
            if not ob:
                continue
            slug   = pm.get("slug") or pm.get("market_slug") or ""
            ya, na = best_pi.get("yes_ask"), best_pi.get("no_ask")
            for yes_price, no_price in [(ob["yes_ask"], na), (ya, ob["no_ask"])]:
                if not yes_price or not no_price:
                    continue
                cost  = yes_price + no_price
                fp    = round(yes_price * cross_scan.POLY_FEE, 5)
                fpi   = round(predictit_fee(yes_price), 5)
                net   = round(1.0 - cost - fp - fpi, 4)
                if net >= cross_scan.MIN_EDGE:
                    n, sz = kelly_contracts(0.5, cost, bankroll, half_kelly=HALF_KELLY)
                    frac  = sz / bankroll if bankroll else 0
                    signals.append({
                        "type": "cross_predictit",
                        "question": pq[:80],
                        "url":  f"https://polymarket.com/event/{slug}" if slug else "",
                        "url2": best_pi.get("url", ""),
                        "entry_cost": cost, "edge_pct": net,
                        "num_contracts": n, "dollar_size": sz, "kelly_frac": frac,
                        "platform": "cross_predictit",
                    })
            await asyncio.sleep(0.03)

    return signals


# ── Output ────────────────────────────────────────────────────────────────────

def _print_signals(signals: list[dict], new_keys: set[str], bankroll: float):
    if not signals:
        print("  ✓ No signals above threshold this scan.")
        return

    arbs = [s for s in signals if s["type"] in ("bundle_buy", "bundle_sell")]
    cross = [s for s in signals if "cross" in s["type"]]
    mm   = [s for s in signals if s["type"] == "mm"]

    for section, items in [
        ("BUNDLE ARB (Polymarket)", arbs),
        ("CROSS-PLATFORM ARB", cross),
        ("MM LEADS (≥ 4¢ spread)", mm),
    ]:
        if not items:
            continue
        print(f"\n  ── {section} ──")
        for s in items[:5]:
            key = s["question"][:60]
            tag = " [NEW]" if key in new_keys else ""
            print(f"  [{s['type']:<15}]{tag} edge {s['edge_pct']*100:+.2f}%  "
                  f"Kelly {s['kelly_frac']*100:.1f}%  "
                  f"{s['num_contracts']}× @ ${s['entry_cost']:.3f}  "
                  f"size ${s['dollar_size']:.2f}")
            print(f"    {s['question'][:70]}")
            print(f"    {s['url']}")
            if s.get("url2"):
                print(f"    {s['url2']}")


# ── Main loop ─────────────────────────────────────────────────────────────────

async def run(interval: int, bankroll: float):
    paper   = PaperTrader()
    prev_keys: set[str] = set()
    scan_n  = 0

    print("=" * 72)
    print(f"  POLYMARKET ARBITRAGE MONITOR  —  demo bankroll ${bankroll:,.0f}")
    print(f"  Scan interval: {interval}s   Half-Kelly sizing   Ctrl-C to stop")
    print("=" * 72)

    while True:
        scan_n += 1
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'─'*72}")
        print(f"  SCAN #{scan_n}  {ts}")
        print(f"{'─'*72}")

        try:
            async with httpx.AsyncClient() as client:
                # 1. Polymarket single-venue scan
                print("  [1/2] Polymarket scan…", end=" ", flush=True)
                poly_signals = await _poly_signals(client, bankroll)
                print(f"{len(poly_signals)} signal(s)")

                # 2. Cross-platform (Kalshi + PredictIt)
                print("  [2/2] Cross-platform scan (Kalshi + PredictIt)…", end=" ", flush=True)
                raw_poly = await poly_scanner.fetch_markets(client, poly_scanner.FETCH_N)
                poly_liquid = [
                    m for m in raw_poly
                    if float(m.get("volume24hr") or 0) >= poly_scanner.MIN_VOLUME
                ]
                cross_signals = await _cross_signals(client, poly_liquid, bankroll)
                print(f"{len(cross_signals)} signal(s)")

        except Exception as e:
            print(f"  [ERR] scan failed: {e}")
            poly_signals, cross_signals = [], []

        all_signals = poly_signals + cross_signals
        cur_keys    = {s["question"][:60] for s in all_signals}
        new_keys    = cur_keys - prev_keys

        _print_signals(all_signals, new_keys, bankroll)

        # Paper trade: open a demo position for each new signal
        open_count = sum(1 for t in paper._trades if t.status == "open")
        for s in all_signals:
            key = s["question"][:60]
            if key in new_keys and not paper.already_open(s["question"]):
                if open_count < MAX_OPEN_TRADES:
                    paper.open_trade(
                        signal_type=s["type"],
                        question=s["question"],
                        url=s.get("url", ""),
                        entry_cost=s["entry_cost"],
                        edge_pct=s["edge_pct"],
                        num_contracts=s["num_contracts"],
                        dollar_size=s["dollar_size"],
                        kelly_frac=s["kelly_frac"],
                        platform=s.get("platform", "polymarket"),
                    )
                    open_count += 1

        prev_keys = cur_keys
        paper.auto_expire(max_age_hours=48)

        # Print paper trading report
        print()
        print(paper.report())

        print(f"\n  Next scan in {interval}s…")
        await asyncio.sleep(interval)


def main():
    ap = argparse.ArgumentParser(description="Polymarket arbitrage monitor")
    ap.add_argument("--interval",  type=int,   default=DEFAULT_INTERVAL,
                    help="Seconds between scans (default 90)")
    ap.add_argument("--bankroll",  type=float, default=DEFAULT_BANKROLL,
                    help="Demo bankroll for Kelly sizing (default $1,000)")
    args = ap.parse_args()

    try:
        asyncio.run(run(args.interval, args.bankroll))
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()

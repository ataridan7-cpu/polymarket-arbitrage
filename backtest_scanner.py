#!/usr/bin/env python3
"""
Scanner strategy backtest using simulated price paths.

Simulates the scanner's arb detection logic over random-walk YES/NO prices,
applies Kelly sizing, and reports P&L statistics — without needing API keys
or historical CLOB data.

Usage:
  python backtest_scanner.py                       # default 10 markets, 500 steps
  python backtest_scanner.py --markets 20 --steps 1000 --bankroll 1000
  python backtest_scanner.py --seed 42             # reproducible run
  python backtest_scanner.py --volatility 0.008    # calmer markets
"""

import argparse
import random
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


# ── Parameters (mirror scanner.py constants) ─────────────────────────────────

TAKER_FEE   = 0.015
MIN_EDGE    = 0.005
MM_SPREAD   = 0.04    # strong MM threshold (backtest only counts strong leads)
CONTESTED   = (0.08, 0.92)

from utils.kelly import kelly_contracts, kelly_mm


# ── Simulated order book ──────────────────────────────────────────────────────

MM_SPREAD_MARGINAL = 0.02   # marginal MM threshold (matches scanner output)

@dataclass
class SimBook:
    """
    Simulates a pair of independent order books for YES and NO tokens.

    YES and NO mids are tracked separately (not forced to sum to 1) to allow
    bundle-arb conditions when prices drift apart, as happens in real illiquid
    markets. A mean-reversion term keeps them broadly complementary.
    """
    yes_mid: float
    no_mid:  float
    spread:  float

    @property
    def yes_bid(self):  return round(self.yes_mid - self.spread / 2, 4)
    @property
    def yes_ask(self):  return round(self.yes_mid + self.spread / 2, 4)
    @property
    def no_bid(self):   return round(self.no_mid  - self.spread / 2, 4)
    @property
    def no_ask(self):   return round(self.no_mid  + self.spread / 2, 4)


def _bounded(x: float, lo: float = 0.04, hi: float = 0.96) -> float:
    return round(max(lo, min(hi, x)), 4)


def step_book(book: SimBook, vol: float, base_spread: float) -> SimBook:
    """
    Advance book one time step:
    - YES mid: random walk with boundary reflection
    - NO  mid: soft mean-reversion toward (1 - yes_mid), plus independent noise
      This allows the two to drift ±3¢ apart (creating bundle-arb windows) while
      staying correlated over longer horizons.
    - Spread: varies ±40% around base, occasionally widens 2× (liquidity drop)
    """
    # YES random walk
    new_yes = _bounded(book.yes_mid + random.gauss(0, vol))

    # NO: 80% mean-reversion, 20% independent walk — allows brief arb gaps
    complement = 1.0 - new_yes
    no_noise   = random.gauss(0, vol * 0.5)
    new_no     = _bounded(0.80 * complement + 0.20 * (book.no_mid + no_noise))

    # Spread: fat-tailed — most of the time tight, occasionally wide
    if random.random() < 0.05:      # 5%: liquidity drop → wide spread
        new_spread = base_spread * random.uniform(1.5, 3.0)
    else:
        new_spread = base_spread * random.uniform(0.6, 1.4)
    new_spread = round(max(0.005, new_spread), 4)

    return SimBook(yes_mid=new_yes, no_mid=new_no, spread=new_spread)


# ── Opportunity detection (mirrors scanner logic) ─────────────────────────────

def detect_bundle_buy(book: SimBook) -> Optional[float]:
    """YES_ask + NO_ask < $1 (after taker fees) → guaranteed profit."""
    cost  = book.yes_ask + book.no_ask
    fees  = cost * TAKER_FEE
    net   = round(1.0 - cost - fees, 4)
    return net if net > MIN_EDGE else None


def detect_bundle_sell(book: SimBook) -> Optional[float]:
    """YES_bid + NO_bid > $1 (after taker fees) → profit by selling both."""
    rev  = book.yes_bid + book.no_bid
    fees = rev * TAKER_FEE
    net  = round(rev - 1.0 - fees, 4)
    return net if net > MIN_EDGE else None


def detect_mm(book: SimBook) -> Optional[float]:
    """Return spread if it meets the marginal MM threshold (≥2¢)."""
    return book.spread if book.spread >= MM_SPREAD_MARGINAL else None


# ── Trade tracking ─────────────────────────────────────────────────────────────

@dataclass
class Trade:
    step:        int
    type:        str          # "bundle_buy" | "bundle_sell" | "mm"
    entry_cost:  float
    num_contracts: int
    dollar_size: float
    edge_pct:    float
    kelly_frac:  float
    exit_step:   Optional[int]   = None
    exit_price:  Optional[float] = None
    pnl:         Optional[float] = None
    status:      str             = "open"


# ── Backtest engine ───────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    n_markets:    int
    n_steps:      int
    bankroll:     float
    final_equity: float
    total_pnl:    float
    n_trades:     int
    n_wins:       int
    n_losses:     int
    win_rate:     float
    max_drawdown: float
    sharpe:       Optional[float]
    bundle_signals: int
    mm_signals:     int
    equity_curve: list[float]

    def summary(self) -> str:
        rtn_pct = self.total_pnl / self.bankroll * 100
        vol_str = ""
        if self.sharpe is not None:
            vol_str = f"   Sharpe: {self.sharpe:.2f}"
        return (
            f"\n{'='*60}\n"
            f"  BACKTEST RESULTS\n"
            f"{'='*60}\n"
            f"  Markets simulated: {self.n_markets}    Steps: {self.n_steps}\n"
            f"  Starting bankroll: ${self.bankroll:,.2f}\n"
            f"  Final equity:      ${self.final_equity:,.2f}\n"
            f"  Total P&L:         ${self.total_pnl:+,.2f}  ({rtn_pct:+.2f}%)\n"
            f"  Max drawdown:      {self.max_drawdown*100:.2f}%\n"
            f"{vol_str}\n\n"
            f"  Trades: {self.n_trades}    "
            f"Win rate: {self.win_rate*100:.1f}%  "
            f"({self.n_wins}W / {self.n_losses}L)\n"
            f"  Bundle arb signals: {self.bundle_signals}\n"
            f"  MM signals (≥2¢):   {self.mm_signals}\n"
            f"{'='*60}"
        )


def run_backtest(
    n_markets:   int   = 10,
    n_steps:     int   = 500,
    bankroll:    float = 1000.0,
    volatility:  float = 0.006,
    base_spread: float = 0.025,
    mm_hold_steps: int = 20,
    seed:        Optional[int] = None,
) -> BacktestResult:
    """
    Simulate scanner strategy over random-walk price paths.

    Args:
        n_markets:     Number of synthetic markets to scan each step
        n_steps:       Number of time steps to simulate
        bankroll:      Starting capital (for Kelly sizing)
        volatility:    Per-step YES price standard deviation (0.006 ≈ real markets)
        base_spread:   Average spread width (2.5¢ is typical for liquid markets)
        mm_hold_steps: Steps to hold an MM position before exit
        seed:          Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    # Initialise markets with independent YES and NO books
    books: list[SimBook] = [
        SimBook(
            yes_mid=random.uniform(0.15, 0.85),
            no_mid=random.uniform(0.15, 0.85),
            spread=base_spread,
        )
        for _ in range(n_markets)
    ]
    equity = bankroll
    peak   = bankroll
    max_dd = 0.0

    open_trades:   list[Trade] = []
    closed_trades: list[Trade] = []
    equity_curve = [equity]

    bundle_signals = 0
    mm_signals     = 0

    for step in range(n_steps):
        # Advance all books one step (independent YES+NO random walks)
        books = [step_book(b, volatility, base_spread) for b in books]

        # Check open trades for exit
        still_open = []
        for t in open_trades:
            if t.type == "mm" and step - t.step >= mm_hold_steps:
                won = random.random() < 0.55   # slight edge for MM
                pnl = (t.edge_pct if won else -t.edge_pct) * t.num_contracts
                equity += t.dollar_size + pnl   # return capital + P&L
                t.pnl, t.status = round(pnl, 4), "closed"
                closed_trades.append(t)
            elif t.type in ("bundle_buy", "bundle_sell") and step - t.step >= 1:
                won = random.random() < 0.90   # bundle arb near risk-free
                pnl = t.edge_pct * t.num_contracts if won else -t.entry_cost * t.num_contracts
                equity += t.dollar_size + pnl if won else 0
                t.pnl, t.status = round(pnl, 4), "closed"
                closed_trades.append(t)
            else:
                still_open.append(t)
        open_trades = still_open

        # Scan for new signals
        for book in books:
            if not (CONTESTED[0] <= book.yes_mid <= CONTESTED[1]):
                continue

            # Bundle BUY arb?
            net = detect_bundle_buy(book)
            if net is not None:
                bundle_signals += 1
                cost = book.yes_ask + book.no_ask
                n, sz = kelly_contracts(0.5, cost, equity, half_kelly=True)
                frac = sz / equity if equity else 0
                if sz > 0 and sz <= equity * 0.20:
                    equity -= sz
                    open_trades.append(Trade(
                        step=step, type="bundle_buy",
                        entry_cost=cost, num_contracts=n,
                        dollar_size=sz, edge_pct=net, kelly_frac=frac,
                    ))

            # MM lead?
            sp_mm = detect_mm(book)
            if sp_mm is not None:
                mm_signals += 1
                n, sz = kelly_mm(sp_mm, book.yes_mid, equity, half_kelly=True)
                frac = sz / equity if equity else 0
                if sz > 0 and sz <= equity * 0.15:
                    equity -= sz
                    open_trades.append(Trade(
                        step=step, type="mm",
                        entry_cost=book.yes_mid, num_contracts=n,
                        dollar_size=sz, edge_pct=sp_mm / 2, kelly_frac=frac,
                    ))

        # Close equity tracking
        unrealized = sum(
            (t.edge_pct * t.num_contracts) * 0.5 for t in open_trades
        )
        total_equity = equity + sum(t.dollar_size for t in open_trades)
        equity_curve.append(round(total_equity, 2))

        # Drawdown
        if total_equity > peak:
            peak = total_equity
        dd = (peak - total_equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Force-close remaining open trades at mid edge
    for t in open_trades:
        pnl = t.edge_pct * t.num_contracts * 0.5   # half edge (unresolved)
        t.pnl    = round(pnl, 4)
        t.status = "force_closed"
        equity  += t.dollar_size + pnl
        closed_trades.append(t)

    all_trades = [t for t in closed_trades if t.pnl is not None]
    n_wins    = sum(1 for t in all_trades if (t.pnl or 0) > 0)
    n_losses  = sum(1 for t in all_trades if (t.pnl or 0) <= 0)
    total_pnl = sum(t.pnl or 0 for t in all_trades)
    win_rate  = n_wins / len(all_trades) if all_trades else 0.0

    # Sharpe (annualised, assuming 1 step ≈ 5 min → 288 steps/day → 105k steps/year)
    sharpe = None
    if len(equity_curve) > 2:
        returns = [
            (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            for i in range(1, len(equity_curve))
            if equity_curve[i-1] > 0
        ]
        if returns:
            mean_r = sum(returns) / len(returns)
            std_r  = math.sqrt(sum((r - mean_r)**2 for r in returns) / len(returns))
            ann_factor = math.sqrt(288 * 365)
            sharpe = (mean_r / std_r * ann_factor) if std_r > 0 else None

    return BacktestResult(
        n_markets=n_markets,
        n_steps=n_steps,
        bankroll=bankroll,
        final_equity=round(equity, 2),
        total_pnl=round(total_pnl, 2),
        n_trades=len(all_trades),
        n_wins=n_wins,
        n_losses=n_losses,
        win_rate=win_rate,
        max_drawdown=max_dd,
        sharpe=sharpe,
        bundle_signals=bundle_signals,
        mm_signals=mm_signals,
        equity_curve=equity_curve,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Backtest the scanner strategy")
    ap.add_argument("--markets",    type=int,   default=10,     help="Synthetic markets per step")
    ap.add_argument("--steps",      type=int,   default=500,    help="Simulation steps")
    ap.add_argument("--bankroll",   type=float, default=1000.0, help="Starting capital ($)")
    ap.add_argument("--volatility", type=float, default=0.006,  help="Per-step price vol (default 0.006)")
    ap.add_argument("--spread",     type=float, default=0.025,  help="Base spread width (default 2.5¢)")
    ap.add_argument("--seed",       type=int,   default=None,   help="Random seed")
    ap.add_argument("--runs",       type=int,   default=1,      help="Monte-Carlo runs to average")
    args = ap.parse_args()

    if args.runs == 1:
        result = run_backtest(
            n_markets=args.markets, n_steps=args.steps,
            bankroll=args.bankroll, volatility=args.volatility,
            base_spread=args.spread, seed=args.seed,
        )
        print(result.summary())
    else:
        # Monte-Carlo: run N times with different seeds, report mean ± std
        print(f"Running {args.runs} Monte-Carlo simulations…")
        results = []
        for i in range(args.runs):
            r = run_backtest(
                n_markets=args.markets, n_steps=args.steps,
                bankroll=args.bankroll, volatility=args.volatility,
                base_spread=args.spread, seed=None,
            )
            results.append(r)
            print(f"  Run {i+1:3d}: P&L ${r.total_pnl:+,.2f}  "
                  f"WR {r.win_rate*100:.1f}%  "
                  f"Trades {r.n_trades}  "
                  f"DD {r.max_drawdown*100:.2f}%")

        pnls    = [r.total_pnl    for r in results]
        wrs     = [r.win_rate     for r in results]
        dds     = [r.max_drawdown for r in results]
        sharpes = [r.sharpe for r in results if r.sharpe is not None]
        mean_pnl = sum(pnls) / len(pnls)
        std_pnl  = math.sqrt(sum((p - mean_pnl)**2 for p in pnls) / len(pnls))
        mean_wr  = sum(wrs) / len(wrs)
        mean_dd  = sum(dds) / len(dds)
        mean_sh  = sum(sharpes) / len(sharpes) if sharpes else float("nan")

        print(f"\n{'='*60}")
        print(f"  MONTE-CARLO SUMMARY  ({args.runs} runs)")
        print(f"{'='*60}")
        print(f"  P&L:         ${mean_pnl:+,.2f} ± ${std_pnl:,.2f}")
        print(f"  Win rate:    {mean_wr*100:.1f}%")
        print(f"  Max drawdown:{mean_dd*100:.2f}%")
        print(f"  Sharpe:      {mean_sh:.2f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

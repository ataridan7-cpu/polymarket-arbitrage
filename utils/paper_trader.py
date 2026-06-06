"""
Paper (demo) trading ledger.

Records signals as demo trades, tracks unrealized P&L at each scan,
and closes positions when they expire or are manually resolved.
Persists to logs/demo_trades.json between runs.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

_DEFAULT_FILE = "logs/demo_trades.json"


@dataclass
class DemoTrade:
    id: str
    timestamp: float
    signal_type: str          # "bundle_buy" | "bundle_sell" | "mm" | "cross"
    platform: str             # "polymarket" | "kalshi" | "predictit" | "cross"
    question: str
    url: str
    entry_cost: float         # cost per contract at entry
    edge_pct: float           # net edge fraction at entry (e.g. 0.015 = 1.5%)
    num_contracts: int
    dollar_size: float
    kelly_frac: float
    status: str = "open"      # "open" | "closed" | "expired"
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    close_timestamp: Optional[float] = None
    close_reason: str = ""
    current_mid: Optional[float] = None
    unrealized_pnl: Optional[float] = None


class PaperTrader:
    """Records, tracks, and reports paper/demo trades."""

    def __init__(self, trades_file: str = _DEFAULT_FILE):
        self.trades_file = trades_file
        os.makedirs(os.path.dirname(trades_file), exist_ok=True)
        self._trades: list[DemoTrade] = self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> list[DemoTrade]:
        if not os.path.exists(self.trades_file):
            return []
        try:
            with open(self.trades_file) as f:
                return [DemoTrade(**r) for r in json.load(f)]
        except Exception:
            return []

    def _save(self):
        with open(self.trades_file, "w") as f:
            json.dump([asdict(t) for t in self._trades], f, indent=2)

    # ── trade lifecycle ───────────────────────────────────────────────────────

    def open_trade(
        self,
        signal_type: str,
        question: str,
        url: str,
        entry_cost: float,
        edge_pct: float,
        num_contracts: int,
        dollar_size: float,
        kelly_frac: float,
        platform: str = "polymarket",
    ) -> DemoTrade:
        t = DemoTrade(
            id=f"{signal_type}_{int(time.time()*1000)}",
            timestamp=time.time(),
            signal_type=signal_type,
            platform=platform,
            question=question,
            url=url,
            entry_cost=entry_cost,
            edge_pct=edge_pct,
            num_contracts=num_contracts,
            dollar_size=dollar_size,
            kelly_frac=kelly_frac,
        )
        self._trades.append(t)
        self._save()
        return t

    def update_mid(self, trade: DemoTrade, current_mid: float):
        """Refresh unrealized P&L for one open trade."""
        trade.current_mid = current_mid
        # Payout at resolution is $1/contract; unrealized = (mid − cost) * n
        trade.unrealized_pnl = round((current_mid - trade.entry_cost) * trade.num_contracts, 4)
        self._save()

    def close_trade(self, trade: DemoTrade, exit_price: float, reason: str = ""):
        trade.status = "closed"
        trade.exit_price = exit_price
        trade.realized_pnl = round((exit_price - trade.entry_cost) * trade.num_contracts, 4)
        trade.close_timestamp = time.time()
        trade.close_reason = reason
        trade.unrealized_pnl = None
        self._save()

    def auto_expire(self, max_age_hours: float = 48.0):
        """Mark trades older than max_age_hours as expired."""
        cutoff = time.time() - max_age_hours * 3600
        for t in self._trades:
            if t.status == "open" and t.timestamp < cutoff:
                mid = t.current_mid or t.entry_cost
                self.close_trade(t, mid, reason="expired")

    def already_open(self, question: str) -> bool:
        """True if a trade for this question is already in open state."""
        key = question[:60]
        return any(t.status == "open" and t.question[:60] == key for t in self._trades)

    # ── reporting ─────────────────────────────────────────────────────────────

    def report(self) -> str:
        open_t   = [t for t in self._trades if t.status == "open"]
        closed_t = [t for t in self._trades if t.status in ("closed", "expired")]

        realized   = sum(t.realized_pnl or 0 for t in closed_t)
        unrealized = sum(t.unrealized_pnl or 0 for t in open_t)
        wins  = sum(1 for t in closed_t if (t.realized_pnl or 0) > 0)
        total_closed = len(closed_t)

        lines = [
            "=" * 72,
            "  PAPER TRADING LEDGER",
            "=" * 72,
            f"",
            f"  Open: {len(open_t)}   Closed: {len(closed_t)}",
            f"  Realized P&L:   ${realized:+.2f}",
            f"  Unrealized P&L: ${unrealized:+.2f}",
            f"  Total P&L:      ${realized+unrealized:+.2f}",
        ]
        if total_closed:
            wr = f"{wins}/{total_closed} ({wins/total_closed*100:.0f}%)"
            lines.append(f"  Win rate:       {wr}")

        if open_t:
            lines += ["", "  OPEN POSITIONS:"]
            for t in open_t:
                upnl = f"${t.unrealized_pnl:+.2f}" if t.unrealized_pnl is not None else "  ?"
                age_h = (time.time() - t.timestamp) / 3600
                lines.append(
                    f"  [{t.signal_type:<11}] {t.question[:46]}…"
                )
                lines.append(
                    f"               {t.num_contracts}× @ ${t.entry_cost:.3f}  "
                    f"edge {t.edge_pct*100:.1f}%  size ${t.dollar_size:.2f}  "
                    f"unreal {upnl}  age {age_h:.1f}h"
                )

        if closed_t:
            lines += ["", "  LAST 10 CLOSED:"]
            for t in closed_t[-10:]:
                pnl_s = f"${t.realized_pnl:+.2f}" if t.realized_pnl is not None else "?"
                lines.append(
                    f"  [{t.signal_type:<11}] {t.question[:46]}…  "
                    f"P&L {pnl_s}  ({t.close_reason})"
                )

        lines.append("=" * 72)
        return "\n".join(lines)

"""Kelly Criterion position sizing for binary prediction market trades."""


def kelly_fraction(p_win: float, net_odds: float) -> float:
    """
    Pure Kelly fraction of bankroll: f = (b·p − q) / b
    b = net_odds (profit per $1 risked), p = win prob, q = 1 − p
    """
    if net_odds <= 0 or p_win <= 0 or p_win >= 1:
        return 0.0
    f = (net_odds * p_win - (1.0 - p_win)) / net_odds
    return max(0.0, f)


def kelly_contracts(
    p_win: float,
    cost_per_contract: float,
    bankroll: float,
    half_kelly: bool = True,
    max_fraction: float = 0.20,
) -> tuple[int, float]:
    """
    Kelly-optimal (num_contracts, dollar_cost) for a binary arb or bundle.

    p_win             — probability of winning leg (use 0.5 for symmetric
                        bundle arb; use YES mid for directional MM entry)
    cost_per_contract — total cost per unit (e.g. yes_ask + no_ask for bundle,
                        or mid price for a single-leg MM position)
    bankroll          — total capital to size against
    half_kelly        — divide by 2 to reduce variance (recommended)
    max_fraction      — hard cap: never risk more than this share of bankroll
    """
    if cost_per_contract <= 0 or bankroll <= 0:
        return 0, 0.0
    net_odds = 1.0 / cost_per_contract - 1.0
    frac = kelly_fraction(p_win, net_odds)
    if half_kelly:
        frac /= 2.0
    frac = min(frac, max_fraction)
    n = max(1, int(bankroll * frac / cost_per_contract))
    return n, round(n * cost_per_contract, 2)


def kelly_mm(
    spread: float,
    mid: float,
    bankroll: float,
    half_kelly: bool = True,
    max_fraction: float = 0.15,
) -> tuple[int, float]:
    """
    Kelly sizing for a market-making position.

    Direction-neutral (p_win = 0.5). Net odds approximated as half-spread
    divided by mid (the "edge" captured per dollar at risk).
    """
    if mid <= 0 or spread <= 0:
        return 0, 0.0
    net_odds = (spread / 2.0) / mid
    frac = kelly_fraction(0.5, net_odds)
    if half_kelly:
        frac /= 2.0
    frac = min(frac, max_fraction)
    n = max(1, int(bankroll * frac / mid))
    return n, round(n * mid, 2)

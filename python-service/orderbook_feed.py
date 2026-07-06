"""
polymarket_orderbook.py — Snapshot Polymarket orderbook depth + recent trade flow
for retrospective signal analysis.

Two record-only signals, designed to be cheap and audit-friendly. After a
day or two of data we can analyze whether these features correlate with
WIN/LOSS outcomes and promote any predictive ones to actual gates.

Endpoints:
  GET /trade-api/v2/markets/{ticker}/orderbook
      → orderbook_fp.yes_dollars: [[price_dollars, count_fp], ...] sorted ascending price
      → orderbook_fp.no_dollars:  same for NO leg
      Each leg shows BIDS in the leg's pricing (so a YES bid at $0.05 means
      buyer pays 5¢ for YES contract → equivalent to selling NO at 95¢).

  GET /trade-api/v2/markets/trades?ticker=...&limit=N
      → trades: array of {count_fp, taker_outcome_side, yes_price_dollars, no_price_dollars, ...}
      → taker_outcome_side: "yes" or "no" → which direction was the aggressor

Two functions are exposed: fetch_orderbook() and fetch_recent_trades().
Both cache results for 5 seconds to balance freshness against API load.
"""

from __future__ import annotations

import time
from typing import Optional, Callable, Awaitable


# Module-level caches: ticker -> (ts, summary). Cleared on TTL.
_OB_CACHE: dict = {}
_TR_CACHE: dict = {}
_CACHE_TTL_S = 5.0


# ── Helpers ───────────────────────────────────────────────────────────────────
def _to_cents(price_str) -> int:
    """'0.6500' → 65 (cents)."""
    try:
        return int(round(float(price_str) * 100))
    except (TypeError, ValueError):
        return 0


def _to_int(count_str) -> int:
    """Polymarket count_fp '36.00' → 36 (truncates fractional contracts from partial fills)."""
    try:
        return int(float(count_str))
    except (TypeError, ValueError):
        return 0


# ── Orderbook summarizer ──────────────────────────────────────────────────────
def summarize_orderbook(raw: dict) -> dict:
    """Compute aggregate metrics from a raw Polymarket orderbook response.

    Both yes_dollars and no_dollars are arrays of BIDS in their leg's pricing:
      yes_dollars: highest YES bid = top of YES book
      no_dollars:  highest NO bid  = top of NO book

    The implied YES ask = 100¢ - top NO bid (since selling YES = buying NO).
    """
    ob = raw.get("orderbook_fp") or raw.get("orderbook") or {}
    yes_book = ob.get("yes_dollars") or ob.get("yes") or []
    no_book  = ob.get("no_dollars")  or ob.get("no")  or []

    # Convert and filter zero-count entries
    yes_pairs = [(_to_cents(p), _to_int(c)) for p, c in yes_book]
    yes_pairs = [(p, c) for p, c in yes_pairs if c > 0 and 0 < p < 100]
    no_pairs  = [(_to_cents(p), _to_int(c)) for p, c in no_book]
    no_pairs  = [(p, c) for p, c in no_pairs  if c > 0 and 0 < p < 100]

    # Sort descending by price (highest = top of book)
    yes_pairs.sort(key=lambda x: -x[0])
    no_pairs.sort(key=lambda x: -x[0])

    # Top-of-book metrics
    yes_top_bid_px   = yes_pairs[0][0] if yes_pairs else None
    yes_top_bid_size = yes_pairs[0][1] if yes_pairs else 0
    no_top_bid_px    = no_pairs[0][0]  if no_pairs  else None
    no_top_bid_size  = no_pairs[0][1]  if no_pairs  else 0

    # Implied asks via symmetry (selling YES at X = buying NO at 100-X)
    yes_top_ask_px   = (100 - no_top_bid_px)  if no_top_bid_px  is not None else None
    yes_top_ask_size = no_top_bid_size
    no_top_ask_px    = (100 - yes_top_bid_px) if yes_top_bid_px is not None else None
    no_top_ask_size  = yes_top_bid_size

    # Spreads (in cents)
    yes_spread = ((yes_top_ask_px - yes_top_bid_px)
                  if (yes_top_ask_px is not None and yes_top_bid_px is not None)
                  else None)

    # Top-5-level depth aggregates
    yes_top5_size = sum(c for _p, c in yes_pairs[:5])
    no_top5_size  = sum(c for _p, c in no_pairs[:5])

    # Imbalance: > 0.5 = more depth on YES side (bullish stacking)
    total_top5 = yes_top5_size + no_top5_size
    imbalance_yes = (yes_top5_size / total_top5) if total_top5 > 0 else 0.5

    # Total depth across all levels (useful for "is this book thin?")
    total_yes_depth = sum(c for _p, c in yes_pairs)
    total_no_depth  = sum(c for _p, c in no_pairs)

    return {
        "yes_top_bid_px":   yes_top_bid_px,
        "yes_top_bid_size": yes_top_bid_size,
        "yes_top_ask_px":   yes_top_ask_px,
        "yes_top_ask_size": yes_top_ask_size,
        "no_top_bid_px":    no_top_bid_px,
        "no_top_bid_size":  no_top_bid_size,
        "no_top_ask_px":    no_top_ask_px,
        "no_top_ask_size":  no_top_ask_size,
        "yes_spread":       yes_spread,
        "yes_top5_size":    yes_top5_size,
        "no_top5_size":     no_top5_size,
        "imbalance_yes":    round(imbalance_yes, 3),
        "total_yes_depth":  total_yes_depth,
        "total_no_depth":   total_no_depth,
        "n_yes_levels":     len(yes_pairs),
        "n_no_levels":      len(no_pairs),
    }


# ── Recent trade flow summarizer ──────────────────────────────────────────────
def summarize_trades(raw: dict, lookback_n: int = 20) -> dict:
    """Compute trade flow aggression metrics from recent Polymarket trades.

    `taker_outcome_side` field tells us which direction was the AGGRESSOR
    (took the resting opposite order). A run of "yes" takers means buyers
    of YES are hitting offers — bullish flow. A run of "no" takers means
    NO buyers are aggressive — bearish flow.
    """
    trades = (raw.get("trades") or [])[:lookback_n]
    if not trades:
        return {
            "n_trades": 0,
            "total_volume": 0,
            "yes_taker_volume": 0,
            "no_taker_volume": 0,
            "yes_aggression": 0.5,
            "latest_yes_px": None,
            "latest_time": None,
        }

    yes_volume = 0
    no_volume  = 0
    total      = 0
    for t in trades:
        size = _to_int(t.get("count_fp"))
        side = t.get("taker_outcome_side") or t.get("taker_side")
        total += size
        if side == "yes":
            yes_volume += size
        elif side == "no":
            no_volume += size

    yes_aggression = (yes_volume / total) if total > 0 else 0.5

    latest = trades[0]
    return {
        "n_trades":         len(trades),
        "total_volume":     total,
        "yes_taker_volume": yes_volume,
        "no_taker_volume":  no_volume,
        "yes_aggression":   round(yes_aggression, 3),
        "latest_yes_px":    _to_cents(latest.get("yes_price_dollars", "0")),
        "latest_time":      latest.get("created_time"),
    }


# ── Async fetchers with caching ───────────────────────────────────────────────
VenueGetFn = Callable[..., Awaitable[dict]]


async def fetch_orderbook(ticker: str, kget: VenueGetFn) -> Optional[dict]:
    """Fetch + summarize orderbook for `ticker`. Cached for 5s per-ticker."""
    now = time.time()
    cached = _OB_CACHE.get(ticker)
    if cached and (now - cached[0] < _CACHE_TTL_S):
        return cached[1]
    try:
        raw = await kget(f"/markets/{ticker}/orderbook")
        summary = summarize_orderbook(raw)
        _OB_CACHE[ticker] = (now, summary)
        return summary
    except Exception:
        return None


async def fetch_recent_trades(ticker: str, kget: VenueGetFn,
                              limit: int = 20) -> Optional[dict]:
    """Fetch + summarize recent trades for `ticker`. Cached for 5s per-ticker."""
    now = time.time()
    cached = _TR_CACHE.get(ticker)
    if cached and (now - cached[0] < _CACHE_TTL_S):
        return cached[1]
    try:
        raw = await kget("/markets/trades", {"ticker": ticker, "limit": limit})
        summary = summarize_trades(raw, lookback_n=limit)
        _TR_CACHE[ticker] = (now, summary)
        return summary
    except Exception:
        return None

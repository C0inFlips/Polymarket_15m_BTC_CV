"""
polymarket_client.py — Polymarket execution layer for the BTC 15m daemon.

Drop-in replacement for the old Kalshi transport. The trading daemon was
originally written against Kalshi's trade-api/v2 REST schema; rather than
rewriting 10k lines of battle-tested monitoring/exit logic, this module
exposes the SAME logical surface (paths, request bodies, response shapes)
and translates everything to Polymarket underneath:

  • Market discovery .... Gamma API  (https://gamma-api.polymarket.com)
                          15-minute BTC "Up or Down" series; slug format
                          btc-updown-15m-{unix_window_start}.
  • Strike discovery .... Polymarket crypto-price endpoint (the "price to
                          beat" = first Chainlink BTC/USD tick at/after the
                          window open). Falls back to a self-captured spot
                          sample at the window boundary.
  • Order execution ..... CLOB (https://clob.polymarket.com) via the
                          official py-clob-client-v2 (EIP-712 signed orders,
                          CLOB V2 — V1 retired 2026-04-28).
                          IOC ⇒ FAK, GTC ⇒ GTC.
  • Positions/balance ... CLOB balance-allowance + Data API positions
                          (https://data-api.polymarket.com).
  • Trades/orderbook .... CLOB books + Data API trade tape.

Mapping conventions (identical to the old exchange semantics):
  YES  == "Up"   outcome token
  NO   == "Down" outcome token
  All prices the daemon sees are integer CENTS on the YES leg, exactly as
  before. Order bodies quote the YES leg as a 0-1 decimal ("price") with
  side "bid" (buy YES exposure) / "ask" (sell YES exposure); the adapter
  nets against the currently-held token just like the old exchange did:
    side=bid  → sell held Down tokens first, then buy Up
    side=ask  → sell held Up tokens first, then buy Down

Environment variables (see .env.local.example):
  POLYMARKET_PRIVATE_KEY      wallet private key that signs CLOB orders
  POLYMARKET_PROXY_ADDRESS    funder address — the wallet that actually holds
                              funds/positions. For website accounts this is
                              the profile "Address … For API use only"
                              (deposit wallet). Leave empty for a raw EOA.
  POLYMARKET_SIGNATURE_TYPE   0=EOA, 1=legacy Magic proxy, 2=legacy browser
                              proxy, 3=deposit wallet (POLY_1271 — current
                              website accounts). Defaults: 3 when
                              PROXY_ADDRESS set, else 0.
  POLYMARKET_API_KEY/SECRET/PASSPHRASE   optional pre-derived L2 creds;
                              when absent they are derived from the key.
  POLYMARKET_HOST             default https://clob.polymarket.com
  POLYMARKET_GAMMA_HOST       default https://gamma-api.polymarket.com
  POLYMARKET_DATA_HOST        default https://data-api.polymarket.com
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

log = logging.getLogger("daemon")

# ── Config ─────────────────────────────────────────────────────────────────────
POLYMARKET_HOST       = os.environ.get("POLYMARKET_HOST", "https://clob.polymarket.com")
POLYMARKET_GAMMA_HOST = os.environ.get("POLYMARKET_GAMMA_HOST", "https://gamma-api.polymarket.com")
POLYMARKET_DATA_HOST  = os.environ.get("POLYMARKET_DATA_HOST", "https://data-api.polymarket.com")
POLYMARKET_SITE_HOST  = os.environ.get("POLYMARKET_SITE_HOST", "https://polymarket.com")

POLYMARKET_PRIVATE_KEY   = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
POLYMARKET_PROXY_ADDRESS = os.environ.get("POLYMARKET_PROXY_ADDRESS", "")
# Signature type 3 (POLY_1271 deposit wallet) is what current website-created
# accounts use; 1/2 remain for legacy Magic/browser proxy accounts.
_sig_default = "3" if POLYMARKET_PROXY_ADDRESS else "0"
POLYMARKET_SIGNATURE_TYPE = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE") or _sig_default)
POLYMARKET_API_KEY        = os.environ.get("POLYMARKET_API_KEY", "")
POLYMARKET_API_SECRET     = os.environ.get("POLYMARKET_API_SECRET", "")
POLYMARKET_API_PASSPHRASE = os.environ.get("POLYMARKET_API_PASSPHRASE", "")

CHAIN_ID    = 137           # Polygon mainnet
SERIES_SLUG = "btc-updown-15m"
WINDOW_S    = 900           # 15 minutes

# Slug pattern for BTC 15m up/down markets — used to filter our positions.
TICKER_PREFIX = "btc-updown-15m-"
_SLUG_RE = re.compile(r"^btc-updown-15m-(\d+)$")


# ── In-memory state ────────────────────────────────────────────────────────────
# Market cache: slug -> (fetched_at_unix, market_dict)  (Kalshi-shaped)
_MARKET_CACHE: dict = {}
_MARKET_CACHE_TTL_S = 2.0          # books move fast; keep this tight

# Gamma metadata cache (token ids etc. never change): slug -> meta dict
_META_CACHE: dict = {}

# Strike ("price to beat") cache: slug -> float
_STRIKE_CACHE: dict = {}
# Fallback boundary spot samples captured by ourselves: window_start -> price
_BOUNDARY_SPOT: dict = {}

# Settlement result cache: slug -> "yes"/"no"
_RESULT_CACHE: dict = {}

# Order registry: order_id -> {"slug","token_id","outcome","side","size",
#                              "price_token","order_type"}
_ORDER_REGISTRY: dict = {}

# Net YES-exposure tracker per slug (positive = long Up, negative = long Down).
# Seeded from the Data API on first use, then maintained from our own fills.
_NET_POSITION: dict = {}
_NET_POSITION_SEEDED: set = set()

_clob_client = None
_clob_lock = threading.Lock()
_funder_address: Optional[str] = None


# ── Auth / client bootstrap ────────────────────────────────────────────────────
def _get_clob_client():
    """Build (once) and return the authenticated ClobClient. L1 auth from the
    wallet private key; L2 API creds from env or derived on the fly."""
    global _clob_client
    if _clob_client is not None:
        return _clob_client
    with _clob_lock:
        if _clob_client is not None:
            return _clob_client
        if not POLYMARKET_PRIVATE_KEY:
            raise RuntimeError(
                "POLYMARKET_PRIVATE_KEY not set — cannot sign Polymarket orders. "
                "Add it to .env.local (see .env.local.example).")
        from py_clob_client_v2 import ClobClient
        from py_clob_client_v2.clob_types import ApiCreds

        kwargs: dict = {
            "key": POLYMARKET_PRIVATE_KEY,
            "chain_id": CHAIN_ID,
        }
        if POLYMARKET_PROXY_ADDRESS:
            kwargs["funder"] = POLYMARKET_PROXY_ADDRESS
            kwargs["signature_type"] = POLYMARKET_SIGNATURE_TYPE
        elif POLYMARKET_SIGNATURE_TYPE:
            kwargs["signature_type"] = POLYMARKET_SIGNATURE_TYPE

        client = ClobClient(POLYMARKET_HOST, **kwargs)
        if POLYMARKET_API_KEY and POLYMARKET_API_SECRET and POLYMARKET_API_PASSPHRASE:
            creds = ApiCreds(
                api_key=POLYMARKET_API_KEY,
                api_secret=POLYMARKET_API_SECRET,
                api_passphrase=POLYMARKET_API_PASSPHRASE,
            )
        else:
            creds = client.create_or_derive_api_key()
        client.set_api_creds(creds)
        _clob_client = client
        log.info("Polymarket CLOB client ready (addr=%s, funder=%s, sig_type=%d)",
                 client.get_address(), POLYMARKET_PROXY_ADDRESS or "EOA",
                 POLYMARKET_SIGNATURE_TYPE)
        return _clob_client


def get_wallet_address() -> Optional[str]:
    """Address that holds funds/positions (proxy wallet if configured)."""
    global _funder_address
    if _funder_address:
        return _funder_address
    if POLYMARKET_PROXY_ADDRESS:
        _funder_address = POLYMARKET_PROXY_ADDRESS
        return _funder_address
    try:
        _funder_address = _get_clob_client().get_address()
    except Exception:
        _funder_address = None
    return _funder_address


# ── Small HTTP helpers ─────────────────────────────────────────────────────────
async def _http_get_json(url: str, params: dict = None, timeout: float = 15.0):
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(url, params=params or {},
                        headers={"User-Agent": "btc15m-daemon/1.0"})
        r.raise_for_status()
        return r.json()


def _to_cents(p) -> Optional[int]:
    try:
        v = round(float(p) * 100)
        return v if 0 < v < 100 else None
    except (TypeError, ValueError):
        return None


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slug_for_window_start(start_ts: int) -> str:
    return f"{TICKER_PREFIX}{int(start_ts)}"


def window_start_from_slug(slug: str) -> Optional[int]:
    m = _SLUG_RE.match(slug or "")
    return int(m.group(1)) if m else None


# ── Gamma metadata (token ids, condition id, end date) ────────────────────────
async def _fetch_meta(slug: str) -> Optional[dict]:
    """Static metadata for a 15m BTC market. Cached forever per slug."""
    meta = _META_CACHE.get(slug)
    if meta is not None:
        return meta
    try:
        events = await _http_get_json(f"{POLYMARKET_GAMMA_HOST}/events",
                                      {"slug": slug})
    except Exception as e:
        log.debug(f"gamma events fetch failed for {slug}: {e}")
        return None
    if not events:
        return None
    ev = events[0]
    markets = ev.get("markets") or []
    if not markets:
        return None
    m = markets[0]
    try:
        outcomes  = json.loads(m.get("outcomes") or "[]")
        token_ids = json.loads(m.get("clobTokenIds") or "[]")
    except (ValueError, TypeError):
        return None
    tok_up = tok_down = None
    for o, t in zip(outcomes, token_ids):
        if str(o).strip().lower() == "up":
            tok_up = t
        elif str(o).strip().lower() == "down":
            tok_down = t
    if not tok_up or not tok_down:
        return None
    start_ts = window_start_from_slug(slug)
    meta = {
        "slug":          slug,
        "condition_id":  m.get("conditionId"),
        "question_id":   m.get("questionID"),
        "market_id":     m.get("id"),
        "token_id_yes":  tok_up,      # "Up"   == YES
        "token_id_no":   tok_down,    # "Down" == NO
        "neg_risk":      bool(m.get("negRisk")),
        "tick_size":     float(m.get("orderPriceMinTickSize") or 0.01),
        "min_size":      float(m.get("orderMinSize") or 5),
        "end_date":      m.get("endDate") or ev.get("endDate"),
        "event_start":   m.get("eventStartTime") or (_iso(start_ts) if start_ts else None),
        "title":         m.get("question") or ev.get("title"),
    }
    _META_CACHE[slug] = meta
    if len(_META_CACHE) > 64:
        for k in sorted(_META_CACHE)[:-64]:
            _META_CACHE.pop(k, None)
    return meta


async def _fetch_gamma_market_state(slug: str) -> Optional[dict]:
    """Dynamic gamma state (closed flag, outcome prices, acceptingOrders)."""
    try:
        rows = await _http_get_json(f"{POLYMARKET_GAMMA_HOST}/markets",
                                    {"slug": slug})
    except Exception:
        return None
    return rows[0] if rows else None


# ── Strike ("price to beat") discovery ─────────────────────────────────────────
async def get_strike(slug: str) -> Optional[float]:
    """The opening reference price for the window — first Chainlink BTC/USD
    tick at/after the window start. Primary source: Polymarket's own
    crypto-price endpoint (the number the UI shows as 'Price to beat').
    Fallback: our own spot sample captured at the window boundary."""
    if slug in _STRIKE_CACHE:
        return _STRIKE_CACHE[slug]
    start_ts = window_start_from_slug(slug)
    if start_ts is None:
        return None
    try:
        data = await _http_get_json(
            f"{POLYMARKET_SITE_HOST}/api/crypto/crypto-price",
            {"symbol": "BTC", "eventStartTime": _iso(start_ts),
             "variant": "fifteen"}, timeout=8.0)
        open_px = data.get("openPrice")
        if open_px:
            px = float(open_px)
            if px > 0:
                _STRIKE_CACHE[slug] = px
                if len(_STRIKE_CACHE) > 64:
                    for k in list(_STRIKE_CACHE)[:-64]:
                        _STRIKE_CACHE.pop(k, None)
                return px
    except Exception as e:
        log.debug(f"strike fetch (crypto-price) failed for {slug}: {e}")
    # Fallback: boundary spot sample we captured ourselves
    px = _BOUNDARY_SPOT.get(start_ts)
    if px:
        log.warning(f"strike for {slug}: using self-captured boundary spot ${px:,.2f} "
                    f"(crypto-price endpoint unavailable)")
        _STRIKE_CACHE[slug] = px
        return px
    return None


def note_boundary_spot(spot: float) -> None:
    """Called opportunistically with fresh spot prices; keeps the first sample
    seen within each 15-min window as a strike fallback."""
    try:
        if not spot or spot <= 0:
            return
        w = int(time.time()) // WINDOW_S * WINDOW_S
        _BOUNDARY_SPOT.setdefault(w, float(spot))
        if len(_BOUNDARY_SPOT) > 16:
            for k in sorted(_BOUNDARY_SPOT)[:-16]:
                _BOUNDARY_SPOT.pop(k, None)
    except Exception:
        pass


# ── Settlement detection ───────────────────────────────────────────────────────
async def get_result(slug: str) -> Optional[str]:
    """'yes' (Up won), 'no' (Down won), or None if not resolved yet.
    Fast path: Polymarket's crypto-price open/close report (the resolution
    source recipe: close >= open ⇒ Up). Authoritative path: gamma closed +
    outcome prices."""
    if slug in _RESULT_CACHE:
        return _RESULT_CACHE[slug]
    start_ts = window_start_from_slug(slug)
    if start_ts is None:
        return None
    # Don't even try before the window has closed.
    if time.time() < start_ts + WINDOW_S:
        return None
    result: Optional[str] = None
    # 1) crypto-price completed report (fast, same source Polymarket resolves on)
    try:
        data = await _http_get_json(
            f"{POLYMARKET_SITE_HOST}/api/crypto/crypto-price",
            {"symbol": "BTC", "eventStartTime": _iso(start_ts),
             "variant": "fifteen"}, timeout=8.0)
        if data.get("completed") and data.get("openPrice") and \
                data.get("closePrice") is not None:
            result = ("yes" if float(data["closePrice"]) >= float(data["openPrice"])
                      else "no")
    except Exception:
        pass
    # 2) gamma authoritative resolution
    if result is None:
        gm = await _fetch_gamma_market_state(slug)
        if gm and gm.get("closed"):
            try:
                prices = json.loads(gm.get("outcomePrices") or "[]")
                outcomes = json.loads(gm.get("outcomes") or "[]")
                for o, p in zip(outcomes, prices):
                    if float(p) >= 0.99:
                        result = "yes" if str(o).strip().lower() == "up" else "no"
                        break
            except (ValueError, TypeError):
                pass
    if result is not None:
        _RESULT_CACHE[slug] = result
        if len(_RESULT_CACHE) > 64:
            for k in list(_RESULT_CACHE)[:-64]:
                _RESULT_CACHE.pop(k, None)
    return result


# ── Order books / quotes ───────────────────────────────────────────────────────
async def _fetch_book(token_id: str) -> Optional[dict]:
    """Raw CLOB book for one token: {'bids':[{price,size}...],'asks':[...]}"""
    try:
        return await _http_get_json(f"{POLYMARKET_HOST}/book",
                                    {"token_id": token_id}, timeout=8.0)
    except Exception:
        return None


def _best_levels(book: Optional[dict]):
    """(best_bid_cents, best_ask_cents) from a raw CLOB book."""
    if not book:
        return None, None
    def _best(levels, pick_max):
        px = None
        for lvl in levels or []:
            try:
                p = float(lvl.get("price"))
                s = float(lvl.get("size") or 0)
            except (TypeError, ValueError):
                continue
            if s <= 0:
                continue
            if px is None or (p > px if pick_max else p < px):
                px = p
        return px
    bid = _best(book.get("bids"), True)
    ask = _best(book.get("asks"), False)
    return _to_cents(bid), _to_cents(ask)


async def build_market_dict(slug: str, *, need_books: bool = True) -> Optional[dict]:
    """Assemble the market dict in the exchange schema the daemon expects.
    Integer-cent quote fields, floor_strike, close_time, status, result."""
    now = time.time()
    cached = _MARKET_CACHE.get(slug)
    if cached and now - cached[0] < _MARKET_CACHE_TTL_S:
        return cached[1]

    meta = await _fetch_meta(slug)
    if not meta:
        return None
    start_ts = window_start_from_slug(slug)

    strike_task = asyncio.create_task(get_strike(slug))
    gamma_task  = asyncio.create_task(_fetch_gamma_market_state(slug))
    if need_books:
        up_book, down_book = await asyncio.gather(
            _fetch_book(meta["token_id_yes"]), _fetch_book(meta["token_id_no"]))
    else:
        up_book = down_book = None
    strike = await strike_task
    gm     = await gamma_task

    yes_bid, yes_ask = _best_levels(up_book)
    no_bid,  no_ask  = _best_levels(down_book)
    # Complete one side from the other via binary symmetry when a book is thin.
    if yes_ask is None and no_bid is not None:
        yes_ask = 100 - no_bid
    if no_ask is None and yes_bid is not None:
        no_ask = 100 - yes_bid
    if yes_bid is None and no_ask is not None:
        yes_bid = 100 - no_ask
    if no_bid is None and yes_ask is not None:
        no_bid = 100 - yes_ask

    closed = bool(gm.get("closed")) if gm else False
    accepting = bool(gm.get("acceptingOrders")) if gm else True
    expired = start_ts is not None and now >= start_ts + WINDOW_S
    status = "active" if (not closed and not expired and accepting) else "closed"

    result = ""
    if expired or closed:
        r = await get_result(slug)
        if r:
            result = r
            status = "settled"

    last_price = None
    if gm is not None:
        last_price = _to_cents(gm.get("lastTradePrice"))

    close_time = meta.get("end_date")
    if not close_time and start_ts is not None:
        close_time = _iso(start_ts + WINDOW_S)

    market = {
        # Exchange-schema fields consumed by the daemon
        "ticker":        slug,
        "event_ticker":  slug,
        "series_ticker": SERIES_SLUG,
        "title":         meta.get("title"),
        "status":        status,
        "result":        result,
        "close_time":    close_time,
        "open_time":     _iso(start_ts) if start_ts else None,
        "expected_expiration_time": close_time,
        "floor_strike":  strike,
        "yes_bid":       yes_bid, "yes_ask": yes_ask,
        "no_bid":        no_bid,  "no_ask":  no_ask,
        "last_price":    last_price,
        # Polymarket-specific extras (harmless to the daemon)
        "condition_id":  meta["condition_id"],
        "token_id_yes":  meta["token_id_yes"],
        "token_id_no":   meta["token_id_no"],
        "neg_risk":      meta["neg_risk"],
        "tick_size":     meta["tick_size"],
        "min_size":      meta["min_size"],
    }
    # A market without a discovered strike is not tradeable by the engine yet.
    if not strike:
        market["status"] = "no_strike"
    _MARKET_CACHE[slug] = (now, market)
    if len(_MARKET_CACHE) > 32:
        for k in list(_MARKET_CACHE)[:-32]:
            _MARKET_CACHE.pop(k, None)
    return market


async def discover_markets(limit: int = 3) -> list:
    """Current + upcoming 15m BTC windows (Kalshi-shaped market dicts)."""
    now = int(time.time())
    cur = now // WINDOW_S * WINDOW_S
    out = []
    for i in range(limit):
        m = await build_market_dict(slug_for_window_start(cur + i * WINDOW_S))
        if m:
            out.append(m)
    return out


# ── Raw orderbook / trade tape in the legacy shapes ────────────────────────────
async def get_orderbook_raw(slug: str) -> dict:
    """Orderbook in the legacy shape consumed by summarize_orderbook():
    orderbook.yes / orderbook.no = arrays of BIDS [[price_dollars, count], ...]."""
    meta = await _fetch_meta(slug)
    if not meta:
        raise RuntimeError(f"unknown market {slug}")
    up_book, down_book = await asyncio.gather(
        _fetch_book(meta["token_id_yes"]), _fetch_book(meta["token_id_no"]))
    def _bids(book):
        rows = []
        for lvl in (book or {}).get("bids") or []:
            try:
                rows.append([f"{float(lvl['price']):.4f}",
                             f"{float(lvl.get('size') or 0):.2f}"])
            except (TypeError, ValueError, KeyError):
                continue
        return rows
    return {"orderbook_fp": {"yes_dollars": _bids(up_book),
                             "no_dollars":  _bids(down_book)}}


async def get_recent_trades_raw(slug: str, limit: int = 20) -> dict:
    """Recent taker trades in the legacy shape consumed by summarize_trades():
    trades: [{count_fp, taker_outcome_side, yes_price_dollars, created_time}]."""
    meta = await _fetch_meta(slug)
    if not meta:
        raise RuntimeError(f"unknown market {slug}")
    try:
        rows = await _http_get_json(
            f"{POLYMARKET_DATA_HOST}/trades",
            {"market": meta["condition_id"], "limit": limit,
             "takerOnly": "true"}, timeout=8.0)
    except Exception:
        rows = []
    trades = []
    for t in rows or []:
        try:
            outcome = str(t.get("outcome") or "").strip().lower()
            side    = str(t.get("side") or "").upper()
            price   = float(t.get("price") or 0)
            size    = float(t.get("size") or 0)
        except (TypeError, ValueError):
            continue
        if outcome not in ("up", "down") or price <= 0 or size <= 0:
            continue
        # YES-taker == aggressively acquiring Up exposure:
        #   BUY Up   or   SELL Down
        if (outcome == "up" and side == "BUY") or (outcome == "down" and side == "SELL"):
            taker_side = "yes"
        else:
            taker_side = "no"
        yes_price = price if outcome == "up" else (1.0 - price)
        ts = t.get("timestamp")
        try:
            created = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
        except (TypeError, ValueError, OSError):
            created = None
        trades.append({
            "count_fp":            f"{size:.2f}",
            "taker_outcome_side":  taker_side,
            "yes_price_dollars":   f"{yes_price:.4f}",
            "no_price_dollars":    f"{1.0 - yes_price:.4f}",
            "created_time":        created,
        })
    return {"trades": trades}


# ── Balance / positions ────────────────────────────────────────────────────────
async def get_balance_raw() -> dict:
    """Legacy /portfolio/balance shape: cash + portfolio value in CENTS."""
    def _bal_sync():
        from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType
        client = _get_clob_client()
        res = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        return res
    res = await asyncio.to_thread(_bal_sync)
    usdc_micro = float((res or {}).get("balance") or 0)
    cash_cents = round(usdc_micro / 1e6 * 100)
    pf_cents = 0
    addr = get_wallet_address()
    if addr:
        try:
            rows = await _http_get_json(f"{POLYMARKET_DATA_HOST}/positions",
                                        {"user": addr, "sizeThreshold": 0.1},
                                        timeout=10.0)
            for p in rows or []:
                try:
                    pf_cents += round(float(p.get("currentValue") or 0) * 100)
                except (TypeError, ValueError):
                    continue
        except Exception as e:
            log.debug(f"portfolio positions fetch failed: {e}")
    return {"balance": cash_cents, "portfolio_value": pf_cents,
            "balance_dollars": f"{cash_cents / 100:.2f}"}


async def _fetch_positions_data_api(slug: Optional[str] = None) -> list:
    """Raw Data-API positions for our wallet (optionally one market)."""
    addr = get_wallet_address()
    if not addr:
        return []
    params: dict = {"user": addr, "sizeThreshold": 0.1, "limit": 100}
    if slug:
        meta = await _fetch_meta(slug)
        if not meta:
            return []
        params["market"] = meta["condition_id"]
    try:
        rows = await _http_get_json(f"{POLYMARKET_DATA_HOST}/positions",
                                    params, timeout=10.0)
        return rows or []
    except Exception as e:
        log.debug(f"data-api positions failed: {e}")
        return []


async def get_positions_raw(slug: Optional[str] = None) -> dict:
    """Legacy /portfolio/positions shape. Net signed 'position' per market
    (positive = long YES/Up, negative = long NO/Down) and 'market_exposure'
    = total cost basis in cents for the net side."""
    rows = await _fetch_positions_data_api(slug)
    by_slug: dict = {}
    for p in rows:
        s = p.get("slug") or ""
        if slug and s != slug:
            continue
        if not slug and not s.startswith(TICKER_PREFIX):
            continue
        outcome = str(p.get("outcome") or "").strip().lower()
        try:
            size = float(p.get("size") or 0)
            avg  = float(p.get("avgPrice") or 0)
        except (TypeError, ValueError):
            continue
        d = by_slug.setdefault(s, {"up": 0.0, "down": 0.0,
                                   "up_cost": 0.0, "down_cost": 0.0})
        if outcome == "up":
            d["up"] += size;   d["up_cost"] += size * avg
        elif outcome == "down":
            d["down"] += size; d["down_cost"] += size * avg
    positions = []
    for s, d in by_slug.items():
        net = round(d["up"] - d["down"])
        if net == 0:
            continue
        if net > 0:
            avg = (d["up_cost"] / d["up"]) if d["up"] > 0 else 0.0
        else:
            avg = (d["down_cost"] / d["down"]) if d["down"] > 0 else 0.0
        exposure_cents = round(abs(net) * avg * 100)
        positions.append({
            "ticker":           s,
            "position":         net,
            "market_exposure":  exposure_cents,
            "total_traded":     exposure_cents,
        })
        # Seed the net-position tracker from exchange truth.
        _NET_POSITION[s] = net
        _NET_POSITION_SEEDED.add(s)
    return {"market_positions": positions}


async def _net_position_for(slug: str) -> int:
    """Adapter-tracked net YES exposure, seeded from the Data API once."""
    if slug not in _NET_POSITION_SEEDED:
        await get_positions_raw(slug)
        _NET_POSITION_SEEDED.add(slug)
        _NET_POSITION.setdefault(slug, 0)
    return _NET_POSITION.get(slug, 0)


# ── Order placement / cancel / status ──────────────────────────────────────────
def _order_type(tif: str):
    from py_clob_client_v2.clob_types import OrderType
    if (tif or "").lower() in ("immediate_or_cancel", "ioc", "fak"):
        return OrderType.FAK
    return OrderType.GTC


def _round_to_tick(price: float, tick: float) -> float:
    steps = max(1, round(price / tick))
    return min(1.0 - tick, max(tick, round(steps * tick, 4)))


async def place_order_raw(body: dict) -> dict:
    """Translate a legacy V2 order body into Polymarket CLOB order(s).

    body: {ticker, client_order_id, side: bid|ask, count: str, price: yes-leg
           decimal str, time_in_force, [action: sell]}

    Returns the legacy shape: {order_id, status, fill_count,
    average_fill_price (yes-leg), client_order_id}.
    """
    slug   = body["ticker"]
    v2side = (body.get("side") or "").lower()          # bid | ask (YES leg)
    count  = int(round(float(body.get("count") or 0)))
    yes_leg = float(body.get("price") or 0)
    tif    = body.get("time_in_force") or "immediate_or_cancel"
    explicit_sell = (body.get("action") or "").lower() == "sell"
    if count <= 0 or not (0.0 < yes_leg < 1.0) or v2side not in ("bid", "ask"):
        raise ValueError(f"invalid order body: {body}")

    meta = await _fetch_meta(slug)
    if not meta:
        raise RuntimeError(f"cannot resolve Polymarket market for {slug}")
    net = await _net_position_for(slug)

    # Net-exposure translation (mirrors the old exchange's YES-leg netting):
    #   bid  (buy YES exposure):  close Down first, then buy Up
    #   ask  (sell YES exposure): close Up first, then buy Down
    legs = []   # (intent, token_key, outcome, side, token_price, size)
    if v2side == "bid":
        close_qty = min(count, -net) if net < 0 else 0
        if explicit_sell:
            close_qty = min(count, -net) if net < 0 else count  # forced reduce
        if close_qty > 0:
            # Selling Down at (1 - yes_leg) == buying YES exposure at yes_leg
            legs.append(("close", "token_id_no", "down", "SELL",
                         1.0 - yes_leg, close_qty))
        open_qty = count - close_qty
        if open_qty > 0 and not explicit_sell:
            legs.append(("open", "token_id_yes", "up", "BUY", yes_leg, open_qty))
    else:  # ask
        close_qty = min(count, net) if net > 0 else 0
        if explicit_sell:
            close_qty = min(count, net) if net > 0 else count
        if close_qty > 0:
            # Selling Up at yes_leg == selling YES exposure
            legs.append(("close", "token_id_yes", "up", "SELL",
                         yes_leg, close_qty))
        open_qty = count - close_qty
        if open_qty > 0 and not explicit_sell:
            # Buying Down at (1 - yes_leg)
            legs.append(("open", "token_id_no", "down", "BUY",
                         1.0 - yes_leg, open_qty))

    if not legs:
        return {"order_id": None, "status": "canceled", "fill_count": 0,
                "average_fill_price": None,
                "client_order_id": body.get("client_order_id")}

    tick = meta.get("tick_size") or 0.01
    otype = _order_type(tif)

    # Pre-resolved order options save the V2 client a tick-size/neg-risk
    # round trip per submission.
    _TICKS = ("0.1", "0.01", "0.005", "0.0025", "0.001", "0.0001")
    tick_str = next((t for t in _TICKS if abs(float(t) - tick) < 1e-9), None)

    def _submit_sync(token_id, side, price, size):
        from py_clob_client_v2.clob_types import (OrderArgsV2,
                                                  PartialCreateOrderOptions)
        from py_clob_client_v2.order_builder.constants import BUY, SELL
        client = _get_clob_client()
        args = OrderArgsV2(
            token_id=token_id,
            price=_round_to_tick(price, tick),
            size=float(size),
            side=BUY if side == "BUY" else SELL,
        )
        options = (PartialCreateOrderOptions(tick_size=tick_str,
                                             neg_risk=bool(meta.get("neg_risk")))
                   if tick_str else None)
        signed = client.create_order(args, options)
        return client.post_order(signed, otype)

    total_filled = 0.0
    sum_yes_leg_x_fill = 0.0
    last_order_id = None
    last_status = "canceled"
    errors = []

    for intent, token_key, outcome, side, price, size in legs:
        token_id = meta[token_key]
        try:
            res = await asyncio.to_thread(_submit_sync, token_id, side, price, size)
        except Exception as e:
            errors.append(f"{intent}/{outcome}: {e}")
            continue
        if not isinstance(res, dict):
            errors.append(f"{intent}/{outcome}: bad response {res!r}")
            continue
        if res.get("errorMsg"):
            errors.append(f"{intent}/{outcome}: {res['errorMsg']}")
        oid = res.get("orderID") or res.get("orderId")
        status = (res.get("status") or "").lower()
        making = float(res.get("makingAmount") or 0)
        taking = float(res.get("takingAmount") or 0)
        if side == "BUY":
            filled = taking                      # tokens received
            avg_tok = (making / taking) if taking > 0 else price
        else:
            filled = making                      # tokens sold
            avg_tok = (taking / making) if making > 0 else price
        avg_yes = avg_tok if outcome == "up" else (1.0 - avg_tok)
        if filled > 0:
            total_filled += filled
            sum_yes_leg_x_fill += filled * avg_yes
            # Update the net tracker: +YES exposure for bid legs, − for ask.
            delta = round(filled)
            if (outcome == "up" and side == "BUY") or (outcome == "down" and side == "SELL"):
                _NET_POSITION[slug] = _NET_POSITION.get(slug, 0) + delta
            else:
                _NET_POSITION[slug] = _NET_POSITION.get(slug, 0) - delta
        if oid:
            last_order_id = oid
            _ORDER_REGISTRY[oid] = {
                "slug": slug, "token_id": token_id, "outcome": outcome,
                "side": side, "size": float(size),
                "price_token": _round_to_tick(price, tick),
                "order_type": str(otype),
                "filled_at_submit": filled,
            }
            if len(_ORDER_REGISTRY) > 512:
                for k in list(_ORDER_REGISTRY)[:-512]:
                    _ORDER_REGISTRY.pop(k, None)
        if status in ("live", "delayed"):
            last_status = "resting"
        elif status in ("matched",):
            last_status = "executed"

    if total_filled <= 0 and last_order_id is None and errors:
        raise RuntimeError("Polymarket order failed: " + "; ".join(errors))
    if errors:
        log.warning(f"Polymarket order partial errors ({slug}): {'; '.join(errors)}")

    fill_count = int(round(total_filled))
    avg_yes_leg = (sum_yes_leg_x_fill / total_filled) if total_filled > 0 else None
    return {
        "order_id":           last_order_id,
        "client_order_id":    body.get("client_order_id"),
        "status":             last_status,
        "fill_count":         fill_count,
        "fill_count_fp":      f"{total_filled:.2f}",
        "average_fill_price": (f"{avg_yes_leg:.4f}" if avg_yes_leg is not None else None),
    }


async def cancel_order_raw(order_id: str) -> dict:
    """Cancel one order. Legacy semantics: {'_not_found': True} when the order
    is already gone (filled/cancelled) so callers treat it as success."""
    def _cancel_sync():
        from py_clob_client_v2.clob_types import OrderPayload
        return _get_clob_client().cancel_order(OrderPayload(orderID=order_id))
    try:
        res = await asyncio.to_thread(_cancel_sync)
    except Exception as e:
        msg = str(e).lower()
        if "not found" in msg or "404" in msg or "does not exist" in msg:
            return {"_not_found": True}
        raise
    if isinstance(res, dict):
        not_canceled = res.get("not_canceled") or {}
        if order_id in (not_canceled or {}):
            reason = str(not_canceled.get(order_id, "")).lower()
            if "match" in reason or "not found" in reason or "cancel" in reason:
                return {"_not_found": True}
    return res if isinstance(res, dict) else {}


def _map_order_status(raw_status: str) -> str:
    s = (raw_status or "").upper()
    if s in ("LIVE", "DELAYED", "UNMATCHED"):
        return "resting"
    if s in ("CANCELED", "CANCELLED"):
        return "canceled"
    return "executed"   # MATCHED / FILLED / anything terminal


async def get_order_raw(order_id: str) -> Optional[dict]:
    """Legacy single-order shape: status / fill_count_fp / remaining_count_fp /
    average_fill_price (yes-leg)."""
    def _get_sync():
        return _get_clob_client().get_order(order_id)
    try:
        o = await asyncio.to_thread(_get_sync)
    except Exception:
        return None
    if not o:
        return None
    reg = _ORDER_REGISTRY.get(order_id) or {}
    try:
        matched  = float(o.get("size_matched") or 0)
        original = float(o.get("original_size") or reg.get("size") or 0)
    except (TypeError, ValueError):
        matched, original = 0.0, 0.0
    status = _map_order_status(o.get("status"))
    try:
        price_tok = float(o.get("price") or reg.get("price_token") or 0)
    except (TypeError, ValueError):
        price_tok = 0.0
    outcome = reg.get("outcome")
    if outcome is None:
        # Order placed before a restart — registry lost. Resolve the token's
        # outcome by matching asset_id against known market metadata.
        asset_id = o.get("asset_id") or o.get("token_id")
        outcome = "up"
        for meta in _META_CACHE.values():
            if asset_id == meta.get("token_id_no"):
                outcome = "down"
                break
    avg_yes = price_tok if outcome == "up" else (1.0 - price_tok)

    # Track fills on registered resting orders → keep the net tracker honest.
    prev = reg.get("last_seen_matched", reg.get("filled_at_submit", 0.0))
    newly = matched - (prev or 0.0)
    if newly > 0 and reg:
        slug = reg["slug"]
        delta = round(newly)
        if (reg["outcome"] == "up" and reg["side"] == "BUY") or \
           (reg["outcome"] == "down" and reg["side"] == "SELL"):
            _NET_POSITION[slug] = _NET_POSITION.get(slug, 0) + delta
        else:
            _NET_POSITION[slug] = _NET_POSITION.get(slug, 0) - delta
        reg["last_seen_matched"] = matched

    return {
        "order_id":            order_id,
        "status":              status,
        "fill_count_fp":       f"{matched:.2f}",
        "remaining_count_fp":  f"{max(0.0, original - matched):.2f}",
        "average_fill_price":  f"{avg_yes:.4f}" if price_tok else None,
        "ticker":              reg.get("slug"),
    }


async def get_open_orders_raw() -> dict:
    """Legacy /portfolio/orders?status=resting shape."""
    def _open_sync():
        from py_clob_client_v2.clob_types import OpenOrderParams
        return _get_clob_client().get_open_orders(OpenOrderParams(),
                                                  only_first_page=True)
    try:
        rows = await asyncio.to_thread(_open_sync)
    except Exception as e:
        log.debug(f"open orders fetch failed: {e}")
        rows = []
    orders = []
    for o in rows or []:
        oid = o.get("id") or o.get("orderID")
        if not oid:
            continue
        orders.append({"order_id": oid, "status": "resting",
                       "ticker": (_ORDER_REGISTRY.get(oid) or {}).get("slug")})
    return {"orders": orders}


# ── Legacy path router ─────────────────────────────────────────────────────────
# The daemon speaks trade-api/v2 paths; route them to the handlers above.
_MARKET_PATH_RE   = re.compile(r"^/markets/([^/]+)$")
_ORDERBOOK_RE     = re.compile(r"^/markets/([^/]+)/orderbook$")
_ORDER_STATUS_RE  = re.compile(r"^/portfolio/orders/([^/]+)$")
_ORDER_CANCEL_RE  = re.compile(r"^/portfolio/(?:events/)?orders/([^/]+)$")


async def pm_get(path: str, params: dict = {}) -> dict:
    """GET router — legacy paths → Polymarket implementations."""
    params = params or {}
    m = _ORDERBOOK_RE.match(path)
    if m:
        return await get_orderbook_raw(m.group(1))
    if path == "/markets/trades" or path == "/markets/trades/":
        return await get_recent_trades_raw(params.get("ticker") or "",
                                           int(params.get("limit") or 20))
    m = _MARKET_PATH_RE.match(path)
    if m:
        mkt = await build_market_dict(m.group(1))
        if not mkt:
            raise RuntimeError(f"market not found: {m.group(1)}")
        return {"market": mkt}
    if path == "/markets":
        # event_ticker / series_ticker discovery
        ev = params.get("event_ticker")
        if ev:
            mkt = await build_market_dict(ev)
            return {"markets": [mkt] if mkt else []}
        return {"markets": await discover_markets(limit=3)}
    if path == "/portfolio/balance":
        return await get_balance_raw()
    if path == "/portfolio/positions":
        return await get_positions_raw(params.get("ticker"))
    m = _ORDER_STATUS_RE.match(path)
    if m:
        o = await get_order_raw(m.group(1))
        if o is None:
            raise RuntimeError(f"order not found: {m.group(1)}")
        return {"order": o}
    if path == "/portfolio/orders":
        return await get_open_orders_raw()
    raise RuntimeError(f"pm_get: unsupported path {path}")


async def pm_post(path: str, body: dict) -> dict:
    """POST router — order placement."""
    if path in ("/portfolio/events/orders", "/portfolio/orders"):
        return await place_order_raw(body)
    raise RuntimeError(f"pm_post: unsupported path {path}")


async def pm_delete(path: str) -> dict:
    """DELETE router — order cancel."""
    m = _ORDER_CANCEL_RE.match(path)
    if m:
        return await cancel_order_raw(m.group(1))
    raise RuntimeError(f"pm_delete: unsupported path {path}")

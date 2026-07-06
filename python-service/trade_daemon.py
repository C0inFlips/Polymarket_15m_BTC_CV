"""
trade_daemon.py â€” Autonomous Polymarket BTC 15m trading daemon (ChainVector edition)

Runs 24/7. Wakes up for each 15-min btc-updown-15m window (Polymarket's
"Bitcoin Up or Down" series), runs the full Markov signal stack, places a
real order if all gates pass, and logs every decision to
logs/daemon_YYYYMMDD.log.

Signals architecture:
  â€¢ Polymarket      â€” execution venue + its own orderbook/trade tape (via
    the polymarket_client adapter: Gamma discovery, CLOB books/orders,
    Data-API positions â€” all exposed in the legacy exchange schema)
  â€¢ Coinbase/OKX/Deribit index â€” RAW spot price oracle only (fallback chain)
  â€¢ ChainVector     â€” SOLE derived-signals provider: probability engine
    (terminal P(above) with exact close_ts), cross-venue futures momentum
    (lead-lag feed + EV weight + veto), prediction-market quote stability &
    model edge, liquidation cascade risk & heatmap, order flow, whale
    pressure, funding, volatility/regime context (recorded).

Strategy templates (--strategy): baseline | conservative | momentum |
regime | edge â€” presets over the gate stack; every knob remains
individually overridable by its own flag.

Usage:
  python trade_daemon.py               # live trading with baked-in defaults
  python trade_daemon.py --dry-run     # simulate only (no real orders)
  python trade_daemon.py --strategy conservative --bankroll 500
"""

import argparse, asyncio, base64, json, logging, math, os, sys, time, uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import httpx

# Windows consoles default to cp1252 and choke on chars like Î” / Â¢ in our log lines.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# â”€â”€ Env loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MUST run before the local imports below: signal_feeds / terminal_prob /
# cv_lead construct the shared ChainVector client at import time, so
# CHAINVECTOR_API_KEY has to be in os.environ by then.
_env_path = Path(__file__).parent.parent / ".env.local"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, os.path.dirname(__file__))
from run_backtest import (
    fetch_candles_5m, fetch_candles_15m,
    build_markov_history, build_transition_matrix, predict_from_momentum,
    price_change_to_state, gk_vol, compute_hurst,
    MARKOV_MIN_GAP, MIN_PERSIST, KELLY_FRACTION, MAX_TRADE_PCT,
    MAX_ENTRY_PRICE_RM, MAX_ENTRY_PRICE_YES, MAX_ENTRY_PRICE_NO,
    MAKER_FEE_RATE, EMPIRICAL_PRICE_BY_D, BLOCKED_UTC_HOURS,
    MIN_MINUTES_LEFT, MAX_MINUTES_LEFT, MIN_HURST, MAX_VOL_MULT, REF_VOL_15M,
)

# Terminal probability â€” ChainVector Probability Engine (six-estimator
# ensemble on tick-derived 1m bars, exact close_ts TTE). Weighted into
# combined_p via --ev-tp-weight â€” the probability engine IS the EV weight.
from terminal_prob import compute_terminal_prob, save_snapshot

# ChainVector client (sole derived-signals provider) + futures lead-lag feed.
# CVLeadFeed polls /momentum in a background thread; per-venue views expose
# the same get_recent_move()/is_signal_consistent() API the old CME/OKX
# feeds had: binance_futures = primary lead venue, okx = secondary. The
# same poller also provides the momentum scorecard for the EV boost/veto.
from chainvector import ChainVectorClient, get_client as get_cv_client
from cv_lead import CVLeadFeed
# Live ChainVector signal snapshots (replaces the perp + CoinGlass recorder
# feeds). snapshot_new_signals() never raises; rrm_evaluate/latest_perp_mid â€”
# Reversal-Risk Monitor; cv_composite â€” flip-probability composite;
# context_signals â€” recorded regime/volatility/risk-index bundle.
from signal_feeds import (snapshot_new_signals, rrm_evaluate, latest_perp_mid,
                          cv_composite, attach_lead_feed, context_signals)

# Process-wide ChainVector client (shared with terminal_prob/signal_feeds/
# cv_lead via the same singleton â€” one cache, one rate-limit budget).
_cv_client = get_cv_client()
# Polymarket orderbook + recent trade flow â€” record-only signals added
# 2026-05-28. Audit logs include orderbook depth snapshots + trade aggression
# metrics on every poll for retrospective analysis. After 1-2 days of data we
# can correlate features (imbalance, spread, aggression) with WIN/LOSS
# outcomes and promote any predictive ones to gates.
from orderbook_feed import fetch_orderbook, fetch_recent_trades

# Polymarket execution adapter: Gamma discovery + CLOB order flow + Data-API
# positions, exposed through the SAME legacy trade-api/v2 path surface the
# daemon was written against (pm_get/pm_post/pm_delete routers). Auth/config
# via POLYMARKET_PRIVATE_KEY / POLYMARKET_PROXY_ADDRESS env vars.
import polymarket_client as pm_client

# 2026-06-22: entry cushion floor + RRM cushion-gating (set from CLI in
# __main__, read by run_signal's entry gate and the RRM live-exit block).
MIN_ENTRY_CUSHION_PCT = 0.0   # skip entries with |dist_pct| < this (0 = off)
RRM_EXIT_CUSHION_MAX = 0.0    # RRM live-exit only if entry |dist_pct| < this (0 = off)

# 2026-06-25: ITM lock-in veto exemption. Skip the 6s futures-lead veto
# and the Hurst+TP disagreement veto ONLY when already comfortably ITM in our
# direction with very high conviction (false-positive regime).
ITM_LOCKIN_VETO_EXEMPT_ENABLED   = True
ITM_LOCKIN_EXEMPT_CUSHION_PCT    = 0.10   # signed dist% in trade direction
ITM_LOCKIN_EXEMPT_MIN_COMBINED_P = 0.94

# Structured JSONL audit log â€” one row per signal evaluation + trade + settle.
# Use for post-hoc analysis (pandas / jq). Independent of the human daemon log.
from audit_log import (
    AuditLogger,
    build_poll_record,
    build_trade_record,
    build_settlement_record,
)

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# (.env.local is loaded at the very top of this file, before local imports.)
# Execution venue config (hosts, wallet key, signature type) lives entirely in
# polymarket_client; the daemon only needs the series identity.
SERIES_SLUG   = pm_client.SERIES_SLUG      # "btc-updown-15m"
TICKER_PREFIX = pm_client.TICKER_PREFIX    # "btc-updown-15m-"

MAX_DAILY_LOSS   = 50.0   # $ hard stop for the day
MAX_GIVEBACK_X   = 1.5    # stop if peak P&L drops by this Ã— MAX_DAILY_LOSS
MAX_DAILY_TRADES = 100  # 2026-06-08: raised from 48 to accommodate higher-volume
                        # regimes under the strict-HC + no-SL + expanded-cap config.
                        # 06-08 was on track for ~70 trades at 94% WR; the cap was
                        # no longer "tilt protection" but "growth ceiling". Loss cap
                        # (-$150) remains the real circuit breaker.

# â”€â”€ Logging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Data/log dirs resolve through paths.py so the program never requires root:
# BTC15M_DATA_DIR env override -> package dir -> ~/.polymarket_btc_15m_cv -> tmp.
from paths import logs_dir as _logs_dir

_log_dir = _logs_dir()

def _make_logger() -> logging.Logger:
    logger = logging.getLogger("daemon")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S UTC")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    # File logging is best-effort: on read-only / permission-restricted
    # deployments (e.g. non-root user in a root-owned container dir) we
    # degrade to console-only rather than crash at import time.
    try:
        fh = logging.FileHandler(_log_dir / f"daemon_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except (PermissionError, OSError) as e:
        logger.warning(f"File logging unavailable ({e}); console-only. "
                       f"Set BTC15M_DATA_DIR to a writable path to enable.")
    return logger

log = _make_logger()

# â”€â”€ 2026-07-02: POLYMARKET BID-STABILITY ENTRY GATE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# window_id -> list[(epoch_s, yes_ask, no_ask)], fed once per poll from the
# freshly fetched market snapshot. At fire time the gate reconstructs OUR
# side's executable bid (100 - opposite ask) and requires it to be stable or
# rising over the recent lookback: net >= 0 AND within max-fade cents of its
# lookback peak. Blocks the "model still likes it but the market is actively
# bidding it DOWN" reversal entry. Fail-open (< min samples passes) and
# transient (re-polls; a dip that recovers can still enter later).
_BIDSTAB_HIST: dict = {}


def _bidstab_update(window_id, market) -> None:
    try:
        ya, na = market.get("yes_ask"), market.get("no_ask")
        if not ya and not na:
            return
        h = _BIDSTAB_HIST.setdefault(str(window_id), [])
        h.append((time.time(), ya, na))
        if len(h) > 400:
            del h[:-400]
        if len(_BIDSTAB_HIST) > 8:
            for _k in sorted(_BIDSTAB_HIST)[:-8]:
                _BIDSTAB_HIST.pop(_k, None)
    except Exception:
        pass


def _bidstab_check(window_id, side, lookback_s, max_fade_c, min_samples):
    """Return None (pass) or a veto-reason string."""
    now = time.time()
    path = []
    for _ts, _ya, _na in _BIDSTAB_HIST.get(str(window_id), []):
        if now - _ts > lookback_s:
            continue
        _opp = _na if side == "YES" else _ya
        if _opp:
            path.append((_ts, 100 - int(_opp)))
    if len(path) < min_samples:
        return None   # warm-up / thin data: fail-open
    first, last = path[0][1], path[-1][1]
    peak = max(b for _, b in path)
    net = last - first
    # 2026-07-03: relaxed 0 -> -2 for BTC. 30d backtest (695 trades): tolerating
    # a 1-2c dip halves clipped winners while keeping ~95% of the avoided losses.
    if net < -2:
        return (f"bid FALLING: {side} bid {first}c -> {last}c "
                f"({net:+d}c over {now - path[0][0]:.0f}s)")
    if peak - last > max_fade_c:
        return (f"bid OFF PEAK: {side} bid {peak}c peak -> {last}c now "
                f"(fade {peak - last}c > {max_fade_c}c)")
    return None


async def _bidstab_burst(side, n_extra, interval_s, window_id):
    """2026-07-02 v2: REAL-TIME micro-stability confirm at the entry moment.

    The poll-history check sees 10-15s-cadence samples; this sees the last
    ~5-10 SECONDS. Re-sample the live book n_extra more times, interval_s
    apart. ANY downtick of our side's executable bid = the market is wobbling
    / hasn't priced the move with certainty -> veto this attempt (transient;
    re-polls ~10s later and can enter once the bid holds). Net-down across
    the burst also vetoes. Fail-open on fetch errors/missing quotes.
    Latency cost when clean: n_extra * interval_s (~6s default)."""
    try:
        seq = []
        for i in range(n_extra + 1):
            if i:
                await asyncio.sleep(interval_s)
            m = await fetch_market()
            if not m:
                return None
            opp = m.get("no_ask") if side == "YES" else m.get("yes_ask")
            if not opp:
                return None
            seq.append(100 - int(opp))
            _bidstab_update(window_id, m)
            if len(seq) >= 2 and seq[-1] < seq[-2]:
                return (f"bid WOBBLE (burst): {side} bid ticked "
                        f"{seq[-2]}c -> {seq[-1]}c within "
                        f"{interval_s:.0f}s (path {'-'.join(map(str, seq))})")
        if seq[-1] < seq[0]:
            return (f"bid NET-DOWN (burst): {side} bid {seq[0]}c -> {seq[-1]}c "
                    f"over {n_extra * interval_s:.0f}s")
        return None
    except Exception:
        return None


# â”€â”€ Transport â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The daemon speaks the legacy trade-api/v2 path surface it was battle-tested
# against; polymarket_client routes those paths to Gamma/CLOB/Data-API calls
# and returns responses in the same shapes (integer-cent quote fields etc).
async def _kget(path: str, params: dict = {}) -> dict:
    return await pm_client.pm_get(path, params)

async def _kpost(path: str, body: dict) -> dict:
    return await pm_client.pm_post(path, body)

async def _kdelete(path: str) -> dict:
    """DELETE â€” used to cancel resting orders. Returns {'_not_found': True}
    when the order is already gone/filled so callers can treat 'order no
    longer cancelable' as success."""
    return await pm_client.pm_delete(path)

async def _korder_status(order_id: str) -> Optional[dict]:
    """Fetch a single order's current state. None on error/not-found."""
    try:
        j = await _kget(f"/portfolio/orders/{order_id}")
        return j.get("order", j)
    except Exception:
        return None

def _order_fill_count(order: dict) -> int:
    """Contracts filled on an order. Per Polymarket GetOrders spec the canonical
    field is `fill_count_fp` (fixed-point STRING, e.g. '10.00'); integer
    `fill_count`/`filled_count` are legacy. Try fp first, fall back to legacy."""
    if not order:
        return 0
    for k in ("fill_count_fp", "fill_count", "filled_count"):
        v = order.get(k)
        if v is not None:
            try:
                return int(round(float(v)))
            except (TypeError, ValueError):
                continue
    return 0

def _order_remaining_count(order: dict) -> Optional[int]:
    """Remaining (unfilled) contracts via `remaining_count_fp`; None if absent."""
    if not order:
        return None
    v = order.get("remaining_count_fp")
    if v is None:
        return None
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None

# Polymarket order status enum (GetOrders spec): only these three values exist.
_ORDER_STATUS_DONE = ("canceled", "executed")   # no longer resting


async def _place_resting_tp_for_trade(trade: dict, *, take_profit_cents: int,
                                      high_price_tp_enabled: bool,
                                      high_price_tp_min_cents: int,
                                      high_price_tp_target_cents: int) -> bool:
    """Place a resting TP for a RECOVERED position (on restart), based on its
    blended entry (trade['limit_price']) and side. Stores resting_tp_id /
    resting_tp_price / resting_tp_filled on `trade`. Returns True if placed.
    Never raises. NOT perp-gated â€” the entry perp is unknown for a recovered
    position and the position already exists, so we just protect it. Mirrors the
    inline target logic: ceiling for high-price entries, else entry+N."""
    entry = int(trade.get("limit_price") or 0)
    if entry <= 0:
        return False
    if high_price_tp_enabled and entry >= high_price_tp_min_cents:
        tp_price = high_price_tp_target_cents
    else:
        tp_price = entry + take_profit_cents
    if tp_price > 99 or tp_price <= entry:
        return False
    qty = int(trade.get("contracts") or 0)
    if qty <= 0:
        return False
    side = trade["side"]
    try:
        # 2026-06-18: V2 migration (legacy POST /portfolio/orders deprecated â†’ 410).
        # V2 quotes the YES leg: SELL = ask (YES pos) / bid (NO pos) at YES-leg price.
        _yes_leg = (tp_price / 100.0) if side == "YES" else (1.0 - tp_price / 100.0)
        body = {
            "ticker": trade["ticker"],
            "client_order_id": uuid.uuid4().hex,
            "side": "ask" if side == "YES" else "bid",
            "count": str(int(qty)),
            "price": f"{_yes_leg:.4f}",
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "taker_at_cross",
        }
        res = await _kpost("/portfolio/events/orders", body)
        oid = (res.get("order") or res).get("order_id") if isinstance(res, dict) else None
        if oid:
            trade["resting_tp_id"] = oid
            trade["resting_tp_price"] = tp_price
            trade["resting_tp_filled"] = trade.get("resting_tp_filled", 0)
            log.warning(f"  [RESTING-TP] recovered {trade['ticker']}: placed sell "
                        f"{qty}c {side} @ {tp_price}\u00a2 (entry {entry}\u00a2) "
                        f"id={oid[:8]}")
            return True
    except Exception as e:
        log.warning(f"  [RESTING-TP] recovered re-place failed (non-fatal): {e!r}")
    return False


def _reconcile_recovered_tp_fill(trade: dict, session: "Session", audit,
                                 filled_c: int, window_id: str) -> None:
    """Account a resting-TP fill on a RECOVERED position: record PnL on the
    filled portion and reduce the position (so settlement doesn't double-count).
    filled_c = NEWLY filled contracts."""
    if filled_c <= 0:
        return
    entry = int(trade.get("limit_price") or 0)
    tp_price = int(trade.get("resting_tp_price") or entry)
    if tp_price <= entry:
        return
    p_d = tp_price / 100.0
    fee = MAKER_FEE_RATE * p_d * (1 - p_d)
    gain_per = (tp_price - entry) / 100.0 - fee
    realized = gain_per * filled_c
    session.record(realized)
    trade["resting_tp_filled"] = trade.get("resting_tp_filled", 0) + filled_c
    old_n = trade["contracts"]
    ratio = max(0, (old_n - filled_c)) / old_n if old_n else 0
    trade["contracts"] = max(0, old_n - filled_c)
    trade["cost"]    = round(trade["cost"] * ratio, 2)
    trade["net_win"] = round(trade["net_win"] * ratio, 2)
    log.warning(f"  [RESTING-TP] recovered FILL {filled_c}c @ {tp_price}\u00a2 "
                f"realized ${realized:+.2f}; {trade['contracts']}c remaining")
    try:
        audit.write({
            "type": "RESTING_TP_FILL", "window_id": window_id,
            "ticker": trade["ticker"], "side": trade["side"],
            "filled_contracts": filled_c, "tp_price_cents": tp_price,
            "realized_pnl": round(realized, 4),
            "remaining_contracts": trade["contracts"],
            "recovered": True,
        })
    except Exception:
        pass


async def _cancel_order_confirmed(order_id: str) -> bool:
    """Cancel an order and CONFIRM it's no longer resting. Returns True only on
    positive evidence (DELETE 404, status canceled/executed, or remaining==0).
    Used before a market sell so a still-live resting order can't double-sell."""
    for _ in range(3):
        confirmed = False
        try:
            res = await _kdelete(f"/portfolio/events/orders/{order_id}")
            if isinstance(res, dict) and res.get("_not_found"):
                confirmed = True
        except Exception:
            pass
        try:
            st = await _korder_status(order_id)
            if st is not None:
                if ((st.get("status") or "").lower() in _ORDER_STATUS_DONE
                        or _order_remaining_count(st) == 0):
                    confirmed = True
        except Exception:
            pass
        if confirmed:
            return True
        await asyncio.sleep(0.5)
    return False


async def monitor_recovered_position(session: "Session", audit, trade: dict,
        window_id: str, *, sl_loss_cents: int, sl_trigger_mode: str,
        sl_poll_interval_s: float, sl_disable_late_mins: float,
        sl_aggressive_sell: bool, take_profit_cents: int) -> None:
    """Self-contained STOP-LOSS + resting-TP monitor for a RECOVERED position
    (restart with an open position). The inline fresh-position monitor can't be
    reused here (it's tied to the fresh-fill context), so this mirrors its CORE
    behavior standalone: poll every sl_poll_interval_s, exit on SL, track the
    re-placed resting TP, and hand off to settlement near close. It deliberately
    does NOT replicate entry-time extras (hedge / fade-bounce / smart-flip /
    RRM / patient-topup) â€” a recovered position just needs SL + TP protection.

    Blocks until the position exits (SL or full TP) or the window nears
    settlement (then returns; the main settlement loop records the settle).
    Recovered positions have no stored tier, so the base sl_loss_cents is used
    (no tier-tight SL) and there is no grace period (the position is already
    established). Never raises out."""
    ticker      = trade["ticker"]
    pos_side    = trade["side"]
    entry_cents = int(trade["limit_price"])
    try:
        m = (await _kget(f"/markets/{ticker}")).get("market", {})
        close_dt = datetime.fromisoformat(
            (m.get("close_time") or "").replace("Z", "+00:00"))
    except Exception:
        log.warning(f"  [RECOVERED-SL] {ticker}: no close_time â€” skipping monitor")
        return

    entry_cost = entry_cents / 100.0
    log.warning(f"  [RECOVERED-SL] monitoring {trade['contracts']}c {pos_side} "
                f"{ticker} entry={entry_cents}\u00a2 stop=-{sl_loss_cents}\u00a2 "
                f"({sl_trigger_mode}), poll {sl_poll_interval_s:.0f}s")

    while True:
        now = datetime.now(timezone.utc)
        mins_remaining = (close_dt - now).total_seconds() / 60.0
        if mins_remaining < sl_disable_late_mins:
            if trade.get("resting_tp_id"):
                await _cancel_order_confirmed(trade["resting_tp_id"])
                trade["resting_tp_id"] = None
            log.info(f"  [RECOVERED-SL] {ticker}: {mins_remaining:.1f}min left â€” "
                     f"handing off to settlement.")
            return
        if window_id not in session.pending:
            return

        # Track the re-placed resting TP (fills reduce the position).
        if trade.get("resting_tp_id"):
            try:
                _rst = await _korder_status(trade["resting_tp_id"])
                if _rst:
                    _new = max(0, _order_fill_count(_rst)
                               - trade.get("resting_tp_filled", 0))
                    if _new > 0:
                        _reconcile_recovered_tp_fill(trade, session, audit, _new, window_id)
                    if (trade["contracts"] <= 0
                            or (_rst.get("status") or "").lower() == "executed"
                            or _order_remaining_count(_rst) == 0):
                        log.warning(f"  [RECOVERED-SL] {ticker}: TP fully filled â€” closed.")
                        trade["resting_tp_id"] = None
                        session.pending.pop(window_id, None)
                        return
            except Exception:
                pass

        # Quote + SL trigger.
        try:
            fresh = (await _kget(f"/markets/{ticker}")).get("market", {})
            _normalize_market(fresh)
        except Exception:
            await asyncio.sleep(sl_poll_interval_s)
            continue
        yes_bid = int(fresh.get("yes_bid") or 0); no_bid = int(fresh.get("no_bid") or 0)
        yes_ask = int(fresh.get("yes_ask") or 0); no_ask = int(fresh.get("no_ask") or 0)
        last_price = int(fresh.get("last_price") or 0)
        current_sell_price = yes_bid if pos_side == "YES" else no_bid
        if current_sell_price <= 0:
            await asyncio.sleep(sl_poll_interval_s)
            continue
        if sl_trigger_mode == "mid":
            if pos_side == "YES" and yes_ask > 0:
                trigger_price = (yes_bid + yes_ask) // 2
            elif pos_side == "NO" and no_ask > 0:
                trigger_price = (no_bid + no_ask) // 2
            else:
                trigger_price = current_sell_price
        elif sl_trigger_mode == "last":
            trigger_price = ((last_price if pos_side == "YES" else 100 - last_price)
                             if last_price > 0 else current_sell_price)
        else:
            trigger_price = current_sell_price
        loss_cents = entry_cents - trigger_price
        if loss_cents < sl_loss_cents:
            await asyncio.sleep(sl_poll_interval_s)
            continue

        # â”€â”€ STOP-LOSS HIT â”€â”€ cancel TP (confirmed) before selling.
        if trade.get("resting_tp_id"):
            if not await _cancel_order_confirmed(trade["resting_tp_id"]):
                log.error(f"  [RECOVERED-SL] {ticker}: could not confirm TP cancel â€” "
                          f"holding sell to avoid double-sell; retry next poll")
                await asyncio.sleep(sl_poll_interval_s)
                continue
            # Reconcile any fills that happened before cancel.
            try:
                _rst = await _korder_status(trade["resting_tp_id"])
                if _rst:
                    _new = max(0, _order_fill_count(_rst)
                               - trade.get("resting_tp_filled", 0))
                    if _new > 0:
                        _reconcile_recovered_tp_fill(trade, session, audit, _new, window_id)
            except Exception:
                pass
            trade["resting_tp_id"] = None
            if trade["contracts"] <= 0:
                session.pending.pop(window_id, None)
                return

        actual = await get_venue_position_count(ticker)
        if actual == 0:
            log.warning(f"  [RECOVERED-SL] {ticker}: Polymarket shows 0 contracts â€” "
                        f"removing stale position.")
            session.pending.pop(window_id, None)
            return
        sell_count = trade["contracts"] if actual is None else min(trade["contracts"], actual)
        if sell_count <= 0:
            session.pending.pop(window_id, None)
            return
        sell_side = "ask" if pos_side == "YES" else "bid"
        sell_yes_leg = ((0.01 if pos_side == "YES" else 0.99) if sl_aggressive_sell
                        else (current_sell_price / 100 if pos_side == "YES"
                              else 1 - current_sell_price / 100))
        try:
            sell_body = {
                "ticker": ticker, "client_order_id": uuid.uuid4().hex,
                "side": sell_side, "count": str(sell_count),
                "price": f"{sell_yes_leg:.4f}",
                "time_in_force": "immediate_or_cancel",
                "self_trade_prevention_type": "taker_at_cross",
            }
            sell_result = await _kpost("/portfolio/events/orders", sell_body)
            sell_filled = int(float(sell_result.get("fill_count") or 0))
        except Exception as e:
            log.error(f"  [RECOVERED-SL] {ticker}: sell error {e}; retry")
            await asyncio.sleep(sl_poll_interval_s)
            continue
        if sell_filled <= 0:
            await asyncio.sleep(sl_poll_interval_s)
            continue
        sell_avg_s = sell_result.get("average_fill_price")
        fallback = (current_sell_price / 100 if pos_side == "YES"
                    else 1 - current_sell_price / 100)
        try:
            sell_avg_yes_leg = float(sell_avg_s) if sell_avg_s else fallback
        except (TypeError, ValueError):
            sell_avg_yes_leg = fallback
        proceeds = (sell_avg_yes_leg if pos_side == "YES" else 1 - sell_avg_yes_leg)
        realized = (proceeds - entry_cost) * sell_filled - (2 * MAKER_FEE_RATE * 0.5) * sell_filled
        session.record(realized)
        actual_sell_cents = (round(sell_avg_yes_leg * 100) if pos_side == "YES"
                             else round((1 - sell_avg_yes_leg) * 100))
        log.warning(f"  [RECOVERED-SL] SL-EXIT {ticker} {sell_filled}c {pos_side} "
                    f"@ {actual_sell_cents}\u00a2 (entry {entry_cents}\u00a2) "
                    f"realized ${realized:+.2f}")
        try:
            audit.write({
                "type": "RECOVERED_SL_EXIT", "window_id": window_id,
                "ticker": ticker, "side": pos_side,
                "sold_contracts": sell_filled, "sell_price_cents": actual_sell_cents,
                "entry_cents": entry_cents, "realized_pnl": round(realized, 4),
            })
        except Exception:
            pass
        # Reduce / clear the position.
        trade["contracts"] = max(0, trade["contracts"] - sell_filled)
        if trade["contracts"] <= 0:
            session.pending.pop(window_id, None)
            return
        # Partial fill â€” scale cost/net_win and keep monitoring the remainder.
        ratio = trade["contracts"] / (trade["contracts"] + sell_filled)
        trade["cost"]    = round(trade["cost"] * ratio, 2)
        trade["net_win"] = round(trade["net_win"] * ratio, 2)
        await asyncio.sleep(sl_poll_interval_s)

# â”€â”€ Timing helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _et_offset() -> int:
    now = datetime.now(timezone.utc)
    yr  = now.year
    mar1 = datetime(yr, 3, 1, tzinfo=timezone.utc)
    edt_start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)
    nov1 = datetime(yr, 11, 1, tzinfo=timezone.utc)
    est_start = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    return -4 if edt_start <= now < est_start else -5

def next_window_close() -> datetime:
    """UTC time of the next btc-updown-15m window close (15-min ET boundary)."""
    off = _et_offset()
    now = datetime.now(timezone.utc)
    et  = now + timedelta(hours=off)
    nxt = (et.minute // 15 + 1) * 15
    et  = et.replace(second=0, microsecond=0)
    if nxt >= 60:
        et = et.replace(minute=0) + timedelta(hours=1)
    else:
        et = et.replace(minute=nxt)
    return et - timedelta(hours=off)

def fmt(secs: float) -> str:
    m, s = divmod(int(abs(secs)), 60)
    return f"{m}m{s:02d}s"

# â”€â”€ Session state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class Session:
    def __init__(self, bankroll: float,
                 rolling_wr_enabled: bool = False,
                 rolling_wr_window: int = 5,
                 rolling_wr_threshold: float = 0.40,
                 rolling_wr_timeout_mins: float = 120.0,
                 rolling_wr_defensive_tiers: tuple = ("standard", "strong"),
                 adaptive_br_enabled: bool = False,
                 adaptive_br_reduced_frac: float = 0.15,
                 adaptive_br_loss_trigger_usd: float = 300.0,
                 adaptive_br_wr_trigger: float = 0.50,
                 adaptive_br_wr_window_h: float = 3.0,
                 adaptive_br_wr_min_trades: int = 5,
                 adaptive_br_recover_wr: float = 0.75,
                 adaptive_br_recover_window: int = 6,
                 adaptive_br_recover_min_wins: int = 3):
        self.bankroll      = bankroll
        self.daily_pnl     = 0.0
        self.daily_trades  = 0
        self.peak_pnl      = 0.0
        self.traded        : set[str] = set()
        self.pending       : dict     = {}
        self.fade_bounce_traded : set[str] = set()
        self._date         = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # â”€â”€ 2026-06-11: ADAPTIVE BANKROLL (risk throttle) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # On a loss cluster (drawdown-since-reference >= trigger) or a weak
        # rolling 3h win rate, shrink the effective bankroll to reduced_frac.
        # Keep trading ALL tiers (live data keeps flowing); recover to 100%
        # on demonstrated win-rate recovery. In-memory only â€” a manual
        # restart resets to NORMAL by design.
        self.adaptive_br_enabled         = adaptive_br_enabled
        self.adaptive_br_reduced_frac    = adaptive_br_reduced_frac
        self.adaptive_br_loss_trigger    = adaptive_br_loss_trigger_usd
        self.adaptive_br_wr_trigger      = adaptive_br_wr_trigger
        self.adaptive_br_wr_window_h     = adaptive_br_wr_window_h
        self.adaptive_br_wr_min_trades   = adaptive_br_wr_min_trades
        self.adaptive_br_recover_wr      = adaptive_br_recover_wr
        self.adaptive_br_recover_window  = adaptive_br_recover_window
        self.adaptive_br_recover_min_wins = adaptive_br_recover_min_wins
        self.adaptive_reduced            = False          # current state
        self.adaptive_ref_peak           = 0.0            # peak daily_pnl since reference
        self.adaptive_wins_since_reduced = 0
        self.adaptive_outcomes_3h: list  = []             # (utc_ts, won) for WR triggers
        self.adaptive_recent: list       = []             # last N outcomes (recovery check)
        self.adaptive_entered_at = None
        self._audit = None                                # set by main_loop for transition records

        # â”€â”€ 2026-06-09: Rolling-WR adaptive regime throttle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # When recent win rate drops below threshold, enter DEFENSIVE MODE
        # which blocks specified tiers (default: standard + strong). Exits
        # on first win OR after timeout. Resets at restart (in-memory only).
        # Idea: today's regime can't always be distinguished by signal
        # features alone, but losses CLUSTER in time. Adaptive throttling
        # via observed outcomes avoids over-fitting to signal averages.
        from collections import deque
        self.rolling_wr_enabled       = rolling_wr_enabled
        self.rolling_wr_window        = rolling_wr_window
        self.rolling_wr_threshold     = rolling_wr_threshold
        self.rolling_wr_timeout_mins  = rolling_wr_timeout_mins
        self.rolling_wr_defensive_tiers = set(rolling_wr_defensive_tiers)
        self.recent_outcomes          = deque(maxlen=rolling_wr_window)
        self.defensive_mode           = False
        self.defensive_entered_at: Optional[datetime] = None

    def new_day_check(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._date:
            log.info(f"â”€â”€ New day {today} â€” resetting P&L counters â”€â”€")
            self.daily_pnl    = 0.0
            self.daily_trades = 0
            self.peak_pnl     = 0.0
            self._date        = today
            # Adaptive bankroll: daily_pnl resets, so the drawdown reference
            # must reset with it. REDUCED state intentionally persists across
            # the UTC rollover (a regime doesn't respect midnight); only a
            # WR recovery or manual restart returns sizing to 100%.
            self.adaptive_ref_peak = 0.0

    def limit_hit(self) -> Optional[str]:
        # 2026-06-09: raised hard upper from $150 to $400 to scale with growing
        # bankroll. At $7K bankroll, 5% = $350; old $150 cap was 2.1% of
        # bankroll, too tight. New ceiling: $400 (5.7% at $7K, 4.0% at $10K).
        # 2026-06-11: when the adaptive bankroll throttle is enabled, it brakes
        # FIRST (shrinks sizing at -$300 drawdown), so the hard stop moves out
        # to $700 and becomes the catastrophe floor behind the throttle.
        # 2026-07-02: recalibrated for the $800 max-trade / $470-$535 per-trade
        # dollar-stop config. The old $400 base meant giveback tripped at $600
        # (1.5x) â€” barely more than ONE stopped loss â€” and paused an 86%-WR
        # session after two morning stops. New base targets TWO worst-case
        # stops (2 x $535 ~ $1,070) for the daily limit, so giveback (1.5x)
        # trips at ~$1,620 ~ THREE stops. Distinguishes "bad-but-in-design
        # variance" from "something is actually broken".
        if self.adaptive_br_enabled:
            max_loss = max(50.0, min(1400.0, self.bankroll * 0.15))
        else:
            max_loss = max(50.0, min(1100.0, self.bankroll * 0.12))
        if self.daily_pnl <= -max_loss:
            return f"daily loss limit (${self.daily_pnl:.2f} / -${max_loss:.0f})"
        giveback = self.peak_pnl - self.daily_pnl if self.peak_pnl > 0 else 0
        if giveback >= max_loss * MAX_GIVEBACK_X:
            return f"session giveback limit (${giveback:.2f} from peak ${self.peak_pnl:.2f})"
        if self.daily_trades >= MAX_DAILY_TRADES:
            return f"daily trade cap ({MAX_DAILY_TRADES})"
        return None

    def record(self, pnl: float):
        self.daily_pnl   += pnl
        self.daily_trades += 1
        self.bankroll     += pnl
        if self.daily_pnl > self.peak_pnl:
            self.peak_pnl = self.daily_pnl
        # â”€â”€ 2026-06-09: rolling-WR throttle outcome tracking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.rolling_wr_enabled:
            won = pnl > 0
            self.recent_outcomes.append(won)
            self._update_defensive_state(triggering_outcome_was_win=won)
        # â”€â”€ 2026-06-11: adaptive bankroll state update â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.adaptive_br_enabled:
            self._adaptive_update(pnl)

    # â”€â”€ Adaptive bankroll â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def effective_bankroll(self) -> float:
        """Bankroll used for position sizing. Shrinks in REDUCED mode."""
        if self.adaptive_br_enabled and self.adaptive_reduced:
            return self.bankroll * self.adaptive_br_reduced_frac
        return self.bankroll

    def _adaptive_audit(self, event: str, reason: str):
        log_fn = log.warning if event == "REDUCED" else log.info
        log_fn(f"[ADAPTIVE-BR] {event} â€” {reason}")
        if self._audit is not None:
            try:
                self._audit.write({
                    "type": "ADAPTIVE_BR", "event": event, "reason": reason,
                    "daily_pnl": round(self.daily_pnl, 2),
                    "ref_peak": round(self.adaptive_ref_peak, 2),
                    "bankroll": round(self.bankroll, 2),
                    "effective_bankroll": round(self.effective_bankroll(), 2),
                })
            except Exception:
                pass

    def _adaptive_update(self, pnl: float):
        now = datetime.now(timezone.utc)
        won = pnl > 0
        # maintain rolling windows
        self.adaptive_outcomes_3h.append((now, won))
        cutoff = now - timedelta(hours=self.adaptive_br_wr_window_h)
        self.adaptive_outcomes_3h = [(t, w) for (t, w) in self.adaptive_outcomes_3h
                                     if t >= cutoff]
        self.adaptive_recent.append(won)
        if len(self.adaptive_recent) > self.adaptive_br_recover_window:
            self.adaptive_recent.pop(0)

        if not self.adaptive_reduced:
            # track peak since reference for drawdown trigger
            if self.daily_pnl > self.adaptive_ref_peak:
                self.adaptive_ref_peak = self.daily_pnl
            drawdown = self.adaptive_ref_peak - self.daily_pnl
            wr_n = len(self.adaptive_outcomes_3h)
            wr   = (sum(1 for _, w in self.adaptive_outcomes_3h if w) / wr_n) if wr_n else 1.0
            if drawdown >= self.adaptive_br_loss_trigger:
                self.adaptive_reduced = True
                self.adaptive_entered_at = now
                self.adaptive_wins_since_reduced = 0
                self._adaptive_audit(
                    "REDUCED",
                    f"drawdown ${drawdown:.0f} >= ${self.adaptive_br_loss_trigger:.0f} "
                    f"(peak ${self.adaptive_ref_peak:.0f} -> ${self.daily_pnl:.0f}). "
                    f"Sizing at {self.adaptive_br_reduced_frac*100:.0f}% bankroll until "
                    f"WR recovers ({self.adaptive_br_recover_min_wins}+ wins and "
                    f">={self.adaptive_br_recover_wr*100:.0f}% of last "
                    f"{self.adaptive_br_recover_window}).")
            elif (wr_n >= self.adaptive_br_wr_min_trades
                    and wr < self.adaptive_br_wr_trigger):
                self.adaptive_reduced = True
                self.adaptive_entered_at = now
                self.adaptive_wins_since_reduced = 0
                self._adaptive_audit(
                    "REDUCED",
                    f"{self.adaptive_br_wr_window_h:.0f}h WR {wr*100:.0f}% "
                    f"({sum(1 for _, w in self.adaptive_outcomes_3h if w)}/{wr_n}) < "
                    f"{self.adaptive_br_wr_trigger*100:.0f}% floor. "
                    f"Sizing at {self.adaptive_br_reduced_frac*100:.0f}% bankroll.")
        else:
            if won:
                self.adaptive_wins_since_reduced += 1
            recent_n = len(self.adaptive_recent)
            recent_wr = (sum(self.adaptive_recent) / recent_n) if recent_n else 0.0
            if (self.adaptive_wins_since_reduced >= self.adaptive_br_recover_min_wins
                    and recent_n >= self.adaptive_br_recover_window
                    and recent_wr >= self.adaptive_br_recover_wr):
                mins = ((now - self.adaptive_entered_at).total_seconds() / 60
                        if self.adaptive_entered_at else 0)
                self.adaptive_reduced = False
                self.adaptive_ref_peak = self.daily_pnl   # re-arm reference here
                self._adaptive_audit(
                    "RECOVERED",
                    f"{self.adaptive_wins_since_reduced} wins in reduced mode, "
                    f"last-{recent_n} WR {recent_wr*100:.0f}% >= "
                    f"{self.adaptive_br_recover_wr*100:.0f}% "
                    f"(reduced for {mins:.0f} min). Back to 100% bankroll; "
                    f"drawdown reference reset to ${self.daily_pnl:.0f}.")

    def _rolling_win_rate(self) -> float:
        if not self.recent_outcomes:
            return 1.0
        return sum(self.recent_outcomes) / len(self.recent_outcomes)

    def _update_defensive_state(self, triggering_outcome_was_win: bool = False):
        """Manage entry/exit of defensive mode based on rolling WR."""
        # Exit conditions: a win, or timeout
        if self.defensive_mode:
            if triggering_outcome_was_win:
                age_s = (datetime.now(timezone.utc) - self.defensive_entered_at).total_seconds() if self.defensive_entered_at else 0
                log.info(
                    f"[REGIME] Exiting DEFENSIVE MODE â€” first win after entry "
                    f"(was in defensive {age_s/60:.0f} min). "
                    f"Rolling WR: {self._rolling_win_rate()*100:.1f}% "
                    f"({sum(self.recent_outcomes)}/{len(self.recent_outcomes)})"
                )
                self.defensive_mode = False
                self.defensive_entered_at = None
                return
            return
        # Entry conditions: rolling WR has dropped below threshold and we have
        # a full window of observations
        if (len(self.recent_outcomes) >= self.rolling_wr_window
                and self._rolling_win_rate() <= self.rolling_wr_threshold):
            self.defensive_mode = True
            self.defensive_entered_at = datetime.now(timezone.utc)
            log.warning(
                f"[REGIME] Entering DEFENSIVE MODE \u2014 rolling WR "
                f"{self._rolling_win_rate()*100:.1f}% "
                f"({sum(self.recent_outcomes)}/{len(self.recent_outcomes)}) "
                f"\u2264 threshold {self.rolling_wr_threshold*100:.0f}%. "
                f"Blocking tiers: {sorted(self.rolling_wr_defensive_tiers)}. "
                f"Will exit on next win or after {self.rolling_wr_timeout_mins:.0f} min."
            )

    def check_defensive_timeout(self):
        """Called from main loop. Exits defensive mode if timeout reached."""
        if (self.defensive_mode and self.defensive_entered_at
                and self.rolling_wr_timeout_mins > 0):
            age_s = (datetime.now(timezone.utc) - self.defensive_entered_at).total_seconds()
            if age_s >= self.rolling_wr_timeout_mins * 60:
                log.info(
                    f"[REGIME] Exiting DEFENSIVE MODE \u2014 timeout reached "
                    f"({self.rolling_wr_timeout_mins:.0f} min). "
                    f"Rolling WR at exit: {self._rolling_win_rate()*100:.1f}% "
                    f"({sum(self.recent_outcomes)}/{len(self.recent_outcomes)})"
                )
                self.defensive_mode = False
                self.defensive_entered_at = None

    def is_tier_blocked_defensive(self, tier: str) -> bool:
        """Return True if `tier` should be skipped due to active defensive mode."""
        return self.defensive_mode and tier in self.rolling_wr_defensive_tiers

# â”€â”€ Market + signal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _normalize_market(m: dict) -> dict:
    """Backfill legacy integer-cent fields (yes_ask/no_ask/yes_bid/no_bid) from the
    current Polymarket schema's *_dollars strings. Mutates and returns the input dict."""
    def _to_cents(v):
        if v is None:
            return None
        try:
            return round(float(v) * 100)
        except (TypeError, ValueError):
            return None
    for legacy, dollar in (("yes_ask", "yes_ask_dollars"), ("no_ask", "no_ask_dollars"),
                           ("yes_bid", "yes_bid_dollars"), ("no_bid", "no_bid_dollars"),
                           ("last_price", "last_price_dollars")):
        if m.get(legacy) is None:
            m[legacy] = _to_cents(m.get(dollar))
    return m


def _is_tradeable(m: dict) -> bool:
    return (m.get("status") == "active"
            and isinstance(m.get("yes_ask"), int) and m["yes_ask"] > 0)


async def fetch_market() -> Optional[dict]:
    """Current (or next) tradeable btc-updown-15m market. Polymarket slugs
    key on the window's Unix start time, so discovery is deterministic â€”
    current window first, then the next two."""
    for delta in [0, 15, 30]:
        start = int(time.time() + delta * 60) // 900 * 900
        event = pm_client.slug_for_window_start(start)
        try:
            data = await _kget("/markets", {"event_ticker": event})
            for m in data.get("markets", []):
                _normalize_market(m)
                if _is_tradeable(m):
                    return m
        except Exception as e:
            log.debug(f"fetch_market event {event} error: {e}")

    # Fallback: grab first open market in series
    try:
        data = await _kget("/markets", {"series_ticker": SERIES_SLUG, "limit": 25})
        active = [m for m in (_normalize_market(x) for x in data.get("markets", []))
                  if _is_tradeable(m)]
        if active:
            return active[0]
    except Exception as e:
        log.debug(f"fetch_market series fallback error: {e}")
    return None


async def get_venue_position_count(ticker: str) -> Optional[int]:
    """Query Polymarket for the actual position count on a specific ticker.

    Returns the absolute number of contracts held (regardless of YES/NO side),
    or None if the query fails. Used as a defensive check before submitting
    SELL IOC orders (stop-loss) so we never accidentally over-sell â€” which
    Polymarket would either reject OR allow as a short position (very bad).

    Why this matters:
      â€¢ Our in-memory `pos["contracts"]` is set from `fill_count` returned by
        Polymarket at BUY time. Almost always accurate, but:
        - Chunked IOC fills can leave a stale count if a partial succeeded
          but the daemon's local accounting got out of sync
        - Manual interventions (canceling/closing positions from the web UI)
          won't update our in-memory state
        - A previous stop-loss retry that partially filled would leave us
          tracking fewer contracts than we actually have
      â€¢ Querying actual position right before SELL gives us ground truth;
        we use `min(in_memory, venue_actual)` for the sell quantity.
    """
    try:
        data = await _kget("/portfolio/positions", {"ticker": ticker, "limit": 5})
        positions = data.get("market_positions") or data.get("positions") or []
    except Exception as e:
        log.debug(f"  get_venue_position_count({ticker}) failed: {e}")
        return None

    for pos in positions:
        if pos.get("ticker") != ticker:
            continue
        pos_count = pos.get("position")
        if pos_count is None:
            try:
                pos_count = int(float(pos.get("position_fp") or 0))
            except (TypeError, ValueError):
                pos_count = 0
        try:
            return abs(int(pos_count))
        except (TypeError, ValueError):
            return 0
    # No matching position found = we have 0 contracts on this ticker
    return 0


async def get_venue_position_avg_cents(ticker: str) -> Optional[int]:
    """Authoritative AVERAGE cost basis (cents/contract) for a ticker position,
    from Polymarket's market_exposure Ã· count. Returns None on failure / no position.

    Used to set the resting-TP target off the REAL average rather than the
    daemon's blended `limit_price`, which drifts above the true average through
    patient-top-up rounding (observed 2026-06-15: tracked 80Â¢ vs actual 75.36Â¢,
    pushing the TP target to 90Â¢ instead of ~85Â¢). Mirrors recover_open_
    positions' cost extraction (handles market_exposure cents and the
    *_dollars string variant)."""
    try:
        data = await _kget("/portfolio/positions", {"ticker": ticker, "limit": 5})
        positions = data.get("market_positions") or data.get("positions") or []
    except Exception as e:
        log.debug(f"  get_venue_position_avg_cents({ticker}) failed: {e}")
        return None
    for pos in positions:
        if pos.get("ticker") != ticker:
            continue
        pos_count = pos.get("position")
        if pos_count is None:
            try:
                pos_count = int(float(pos.get("position_fp") or 0))
            except (TypeError, ValueError):
                pos_count = 0
        count = abs(int(pos_count or 0))
        if count <= 0:
            return None
        if pos.get("market_exposure") is not None:
            cost_total_cents = float(pos["market_exposure"])
        elif pos.get("market_exposure_dollars") is not None:
            try:
                cost_total_cents = float(pos["market_exposure_dollars"]) * 100.0
            except (TypeError, ValueError):
                return None
        else:
            return None
        avg = round(cost_total_cents / count)
        if 1 <= avg <= 99:
            return int(avg)
        return None
    return None


# 2026-06-19: LOG-ONLY Coinbase spot micro-momentum. Fed from get_btc_price
# (Coinbase branch only) so the series is pure Coinbase, the Polymarket BTC
# settlement reference. Read at NEW_SIGNALS fire time; never gates a trade.
_CB_SPOT_HIST: list = []   # [(ts_ms, coinbase_spot)] rolling

def _cb_spot_push(px) -> None:
    try:
        if px and px > 0:
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            _CB_SPOT_HIST.append((now_ms, float(px)))
            if len(_CB_SPOT_HIST) > 400:
                del _CB_SPOT_HIST[:200]
    except Exception:
        pass

def coinbase_momentum(window_s: float):
    """Signed %% change of Coinbase spot over ~window_s (positive = up).
    Uses freshest tick vs the most recent tick at least window_s old.
    Returns None when history is insufficient. Log-only."""
    try:
        if len(_CB_SPOT_HIST) < 2:
            return None
        now_ms, now_px = _CB_SPOT_HIST[-1]
        target = now_ms - window_s * 1000.0
        past = None
        for ts, px in reversed(_CB_SPOT_HIST):
            if ts <= target:
                past = (ts, px)
                break
        if past is None or past[1] <= 0:
            return None
        return round((now_px - past[1]) / past[1] * 100.0, 4)
    except Exception:
        return None


def _build_cv_composite(side, mkt, mins_left, ns, recent_5m):
    """Assemble RAW signals + vol proxy and call cv_composite (LOG-ONLY)."""
    try:
        sgn = 1.0 if side == "YES" else -1.0
        spot = (mkt.get("btc_price") or mkt.get("eth_price")
                or mkt.get("sol_price") or mkt.get("doge_price")
                or mkt.get("hype_price") or mkt.get("spot") or 0.0)
        strike = mkt.get("strike") or 0.0
        r5 = [x for x in (recent_5m or []) if isinstance(x, (int, float))]
        if len(r5) >= 2:
            m = sum(r5) / len(r5)
            sigma = (sum((x - m) ** 2 for x in r5) / len(r5)) ** 0.5
        else:
            sigma = 0.05
        raw = {
            "liq_skew": ns.get("liq_skew"),
            "book_skew": (sgn * ns["book_skew"]) if ns.get("book_skew") is not None else None,
            "oi_mom": ns.get("oi_d5m_pct"),
            "cb_mom": coinbase_momentum(60.0),
            "perp": (sgn * ns["perp_m60s"]) if ns.get("perp_m60s") is not None else None,
            "sigma_pct_min": sigma,
        }
        return cv_composite(side, float(strike), float(spot), float(mins_left), raw)
    except Exception as e:
        return {"ok": False, "err": repr(e)}


def _build_cv_composite_exit(side, spot, strike, mins_left):
    """Exit-time cv_composite from a FRESH live signal snapshot (LOG-ONLY).
    Lets us later label predict-cross / SL / fast-exit cuts as cut-winners vs
    cut-losers against the flip_prob at the instant of the cut."""
    try:
        if not spot or not strike:
            return {"ok": False, "err": "no_spot_or_strike"}
        _nsx = snapshot_new_signals(side)
        return _build_cv_composite(
            side, {"spot": float(spot), "strike": float(strike)},
            float(mins_left), _nsx, None)
    except Exception as e:
        return {"ok": False, "err": repr(e)}


async def get_btc_price() -> float:
    # 2026-06-18: multi-source spot with fallbacks. Coinbase is primary, but on
    # the Ubuntu deploy it can be DNS-sinkholed/geo-blocked, and ANY transient
    # error (timeout/5xx/rate-limit) used to fall through to 0.0. A 0 price is
    # catastrophic downstream: it reads as BTC ~100% below strike -> p(YES)=0 /
    # p(NO)=100% -> a phantom max-confidence NO trade. So we try Coinbase ->
    # OKX index -> Deribit index and only return 0.0 if ALL sources fail (and the
    # run_signal guard then blocks the trade). OKX/Deribit confirmed reachable
    # from the box even when Coinbase is blocked.
    async with httpx.AsyncClient(timeout=5) as c:
        try:
            r = await c.get("https://api.exchange.coinbase.com/products/BTC-USD/ticker")
            p = float(r.json().get("price", 0) or 0)
            if p > 0:
                _cb_spot_push(p)
                pm_client.note_boundary_spot(p)
                return p
        except Exception:
            pass
        try:
            r = await c.get("https://www.okx.com/api/v5/market/index-tickers?instId=BTC-USD")
            data = r.json().get("data") or []
            p = float(data[0].get("idxPx", 0) or 0) if data else 0.0
            if p > 0:
                log.warning(f"get_btc_price: Coinbase failed - using OKX index ${p:,.2f}")
                pm_client.note_boundary_spot(p)
                return p
        except Exception:
            pass
        try:
            r = await c.get("https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd")
            p = float((r.json().get("result") or {}).get("index_price", 0) or 0)
            if p > 0:
                log.warning(f"get_btc_price: Coinbase+OKX failed - using Deribit index ${p:,.2f}")
                pm_client.note_boundary_spot(p)
                return p
        except Exception:
            pass
    log.error("get_btc_price: ALL sources failed (Coinbase/OKX/Deribit) - returning 0.0; "
              "run_signal will block the trade.")
    return 0.0


async def run_signal(market: dict, bankroll: float,
                       last_bar_adverse_threshold: float = 0.10,
                       # 2026-05-28 PM: optional TP from previous poll's cache.
                       # Used to bypass the last-bar-adverse gate when ALL of
                       # gap/persist/TP point at extremes â€” in lock-in markets
                       # the very 5m bar that created the consensus often trips
                       # this gate, blocking the trade it was meant to inform.
                       tp_bs_p_above: Optional[float] = None,
                       last_bar_extreme_gap_min: float = 0.30,
                       last_bar_extreme_tp: float = 0.85,
                       # 2026-06-03: tunable STANDARD price caps. Defaults match
                       # constants in run_backtest.py (72c YES / 65c NO). Backtest
                       # of 9-day audit data showed expanding NO cap to 70-75c
                       # captures clean winning trades that currently fall in the
                       # gap between STANDARD (price cap) and STRONG-FLOOR (signal
                       # requirements). Estimated +$200-400/week at NO cap=70-75.
                       standard_price_cap_yes: Optional[int] = None,
                       standard_price_cap_no:  Optional[int] = None,
                       # 2026-06-15: GOLDEN-ZONE EXPANSION. Backtest (21d, 352
                       # trades) showed widening the band to 55-80c and dropping
                       # the near-strike distance gate + Hurst gate FOR GOLDEN
                       # roughly doubles golden volume (157->352) and PnL
                       # (+$2727->+$5834) while holding 80-84% WR. The 65-73c
                       # price band itself encodes moderate conviction, so near-
                       # strike golden-priced trades stay good (unlike strong-
                       # floor). 80c+ extension hurt, so cap at 80. Defaults
                       # preserve the original 65-73 / gated behavior.
                       golden_price_lo: int = 65,
                       golden_price_hi: int = 73,
                       golden_no_dist:  bool = False,
                       golden_no_hurst: bool = False) -> dict:
    strike   = float(market.get("floor_strike") or 0)
    yes_ask  = int(market.get("yes_ask") or 50)
    no_ask   = int(market.get("no_ask")  or 50)
    ticker   = market.get("ticker", "")
    close_ts = 0
    if market.get("close_time"):
        try:
            close_ts = datetime.fromisoformat(
                market["close_time"].replace("Z", "+00:00")
            ).timestamp()
        except Exception:
            pass

    btc_price    = await get_btc_price()
    minutes_left = max(0, (close_ts - time.time()) / 60) if close_ts else 7.5
    dist_pct     = (btc_price - strike) / strike * 100 if strike > 0 else 0.0

    # Candles (sync â€” run in executor to avoid blocking event loop)
    loop = asyncio.get_event_loop()
    candles_5m, candles_15m = await asyncio.gather(
        loop.run_in_executor(None, fetch_candles_5m,  2),
        loop.run_in_executor(None, fetch_candles_15m, 2),
    )

    check_ts = time.time()

    # Compute the last 6 closed 5-min returns (% change) â€” captured in the
    # audit log alongside the Markov signal so we can post-hoc inspect what
    # the price action looked like at each decision moment.
    sorted_5m   = sorted(candles_5m, key=lambda c: c[0])
    recent_5m_pct: list[float] = []
    for prev, curr in zip(sorted_5m[-7:], sorted_5m[-6:]):
        if prev[4] > 0:
            recent_5m_pct.append((curr[4] - prev[4]) / prev[4] * 100.0)

    # GK vol + Hurst
    ctx15   = [c for c in candles_15m if c[0] + 900 <= check_ts]
    last15  = list(reversed(ctx15[-32:])) if len(ctx15) >= 12 else []
    gk      = gk_vol(last15[:16])    if last15 else None
    hurst   = compute_hurst(last15[:24]) if last15 else None

    # d-score
    d_score = None
    if gk and gk > 0 and strike > 0:
        try:
            candles_left = max(minutes_left / 15.0, 1/60)
            d_score = math.log(btc_price / strike) / (gk * math.sqrt(candles_left))
        except Exception:
            pass

    # Markov
    history       = build_markov_history(candles_5m, check_ts)
    c5_by_ts      = {c[0]: c for c in candles_5m}
    c5_ts         = int(check_ts // 300) * 300 - 300
    c5_bar        = c5_by_ts.get(c5_ts) or c5_by_ts.get(c5_ts - 300)
    c5_prev       = c5_by_ts.get(c5_ts - 300)
    current_state = 4
    if c5_bar and c5_prev and c5_prev[4] > 0:
        current_state = price_change_to_state(
            (c5_bar[4] - c5_prev[4]) / c5_prev[4] * 100.0
        )

    full_history = (history + [current_state]) if history else [current_state, current_state]
    P            = build_transition_matrix(full_history)
    forecast     = predict_from_momentum(P, current_state, minutes_left, dist_pct)
    p_yes        = forecast["p_yes"]
    gap          = abs(p_yes - 0.5)
    persist      = forecast["persist"]
    has_history  = len(full_history) >= 20

    # Gates â€” all thresholds imported from run_backtest so research_loop
    # proposals actually flow through to live trading when merged.
    utc_hour  = datetime.now(timezone.utc).hour
    blocked   = utc_hour in BLOCKED_UTC_HOURS
    vol_ok    = gk is None or gk <= REF_VOL_15M * MAX_VOL_MULT
    hurst_ok  = hurst is None or hurst >= MIN_HURST
    markov_ok = has_history and gap >= MARKOV_MIN_GAP and persist >= MIN_PERSIST
    # 2026-05-28 PM: golden-zone check is now SYMMETRIC. Pre-change it only
    # looked at yes_ask 65-73, so NO trades at no_ask 65-73 (same asymmetric
    # risk:reward profile â€” paying 65-73Â¢ for $1 payout) were stuck on the
    # standard 3-10 min entry window. Now either side qualifying activates
    # the widened 2-14 min window.
    is_golden = (golden_price_lo <= yes_ask <= golden_price_hi) or \
                (golden_price_lo <= no_ask <= golden_price_hi)
    # 2026-06-15: golden-zone gate relaxations (backtest-validated). The price
    # band itself is the conviction filter, so golden can safely skip the
    # near-strike distance gate and the mean-reverting Hurst gate.
    if is_golden and golden_no_hurst:
        hurst_ok = True
    # Golden zone widening (2026-05-27): expanded from 3-12 â†’ 2-14 min.
    #
    # NOTE: This is a DEPARTURE from the 2026-05-25 research baseline
    # which found MAX_MINUTES_LEFT > 10 degraded profit factor. The
    # rationale for widening anyway:
    #   â€¢ The research was generated with Markov-only signals. Since then
    #     we added the ChainVector probability engine, futures lead-lag
    #     (binance_futures + OKX venues) and multi-tier EV gate. Forward-
    #     looking probability info may justify earlier entry at golden
    #     prices that pure-Markov couldn't.
    #   â€¢ Late side (2 min) catches near-close lock-in trades that the
    #     late-window-sure tier can confirm via TP â‰¥ 0.85.
    # Validate via audit log analysis: count fills in the 12-14 min and
    # 2-3 min windows + compare WR to the 3-12 baseline.
    if is_golden:
        time_ok       = 2 <= minutes_left <= 14
        time_win_str  = "2-14"
    else:
        time_ok       = MIN_MINUTES_LEFT <= minutes_left <= MAX_MINUTES_LEFT
        time_win_str  = f"{MIN_MINUTES_LEFT}-{MAX_MINUTES_LEFT}"
    side_is_yes = p_yes > 0.5
    limit_price = round(yes_ask if side_is_yes else no_ask)
    # 2026-06-03: CLI-tunable price caps (override constants from run_backtest)
    eff_yes_cap = standard_price_cap_yes if standard_price_cap_yes is not None else MAX_ENTRY_PRICE_YES
    eff_no_cap  = standard_price_cap_no  if standard_price_cap_no  is not None else MAX_ENTRY_PRICE_NO
    price_cap   = eff_yes_cap if side_is_yes else eff_no_cap
    price_ok    = limit_price <= price_cap
    # Adaptive near-strike threshold: when Markov gap is weak (â‰¤ 0.20),
    # require a larger buffer from strike. Near-strike trades on thin
    # signals are noise â€” BTC easily moves $30-80 in 5-10 min. Strong
    # signals get the original 0.02% lenient threshold preserved.
    NEAR_STRIKE_STRONG_GAP = 0.20
    NEAR_STRIKE_DIST_STRONG = 0.02   # |dist| â‰¥ 0.02% when gap â‰¥ 0.20
    NEAR_STRIKE_DIST_WEAK   = 0.10   # |dist| â‰¥ 0.10% when gap < 0.20
    dist_threshold = NEAR_STRIKE_DIST_STRONG if gap >= NEAR_STRIKE_STRONG_GAP else NEAR_STRIKE_DIST_WEAK
    dist_ok   = abs(dist_pct) >= dist_threshold
    if is_golden and golden_no_dist:
        dist_ok = True   # golden price band is the conviction filter

    # 2026-06-18: HARD GUARD on spot price. get_btc_price() returns 0.0 when all
    # price sources fail; a 0 (or feed-glitched) spot reads as BTC far below the
    # strike -> p(YES)=0 / p(NO)=100% -> a phantom max-confidence NO trade (e.g.
    # the NO @ 4.5c longshot). Never trade on a non-positive spot, or one that
    # disagrees with the latest 5m candle close by >25% (clear feed glitch).
    price_valid = btc_price > 0
    _ref_close  = sorted_5m[-1][4] if sorted_5m else None
    if price_valid and _ref_close and _ref_close > 0:
        if abs(btc_price - _ref_close) / _ref_close > 0.25:
            price_valid = False

    reasons: list[str] = []
    if not has_history:  reasons.append(f"building history ({len(full_history)}/20 candles)")
    if not price_valid:  reasons.append(
        f"invalid BTC spot ${btc_price:,.2f} (feed failure"
        + (f"; ref close ${_ref_close:,.0f}" if _ref_close else "") + ")")
    if not markov_ok:    reasons.append(f"Markov gap {gap:.3f}<{MARKOV_MIN_GAP} or persist {persist:.2f}<{MIN_PERSIST}")
    if blocked:          reasons.append(f"blocked UTC hour {utc_hour}:00")
    if not vol_ok:       reasons.append(f"high vol (GK={gk:.5f})")
    if not hurst_ok:     reasons.append(f"mean-reverting (Hurst={hurst:.2f})")
    if not time_ok:      reasons.append(f"timing {minutes_left:.1f}min outside {time_win_str}min window")
    if not price_ok:     reasons.append(f"price {limit_price}Â¢ > {'YES' if side_is_yes else 'NO'} cap {price_cap}Â¢")
    if not dist_ok:      reasons.append(
        f"near-strike noise (dist={dist_pct:+.4f}%, need â‰¥{dist_threshold:.2f}% with gap={gap:.3f})"
    )

    # Last-bar adverse-momentum gate (added 2026-05-27 PM after a NO @ 73Â¢
    # loss where the LAST 5m bar was +0.14% â€” a clear bounce just starting
    # â€” but the overall 6-bar momentum was still bearish so the existing
    # net-momentum check didn't catch it. This gate looks specifically at
    # the MOST RECENT bar; if it opposes trade direction by more than the
    # threshold, treat the trade as a late-stage reversal risk.
    # Threshold is in % per 5-min bar. Default 0.10% = $75-ish move on BTC
    # at $75k. Set to 0 to disable.
    #
    # 2026-05-28 PM extreme-signal bypass: when ALL of Markov gap, p_yes,
    # AND TP align at extremes (gapâ‰¥0.30 AND TP-direction at HIGH-CONV-grade
    # â‰¥0.85 / â‰¤0.15), the recent 5m bar that "opposes" direction is almost
    # always the bar that CREATED the consensus (e.g., BTC dropped sharply
    # to drive a NO lock-in, then bounces 0.17% in noise). Blocking these
    # is the gate's worst false-positive mode. Today's 14:15 drought window
    # (gap=0.50, p_yes=0, TP=0.034 for NO at 97Â¢) is a textbook example.
    last_bar_ok = True
    yes_lean_for_dir = p_yes > 0.5
    # Compute extreme-signal bypass flag
    tp_extreme_for_dir = False
    if tp_bs_p_above is not None:
        if yes_lean_for_dir:
            tp_extreme_for_dir = tp_bs_p_above >= last_bar_extreme_tp
        else:
            tp_extreme_for_dir = tp_bs_p_above <= (1.0 - last_bar_extreme_tp)
    extreme_signal_bypass = (
        gap >= last_bar_extreme_gap_min and tp_extreme_for_dir
    )
    if last_bar_adverse_threshold > 0 and recent_5m_pct and not extreme_signal_bypass:
        last_bar_pct = recent_5m_pct[-1]
        if yes_lean_for_dir and last_bar_pct < -last_bar_adverse_threshold:
            last_bar_ok = False
            reasons.append(
                f"last-bar adverse ({last_bar_pct:+.3f}% vs YES direction, "
                f"threshold Â±{last_bar_adverse_threshold:.2f}%)"
            )
        elif not yes_lean_for_dir and last_bar_pct > last_bar_adverse_threshold:
            last_bar_ok = False
            reasons.append(
                f"last-bar adverse ({last_bar_pct:+.3f}% vs NO direction, "
                f"threshold Â±{last_bar_adverse_threshold:.2f}%)"
            )

    all_ok = price_valid and markov_ok and not blocked and vol_ok and hurst_ok and time_ok and price_ok and dist_ok and last_bar_ok
    rec    = ("YES" if p_yes > 0.5 else "NO") if all_ok else "NO_TRADE"

    # Kelly sizing
    p_win     = p_yes if rec == "YES" else (1 - p_yes)
    p_d       = limit_price / 100
    fee_c     = MAKER_FEE_RATE * p_d * (1 - p_d)
    net_win   = (1 - p_d) - fee_c
    cost_c    = p_d + fee_c
    b         = net_win / cost_c if cost_c > 0 else 1.0
    kelly_full = max(0.0, (b * p_win - (1 - p_win)) / b) if rec != "NO_TRADE" else 0.0

    if   65 <= limit_price <= 73: frac = 0.35
    elif 73 < limit_price <= 79:  frac = 0.12
    elif 79 < limit_price <= 85:  frac = 0.08
    else:                         frac = 0.05

    risk_pct  = min(MAX_TRADE_PCT, frac * kelly_full)
    dyn_cap   = max(25, round(bankroll / 200 * 25))
    contracts = min(max(1, round(bankroll * risk_pct / cost_c)), dyn_cap) if rec != "NO_TRADE" else 0
    max_loss  = round(cost_c * contracts, 2)
    ev        = round(contracts * (net_win * p_win - cost_c * (1 - p_win)), 2)

    return {
        "approved":        rec != "NO_TRADE",
        "recommendation":  rec,
        "ticker":          ticker,
        "limit_price":     limit_price,
        "contracts":       contracts,
        "max_loss_usd":    max_loss,
        "expected_value":  ev,
        "rejection_reasons": reasons,
        "signal": {
            "p_yes":        round(p_yes, 4),
            "gap":          round(gap, 4),
            "persist":      round(persist, 3),
            "hurst":        round(hurst, 3) if hurst is not None else None,
            "gk_vol":       round(gk, 6) if gk else None,
            "d_score":      round(d_score, 3) if d_score else None,
            "minutes_left": round(minutes_left, 1),
            "history_len":  len(full_history),
            "utc_hour":     utc_hour,
            "is_golden":    is_golden,
        },
        "market": {
            "btc_price": round(btc_price, 2),
            "strike":    strike,
            "dist_pct":  round(dist_pct, 4),
            "yes_ask":   yes_ask,
            "no_ask":    no_ask,
        },
        "recent_5m_pct": [round(p, 4) for p in recent_5m_pct],
    }


async def get_live_bankroll() -> Optional[float]:
    """Polymarket /portfolio/balance current schema:
      { "balance": <cash_cents>, "balance_dollars": "<str>",
        "portfolio_value": <cents>, ... }
    Older schema used a nested {available_balance_cents, portfolio_value_cents} dict.
    Handle both."""
    try:
        data = await _kget("/portfolio/balance")
        bal  = data.get("balance")
        if isinstance(bal, dict):
            cents = (bal.get("available_balance_cents", 0)
                     + bal.get("portfolio_value_cents", 0))
        else:
            cents = (bal or 0) + (data.get("portfolio_value") or 0)
        return cents / 100 if cents else None
    except Exception as e:
        log.warning(f"Could not fetch live balance: {e}")
        return None


async def check_settlement(ticker: str) -> Optional[str]:
    """Returns 'YES', 'NO', or None if not settled yet."""
    try:
        data = await _kget(f"/markets/{ticker}")
        m = data.get("market", data)
        result = m.get("result")
        if result in ("yes", "no"):
            return result.upper()
    except Exception:
        pass
    return None


# â”€â”€ Startup position recovery â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
async def recover_open_positions(session: "Session") -> int:
    """On startup, query Polymarket for any open btc-updown-15m positions and rebuild the
    in-memory `session.pending` so a crashed-mid-window daemon can still record
    the eventual SETTLED WIN/LOSS line and accurate Daily P&L.

    Returns the number of positions recovered.

    Defensive about field shape because Polymarket's positions schema has multiple
    legacy/new field names (position vs position_fp, market_exposure vs
    market_exposure_dollars). Only recovers active (not yet settled) btc-updown-15m
    markets â€” skips manually-opened positions on other series.
    """
    try:
        data = await _kget("/portfolio/positions",
                           {"count_filter": "position", "limit": 50})
    except Exception as e:
        log.warning(f"Position recovery: failed to query Polymarket: {e}")
        return 0

    positions = data.get("market_positions") or data.get("positions") or []
    recovered = 0

    for pos in positions:
        ticker = pos.get("ticker") or ""
        if not ticker.startswith(TICKER_PREFIX):
            # Skip non-15m-BTC markets â€” could be manual user trades
            continue

        # Position quantity. Try int field first, then fixed-point string.
        pos_count = pos.get("position")
        if pos_count is None:
            try:
                pos_count = int(float(pos.get("position_fp") or 0))
            except (TypeError, ValueError):
                pos_count = 0
        if not pos_count:
            continue

        # Polymarket convention: positive = long YES, negative = long NO
        side      = "YES" if pos_count > 0 else "NO"
        contracts = abs(int(pos_count))

        # Market exposure = total cost paid (in cents). Fall back to dollars
        # field or to a 50Â¢ estimate so we never crash on schema drift.
        cost_total: float
        if pos.get("market_exposure") is not None:
            cost_total = pos["market_exposure"] / 100
        elif pos.get("market_exposure_dollars") is not None:
            try:
                cost_total = float(pos["market_exposure_dollars"])
            except (TypeError, ValueError):
                cost_total = contracts * 0.50
        else:
            cost_total = contracts * 0.50

        # Confirm market is still open / not yet settled
        try:
            mkt_data  = await _kget(f"/markets/{ticker}")
            m         = mkt_data.get("market", mkt_data)
            if m.get("result") in ("yes", "no"):
                # Already settled â€” Polymarket has already credited/debited the
                # account, no point re-tracking. Skip.
                continue
            close_iso = m.get("close_time", "")
        except Exception:
            close_iso = ""

        # Window id = close-time stamp matching what new_window_close() emits.
        try:
            close_dt  = datetime.fromisoformat(close_iso.replace("Z", "+00:00"))
            window_id = close_dt.strftime("%Y%m%d%H%M")
        except Exception:
            window_id = ticker  # last-resort fallback

        avg_price_dollars = (cost_total / contracts) if contracts else 0.0
        avg_price_cents   = round(avg_price_dollars * 100)
        net_win_total     = contracts - cost_total  # $1/contract payout âˆ’ cost

        session.pending[window_id] = {
            "ticker":      ticker,
            "side":        side,
            "contracts":   contracts,
            "limit_price": avg_price_cents,
            "cost":        round(cost_total, 2),
            "net_win":     round(net_win_total, 2),
            "order_id":    "recovered",
        }
        session.traded.add(window_id)
        recovered += 1
        log.info(
            f"RECOVERED â€” {ticker} {side} {contracts}c @ ~{avg_price_cents}Â¢ "
            f"cost=${cost_total:.2f} (window {window_id})"
        )

    return recovered


# â”€â”€ Main loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _log_active_params():
    """Print the currently-loaded strategy params so we can always tell what
    the live daemon is actually running and revert if results turn south."""
    BASELINE = {
        "MIN_MINUTES_LEFT":  6,
        "MAX_MINUTES_LEFT":  9,
        "MARKOV_MIN_GAP":    0.11,
        "MIN_PERSIST":       0.82,
        "MIN_HURST":         0.50,
        "MAX_VOL_MULT":      1.25,
        "BLOCKED_UTC_HOURS": {8, 11, 16, 18, 21},
        "MAX_ENTRY_PRICE_YES": 72,
        "MAX_ENTRY_PRICE_NO":  65,
        "KELLY_FRACTION":    0.18,
    }
    active = {
        "MIN_MINUTES_LEFT":  MIN_MINUTES_LEFT,
        "MAX_MINUTES_LEFT":  MAX_MINUTES_LEFT,
        "MARKOV_MIN_GAP":    MARKOV_MIN_GAP,
        "MIN_PERSIST":       MIN_PERSIST,
        "MIN_HURST":         MIN_HURST,
        "MAX_VOL_MULT":      MAX_VOL_MULT,
        "BLOCKED_UTC_HOURS": set(BLOCKED_UTC_HOURS),
        "MAX_ENTRY_PRICE_YES": MAX_ENTRY_PRICE_YES,
        "MAX_ENTRY_PRICE_NO":  MAX_ENTRY_PRICE_NO,
        "KELLY_FRACTION":    KELLY_FRACTION,
    }
    log.info("â”€â”€â”€ Active strategy params â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    drift = []
    for k, v in active.items():
        base = BASELINE[k]
        flag = "   "
        if v != base:
            flag = " * "
            drift.append(f"{k}: {base} â†’ {v}")
        log.info(f" {flag} {k:22s} = {v}   (baseline: {base})")
    if drift:
        log.info(f"  {len(drift)} param(s) differ from baseline. Revert by restoring values in run_backtest.py.")
    else:
        log.info("  All params at baseline.")
    log.info("â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")


def _is_orderbook_lockin(*,
                          orderbook: Optional[dict],
                          new_rec: str,
                          spread_max: int = 2,
                          price_min: int = 95,
                          gap: float = 0.0,
                          gap_min: float = 0.30) -> tuple[bool, Optional[str]]:
    """Detect 'lock-in' market signature in the Polymarket orderbook.

    A lock-in is when the orderbook screams a directional consensus:
      â€¢ Tight spread (â‰¤ `spread_max` cents) â€” market makers agree on price
      â€¢ Top-of-book price on OUR trade direction at extreme (â‰¥ `price_min`)
      â€¢ Markov gap also strong (â‰¥ `gap_min`) â€” internal model agrees

    When all three hold, we have orthogonal confirmation from a SECOND data
    source (Polymarket's own market microstructure) that the trade direction is
    locked in. Used to bypass TP-based gates that fail because Deribit's
    13h IV undershoots 15min binary probability â€” the orderbook captures the
    very thing TP can't see.

    Returns (is_lockin, reason_if_not). When True the caller may bypass HC's
    TP threshold and / or raise the LATE-SURE cap.
    """
    if not orderbook:
        return False, "no orderbook data"
    spread = orderbook.get("yes_spread")
    if spread is None:
        return False, "no spread in orderbook"
    if spread > spread_max:
        return False, f"spread={spread}Â¢ > {spread_max}Â¢"
    # Direction-specific top-of-book check
    if new_rec == "YES":
        top_bid = orderbook.get("yes_top_bid_px")
    else:
        top_bid = orderbook.get("no_top_bid_px")
    if top_bid is None:
        return False, f"no {new_rec} top-of-book in orderbook"
    if top_bid < price_min:
        return False, f"top {new_rec} bid {top_bid}Â¢ < {price_min}Â¢"
    if gap < gap_min:
        return False, f"Markov gap {gap:.3f} < {gap_min:.2f}"
    return True, None


def _is_late_window_sure(*,
                         mins_left: float,
                         limit_price: int,
                         tp_cached: Optional[dict],
                         new_rec: str,
                         markov_p_yes: float,
                         recent_5m: list,
                         late_window_mins: float,
                         late_window_price_max: int,
                         late_window_min_tp: float,
                         # 2026-05-28 PM: mirrors HC's lock-in TP bypass â€”
                         # when orderbook lock-in is confirmed in our direction,
                         # skip the TP threshold check (Deribit IV undershoots
                         # 15min binary probability; the orderbook is doing
                         # the confirmation work TP can't do at this horizon).
                         orderbook_lockin_bypass: bool = False) -> tuple[bool, Optional[str]]:
    """Late-window high-conviction override.

    A "sure" market is one where, late in the entry window (default <=5 min),
    the options-market TP says >=85% in our direction AND Markov also leans
    that way AND recent 5min BTC momentum doesn't disagree.

    Returns (is_sure, reason_if_not). When True, the caller may bid up to
    `late_window_price_max` (default 89Â¢) at a more lenient EV floor.
    """
    # 2026-05-28 PM: changed `>=` to `>` so the boundary poll (mins=5.0
    # exactly) also qualifies. Pre-change behavior missed the first poll
    # at the boundary which is often the BEST entry point.
    if mins_left > late_window_mins:
        return False, f"mins_left={mins_left:.1f} > {late_window_mins:.1f}"
    if limit_price > late_window_price_max:
        return False, f"price {limit_price}Â¢ > late-cap {late_window_price_max}Â¢"
    # TP check â€” bypass-able when orderbook lock-in is confirmed (mirrors HC).
    if tp_cached is None or tp_cached.get("error"):
        if not orderbook_lockin_bypass:
            return False, "TP unavailable"
    else:
        tp_p_yes = tp_cached.get("bs_p_above", 0.5)
        if new_rec == "YES":
            if tp_p_yes < late_window_min_tp:
                if not orderbook_lockin_bypass:
                    return False, f"TP_p_yes={tp_p_yes:.2f} < {late_window_min_tp:.2f}"
        else:
            # For NO trades, TP probability of being above the strike must be
            # symmetrically LOW (i.e., TP says it'll close below).
            if tp_p_yes > 1.0 - late_window_min_tp:
                if not orderbook_lockin_bypass:
                    return False, f"TP_p_yes={tp_p_yes:.2f} > {1.0 - late_window_min_tp:.2f}"

    # Markov must at least lean our direction (no gap requirement â€” that's the
    # whole point of this tier, to take trades when Markov is weakly confident
    # but TP is strongly confident).
    markov_yes_lean = markov_p_yes > 0.5
    rec_yes         = (new_rec == "YES")
    if markov_yes_lean != rec_yes:
        return False, f"Markov leans {'YES' if markov_yes_lean else 'NO'} but rec={new_rec}"

    # Recent 5min momentum confirmation. We look at the most recent 5min bin â€”
    # if BTC moved meaningfully AGAINST our direction in the last 5 min, the
    # "rising confidence" criterion fails. A small move (|m| < 0.05%) is fine.
    if recent_5m and len(recent_5m) >= 1:
        last_5m = recent_5m[-1]
        if rec_yes and last_5m < -0.08:
            return False, f"5m momentum {last_5m:+.3f}% against YES"
        if not rec_yes and last_5m > 0.08:
            return False, f"5m momentum {last_5m:+.3f}% against NO"

    return True, None


def _is_high_conviction(*,
                        mins_left: float,
                        limit_price: int,
                        tp_cached: Optional[dict],
                        new_rec: str,
                        sig: dict,
                        recent_5m: list,
                        high_conv_gap_min: float,
                        high_conv_persist_min: float,
                        high_conv_tp_strong: float,
                        high_conv_price_max: int,
                        high_conv_max_mins: float,
                        orderbook_lockin_bypass: bool = False,
                        hc_low_hurst_veto_enabled: bool = False,
                        hc_low_hurst_threshold: float = 0.30,
                        hc_low_hurst_markov_extremity: float = 0.35,
                        # 2026-06-03: HC distance-from-strike floor.
                        # When |dist_pct| < threshold, BTC is too close to strike
                        # for HC's extreme-directional bet to be safe. Backtest
                        # of 9-day audit showed strict HC (gapâ‰¥0.40, tpâ‰¥0.90)
                        # combined with |dist|â‰¥0.25% gave 13/13 wins (100% WR,
                        # +$127 PnL). The dist gate eliminates the 1 remaining
                        # loss without sacrificing winners.
                        dist_pct: Optional[float] = None,
                        hc_dist_min: float = 0.0) -> tuple[bool, Optional[str]]:
    """High-conviction signal-based override.

    Activates when ALL signals (Markov gap, Markov persistence, TP) agree
    strongly at extremes â€” used to lift the price cap for markets that are
    almost certainly going to settle our way but have already been priced
    aggressively. Allowed slightly outside the standard entry window
    (`high_conv_max_mins`, default 12 min) because extreme signals justify
    earlier entry â€” the reversal risk that the 10-min cap protects against
    is small when both Markov and TP say >99% same direction.

    Returns (is_high_conv, reason_if_not). When True, the caller may bid up
    to `high_conv_price_max` (default 97Â¢) but **must** show positive EV at
    that price (no negative-EV tolerance).
    """
    if mins_left > high_conv_max_mins:
        return False, f"mins_left={mins_left:.1f} > high-conv max {high_conv_max_mins:.1f}"
    if limit_price > high_conv_price_max:
        return False, f"price {limit_price}Â¢ > high-conv cap {high_conv_price_max}Â¢"
    if sig.get("gap", 0.0) < high_conv_gap_min:
        return False, f"gap={sig.get('gap',0.0):.3f} < {high_conv_gap_min:.2f}"
    if sig.get("persist", 0.0) < high_conv_persist_min:
        return False, f"persist={sig.get('persist',0.0):.2f} < {high_conv_persist_min:.2f}"
    # TP check â€” bypass-able when orderbook lock-in is confirmed.
    if tp_cached is None or tp_cached.get("error"):
        if not orderbook_lockin_bypass:
            return False, "TP unavailable"
    else:
        tp_p_yes = tp_cached.get("bs_p_above", 0.5)
        if new_rec == "YES":
            if tp_p_yes < high_conv_tp_strong:
                if not orderbook_lockin_bypass:
                    return False, f"TP_p_yes={tp_p_yes:.3f} < {high_conv_tp_strong:.2f}"
        else:
            if tp_p_yes > 1.0 - high_conv_tp_strong:
                if not orderbook_lockin_bypass:
                    return False, f"TP_p_yes={tp_p_yes:.3f} > {1.0 - high_conv_tp_strong:.2f}"

    # Markov direction must match TP direction
    markov_yes_lean = sig["p_yes"] > 0.5
    rec_yes         = (new_rec == "YES")
    if markov_yes_lean != rec_yes:
        return False, f"Markov leans {'YES' if markov_yes_lean else 'NO'}, rec={new_rec}"

    # Recent 5m momentum confirmation â€” don't bet 95+Â¢ on NO if BTC just
    # bounced up >0.1% in the last 5 min (and vice versa).
    if recent_5m and len(recent_5m) >= 1:
        last_5m = recent_5m[-1]
        if rec_yes and last_5m < -0.10:
            return False, f"5m momentum {last_5m:+.3f}% against YES"
        if not rec_yes and last_5m > 0.10:
            return False, f"5m momentum {last_5m:+.3f}% against NO"

    # â”€â”€ 2026-06-03 NEW: TIER 4 â€” LOW-HURST HC veto â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # When Hurst < threshold (mean-reverting regime) AND Markov is extreme,
    # the directional bet is fighting the regime. BTC tends to mean-revert
    # back through strike, invalidating the strong directional signal.
    # 2026-06-03 11:22 NO @ 97c (Markov=0.031, Hurst=0.27): BTC reversed
    # +$325 in 4 min, settled YES, -$43 loss with no hedge protection.
    # This gate blocks that pattern at entry.
    if hc_low_hurst_veto_enabled:
        h = sig.get("hurst")
        markov_p = sig.get("p_yes", 0.5)
        if (h is not None
                and h < hc_low_hurst_threshold
                and abs(markov_p - 0.5) >= hc_low_hurst_markov_extremity):
            return False, (f"low-hurst HC veto: Hurst={h:.2f} < "
                           f"{hc_low_hurst_threshold:.2f} AND |Markov-0.5|="
                           f"{abs(markov_p - 0.5):.2f} >= "
                           f"{hc_low_hurst_markov_extremity:.2f} "
                           f"(mean-reverting regime + extreme directional = "
                           f"reversal trap)")

    # â”€â”€ 2026-06-03: HC distance-from-strike floor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Blocks HC when BTC is too close to the strike, where reversals can
    # easily flip the contract. 9-day backtest showed |dist|â‰¥0.25% combined
    # with strict gap/tp filters gives 100% WR.
    if hc_dist_min > 0:
        if dist_pct is None:
            return False, (f"HC dist filter requires dist_pct (got None); "
                           f"floor={hc_dist_min:.3f}%")
        if abs(dist_pct) < hc_dist_min:
            return False, (f"HC dist veto: |dist|={abs(dist_pct):.3f}% < "
                           f"floor {hc_dist_min:.3f}% (too close to strike)")

    return True, None


def _is_high_conv_directional(*,
                               btc_price: float,
                               strike: float,
                               recent_5m: list,
                               rec: str,
                               momentum_min_pct: float,
                               distance_min_pct: float,
                               strong_distance_pct: float) -> tuple[bool, Optional[str]]:
    """Directional confirmation for vol bypass.

    GK volatility doesn't distinguish noisy vol (BTC bouncing) from directional
    vol (BTC trending hard one way). A directionally-moving BTC reflects strong
    information flow, not unpredictability â€” exactly the situation where
    high-conv signals deserve to trade despite the vol gate firing.

    Two-tier check:
      1. BTC must be meaningfully far from strike (â‰¥ distance_min_pct%) AND
      2. EITHER recent 5m return confirms direction (â‰¥ momentum_min_pct% in
         our favor), OR distance is strong (â‰¥ strong_distance_pct%, in which
         case the distance itself is the directional signal â€” useful when
         the move is recent and the closed 5m bin hasn't caught up yet).

    Returns (is_confirmed, reason_if_not). When True, vol may be bypassed for
    a high-conv-qualified trade.
    """
    if strike <= 0:
        return False, "invalid strike"
    distance_pct = (btc_price - strike) / strike * 100.0  # signed: positive = BTC above

    # Distance check (required floor)
    if rec == "NO":
        if distance_pct > -distance_min_pct:
            return False, (f"BTC only {distance_pct:+.3f}% from strike "
                           f"(need â‰¤ -{distance_min_pct:.3f}% for NO bypass)")
    else:  # YES
        if distance_pct < distance_min_pct:
            return False, (f"BTC only {distance_pct:+.3f}% from strike "
                           f"(need â‰¥ +{distance_min_pct:.3f}% for YES bypass)")

    abs_distance = abs(distance_pct)
    # Strong-distance fast path: BTC has clearly moved into our zone, distance
    # alone is the directional signal â€” no 5m momentum required. Captures
    # cases where the directional move just happened and hasn't yet shown up
    # in a closed 5m candle.
    if abs_distance >= strong_distance_pct:
        return True, None

    # Otherwise require explicit 5m momentum confirmation
    if not recent_5m or len(recent_5m) < 1:
        return False, (f"BTC distance {distance_pct:+.3f}% below strong threshold "
                       f"{strong_distance_pct:.3f}% AND no 5m data")
    last_5m = recent_5m[-1]
    if rec == "NO":
        if last_5m > -momentum_min_pct:
            return False, (f"5m momentum {last_5m:+.3f}% "
                           f"(need â‰¤ -{momentum_min_pct:.3f}% or distance â‰¥ {strong_distance_pct:.3f}%)")
    else:
        if last_5m < momentum_min_pct:
            return False, (f"5m momentum {last_5m:+.3f}% "
                           f"(need â‰¥ +{momentum_min_pct:.3f}% or distance â‰¥ {strong_distance_pct:.3f}%)")

    return True, None


def _is_late_window_directional(*, mins_left: float, gap: float, persist: float,
                                  tp_p_yes: Optional[float], new_rec: str,
                                  btc_price: float, strike: float,
                                  recent_5m: list,
                                  late_dir_mins: float,
                                  late_dir_gap_min: float,
                                  late_dir_persist_min: float,
                                  late_dir_distance_min: float,
                                  late_dir_momentum_min: float,
                                  late_dir_strong_distance: float
                                  ) -> tuple[bool, Optional[str]]:
    """Late-window directional vol bypass â€” looser than high-conv but late only.

    Targets the "developing directional move late in window" case where:
      - We're past the chop phase (window has clearly broken one way)
      - Markov gap + persist are strong but TP hasn't reached extreme
      - BTC has actively moved into the trade direction
      - Less time remains for reversal

    Two-tier momentum check (mirrors high-conv-directional):
      1. distance â‰¥ distance_min (required floor â€” BTC actually positioned)
      2. EITHER |distance| â‰¥ strong_distance (no momentum required â€” distance
         alone is enough when BTC has clearly moved past the strike)
         OR recent 5m return confirms direction with magnitude â‰¥ momentum_min

    Returns (is_late_dir, reason_if_not).
    """
    if mins_left > late_dir_mins:
        return False, f"mins_left={mins_left:.1f} > late-dir max {late_dir_mins:.1f}"
    if gap < late_dir_gap_min:
        return False, f"gap={gap:.3f} < {late_dir_gap_min:.2f}"
    if persist < late_dir_persist_min:
        return False, f"persist={persist:.2f} < {late_dir_persist_min:.2f}"
    if tp_p_yes is None:
        return False, "TP unavailable"
    if new_rec == "YES" and tp_p_yes < 0.5:
        return False, f"TP_p_yes={tp_p_yes:.3f} disagrees with YES"
    if new_rec == "NO" and tp_p_yes > 0.5:
        return False, f"TP_p_yes={tp_p_yes:.3f} disagrees with NO"
    if strike <= 0:
        return False, "invalid strike"
    dist_pct = (btc_price - strike) / strike * 100.0
    if new_rec == "NO":
        if dist_pct > -late_dir_distance_min:
            return False, (f"BTC only {dist_pct:+.3f}% from strike "
                           f"(need â‰¤ -{late_dir_distance_min:.3f}%)")
    else:
        if dist_pct < late_dir_distance_min:
            return False, (f"BTC only {dist_pct:+.3f}% from strike "
                           f"(need â‰¥ +{late_dir_distance_min:.3f}%)")
    # Strong-distance fast path
    if abs(dist_pct) >= late_dir_strong_distance:
        return True, None
    # Otherwise need 5m momentum confirmation
    if not recent_5m or len(recent_5m) < 1:
        return False, "no 5m data and distance below strong threshold"
    last_5m = recent_5m[-1]
    if new_rec == "NO" and last_5m > -late_dir_momentum_min:
        return False, (f"5m momentum {last_5m:+.3f}% "
                       f"(need â‰¤ -{late_dir_momentum_min:.3f}% or |dist|â‰¥{late_dir_strong_distance:.3f}%)")
    if new_rec == "YES" and last_5m < late_dir_momentum_min:
        return False, (f"5m momentum {last_5m:+.3f}% "
                       f"(need â‰¥ +{late_dir_momentum_min:.3f}% or |dist|â‰¥{late_dir_strong_distance:.3f}%)")
    return True, None


# â”€â”€ Safety hedge helpers (2026-05-31) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compute_hedge_target(yes_contracts: int, yes_entry_cents: int,
                          opposite_ask_cents: int, sl_loss_cents: int,
                          opposite_settle_assumed: float) -> int:
    """Return the target number of OPPOSITE-side contracts to hedge a YES (or NO)
    primary position so that an SL exit is approximately offset.

    Math:
      YES loss when SL fires â‰ˆ sl_loss_cents per contract (conservative).
      OPPOSITE gain at settle = opposite_settle_assumed - (opposite_ask_cents/100).
      To zero out: M = N Ã— yes_loss_per_contract / opposite_gain_per_contract.

    Returns 0 if the hedge math fails (e.g., opposite ask too high to gain).
    """
    yes_loss_per = sl_loss_cents / 100.0
    opp_gain_per = opposite_settle_assumed - (opposite_ask_cents / 100.0)
    if opp_gain_per <= 0:
        return 0
    return int(round(yes_contracts * yes_loss_per / opp_gain_per))


def hedge_eligible(tier: str, side: str, entry_cents: int,
                    opposite_ask_cents: int,
                    hedge_enabled: bool, hedge_tiers: tuple,
                    hedge_min_yes_entry: int, hedge_max_yes_entry: int,
                    hedge_max_no_cost: int) -> tuple:
    """Check if a position is eligible for a safety hedge.

    Returns (eligible: bool, reason: str). The `reason` is logged when we
    skip â€” important for understanding why hedges aren't attaching.
    """
    if not hedge_enabled:
        return False, "hedge disabled"
    if tier not in hedge_tiers:
        return False, f"tier '{tier}' not in {hedge_tiers}"
    if not (hedge_min_yes_entry <= entry_cents <= hedge_max_yes_entry):
        return False, (f"entry {entry_cents}\u00a2 outside "
                       f"hedge band {hedge_min_yes_entry}-{hedge_max_yes_entry}\u00a2")
    if opposite_ask_cents is None or opposite_ask_cents <= 0:
        return False, "opposite side ask unavailable"
    if opposite_ask_cents > hedge_max_no_cost:
        return False, (f"opposite ask {opposite_ask_cents}\u00a2 > "
                       f"max {hedge_max_no_cost}\u00a2")
    return True, ""


# â”€â”€ Smart-flip helpers (2026-06-01) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def smart_flip_eligibility(tier: str, primary_side: str, mins_remaining: float,
                            opp_bid_cents: int, hedge_active: bool,
                            enabled: bool, eligible_tiers: tuple,
                            min_opp: int, max_opp: int,
                            min_mins_remaining: float) -> tuple:
    """Check if a position is eligible for a smart defensive flip after SL.

    Returns (eligible: bool, reason: str). The `reason` is logged when we skip.

    Gates:
      - Feature enabled
      - No active hedge on this position (mutually exclusive)
      - Primary tier is HC or STRONG (entries 70-88Â¢ â€” the sweet spot)
      - Enough time remains to fill, monitor, and exit
      - Opposite bid in the sweet-spot band [50, 75]Â¢ (BTC clearly past strike
        but not so far that flip math fails)
    """
    if not enabled:
        return False, "flip disabled"
    if hedge_active:
        return False, "hedge already active (mutually exclusive)"
    if tier not in eligible_tiers:
        return False, f"tier '{tier}' not in {eligible_tiers}"
    if mins_remaining < min_mins_remaining:
        return False, f"{mins_remaining:.1f}min < {min_mins_remaining:.1f}min remaining"
    if opp_bid_cents is None or opp_bid_cents <= 0:
        return False, "opposite bid unavailable"
    if opp_bid_cents < min_opp:
        return False, (f"opp_bid {opp_bid_cents}\u00a2 < {min_opp}\u00a2 "
                       f"(BTC barely past strike \u2014 reversal likely)")
    if opp_bid_cents > max_opp:
        return False, (f"opp_bid {opp_bid_cents}\u00a2 > {max_opp}\u00a2 "
                       f"(insufficient upside to sell target)")
    return True, ""


def compute_smart_flip_size(primary_loss_usd: float, opp_entry_cents: int,
                              sell_target_cents: int, recovery_ratio: float,
                              max_capital_usd: float) -> tuple:
    """Compute the flip contract count.

    Sized to recover `recovery_ratio` of primary_loss when flip sells at
    `sell_target_cents`. Capped by `max_capital_usd` to prevent runaway sizes.

    Returns (M: int, reason: str). M=0 means "skip" with reason explaining why.
    """
    gain_per_contract_dollars = (sell_target_cents - opp_entry_cents) / 100.0
    if gain_per_contract_dollars <= 0:
        return 0, f"no upside ({sell_target_cents}c <= {opp_entry_cents}c)"
    target_recovery_dollars = primary_loss_usd * recovery_ratio
    M_raw = int(round(target_recovery_dollars / gain_per_contract_dollars))
    M_by_capital = int(max_capital_usd / (opp_entry_cents / 100.0))
    M = min(M_raw, M_by_capital)
    if M <= 0:
        return 0, "size = 0"
    return M, ""


async def _hedge_topup_inline(hedge_state: dict, current_yes_filled: int,
                               ticker: str, primary_side: str,
                               yes_entry_cents: int, effective_sl_cents: int,
                               hedge_no_settle_assumed: float,
                               hedge_max_no_cost: int,
                               hedge_max_capital_mult: float,
                               primary_capital_usd: float,
                               max_chunks_this_call: int = 2):
    """Incrementally fill more of the hedge based on current YES progress.

    Called between YES chunks during interleaved buying (Phase 2). Modifies
    `hedge_state` in place. Sizes hedge to the CURRENT cumulative YES filled
    count â€” so partial YES fills get partially-sized hedges automatically.

    `max_chunks_this_call` bounds how many NO chunks we submit per call so
    we don't block the YES loop for too long on a thin orderbook.

    State dict fields used and updated:
      cumulative_filled, sum_filled_x_price, order_ids, fill_log, warnings,
      last_target, last_no_ask, last_topup_ts
    """
    if not hedge_state.get("eligible"):
        return
    if current_yes_filled <= 0:
        return
    # Fetch fresh orderbook (this is a Polymarket REST call â€” ~100-300ms)
    try:
        mkt_data = await _kget(f"/markets/{ticker}")
        fresh_mkt = mkt_data.get("market", mkt_data)
        _normalize_market(fresh_mkt)
    except Exception as e:
        hedge_state["warnings"].append(f"hedge orderbook fetch failed: {e}")
        return
    if primary_side == "YES":
        opposite_ask = int(fresh_mkt.get("no_ask") or 0)
    else:
        opposite_ask = int(fresh_mkt.get("yes_ask") or 0)
    if opposite_ask <= 0:
        hedge_state["warnings"].append("opposite ask unavailable mid-fill")
        return
    if opposite_ask > hedge_max_no_cost:
        hedge_state["warnings"].append(
            f"opposite ask {opposite_ask}\u00a2 > max {hedge_max_no_cost}\u00a2 (skipping topup)"
        )
        return
    hedge_state["last_no_ask"] = opposite_ask
    # Compute target hedge size for current cumulative YES
    target_M = compute_hedge_target(
        yes_contracts=current_yes_filled,
        yes_entry_cents=yes_entry_cents,
        opposite_ask_cents=opposite_ask,
        sl_loss_cents=effective_sl_cents,
        opposite_settle_assumed=hedge_no_settle_assumed,
    )
    if target_M <= 0:
        return
    # Capital cap check (running total)
    proposed_hedge_capital = (
        hedge_state.get("cumulative_cost_usd", 0)
        + (target_M - hedge_state["cumulative_filled"]) * opposite_ask / 100.0
    )
    if (primary_capital_usd + proposed_hedge_capital) / primary_capital_usd > hedge_max_capital_mult:
        # Trim target to fit
        max_more_capital = (primary_capital_usd * hedge_max_capital_mult
                            - primary_capital_usd
                            - hedge_state.get("cumulative_cost_usd", 0))
        max_more_contracts = max(0, int(max_more_capital / (opposite_ask / 100.0)))
        target_M = min(target_M, hedge_state["cumulative_filled"] + max_more_contracts)
    hedge_state["last_target"] = target_M
    needed = target_M - hedge_state["cumulative_filled"]
    if needed <= 0:
        return
    # Determine v2 API params for OPPOSITE-side buy on same ticker
    opposite_side_label = "NO" if primary_side == "YES" else "YES"
    if opposite_side_label == "NO":
        v2_side = "ask"
        v2_price_yes_leg = 1 - opposite_ask / 100.0
    else:
        v2_side = "bid"
        v2_price_yes_leg = opposite_ask / 100.0
    # Submit up to max_chunks_this_call chunks
    HEDGE_CHUNK = 5
    chunks_done = 0
    while needed > 0 and chunks_done < max_chunks_this_call:
        this_chunk = min(HEDGE_CHUNK, needed)
        body = {
            "ticker":                     ticker,
            "client_order_id":            uuid.uuid4().hex,
            "side":                       v2_side,
            "count":                      str(this_chunk),
            "price":                      f"{v2_price_yes_leg:.4f}",
            "time_in_force":              "immediate_or_cancel",
            "self_trade_prevention_type": "taker_at_cross",
        }
        try:
            result = await _kpost("/portfolio/events/orders", body)
        except Exception as e:
            hedge_state["warnings"].append(f"hedge chunk failed: {e}")
            break
        chunk_filled = int(float(result.get("fill_count") or 0))
        if chunk_filled <= 0:
            hedge_state["fill_log"].append(
                f"hedge chunk: 0/{this_chunk} (depth exhausted at {opposite_ask}\u00a2)"
            )
            break  # orderbook exhausted at this ask
        avg_s = result.get("average_fill_price")
        try:
            avg_yes_leg = float(avg_s) if avg_s else v2_price_yes_leg
        except (TypeError, ValueError):
            avg_yes_leg = v2_price_yes_leg
        cost_per = (1 - avg_yes_leg) if opposite_side_label == "NO" else avg_yes_leg
        hedge_state["cumulative_filled"] += chunk_filled
        hedge_state["sum_filled_x_price"] += chunk_filled * avg_yes_leg
        hedge_state["cumulative_cost_usd"] += chunk_filled * cost_per
        oid = result.get("order_id") or result.get("client_order_id")
        if oid:
            hedge_state["order_ids"].append(oid)
        hedge_state["fill_log"].append(
            f"hedge chunk: {chunk_filled}/{this_chunk} @ ${avg_yes_leg:.4f} "
            f"(opposite={opposite_side_label}, cost_per={cost_per:.4f})"
        )
        needed -= chunk_filled
        chunks_done += 1
    hedge_state["last_topup_ts"] = datetime.now(timezone.utc).isoformat()


async def main_loop(dry_run: bool, bankroll: float,
                     term_prob_disabled: bool = False,
                     term_prob_relax:    bool = False,
                     futures_lead_disabled: bool = False,
                     futures_lead_lookback_s: float = 6.0,
                     futures_lead_veto_bps:   float = 5.0,
                     # ChainVector momentum scorecard (/momentum): capped tanh
                     # nudge of combined_p when the cross-venue aggregate agrees
                     # with the trade, and a hard veto when it is STRONGLY
                     # against with breadth confirmation.
                     cv_mom_boost_enabled:   bool  = True,
                     cv_mom_boost_weight:    float = 0.05,
                     cv_mom_boost_scale:     float = 50.0,
                     cv_mom_veto_enabled:    bool  = True,
                     cv_mom_veto_score:      float = 65.0,
                     cv_mom_veto_breadth:    float = 0.60,
                     # ChainVector probability engine hard floor: veto when the
                     # six-estimator ensemble gives our side <= this at exact TTE
                     # (the soft integration is the --ev-tp-weight blend).
                     cv_prob_veto_enabled:   bool  = True,
                     cv_prob_veto_max:       float = 0.22,
                     # ChainVector prediction quote-stability veto: block when
                     # /predictions/stability momentum_score (signed toward our
                     # side) shows the Polymarket quote repricing hard against us.
                     cv_stab_veto_enabled:   bool  = True,
                     cv_stab_mom_against:    float = 45.0,
                     # ChainVector liquidation cascade-risk veto: block when
                     # cascade risk_score is extreme AND the at-risk side's
                     # forced flow points against the position.
                     cv_cascade_veto_enabled: bool  = True,
                     cv_cascade_veto_score:   float = 75.0,
                     consensus_veto_enabled: bool = True,
                     consensus_min_move_pct: float = 0.005,
                     consensus_okx_lookback_s: float = 6.0,
                     # 2026-06-01: Smart bypass for consensus veto.
                     # 24h audit (17 vetoes) showed 65% of vetoes had the
                     # last 30min of 5m bars STRONGLY favoring the trade
                     # direction â€” the 6s blip was clearly counter-trend
                     # noise. Three independent bypasses prevent false
                     # positives without weakening the core veto.
                     consensus_smart_bypass:    bool  = True,
                     consensus_long_window_s:   float = 60.0,
                     consensus_long_min_pct:    float = 0.030,
                     consensus_5m_favor_pct:    float = 0.10,
                     consensus_far_dist_pct:    float = 0.25,
                     consensus_far_max_move_pct: float = 0.020,
                     high_conv_confirmed_frac: float = 0.10,
                     hc_block_on_split_externals: bool = True,
                     standard_confirmed_boost: float = 1.5,
                     sl_enabled:              bool  = True,
                     sl_loss_cents:           int   = 45,
                     sl_loss_cents_high_conv: int   = 30,
                     max_loss_per_trade:      float = 0.0,
                     # 2026-07-01: separate dollar-stop cap for high_conv. This
                     # tier has NO effective stop today (sl_loss_cents_high_conv
                     # is set to 99 live) so deep-ITM reversals ride to
                     # settlement (e.g. -$615/-$618). BID_TRAJECTORY backtest:
                     # deepest high_conv WINNER dip was $343; the catchable
                     # losers dipped $459/$606 â€” a $400 cap clips 0 winners and
                     # truncates those tails. 0 = disabled (falls back to no HC
                     # dollar stop, prior behavior).
                     max_loss_per_trade_high_conv: float = 0.0,
                     sl_grace_mins:           float = 1.5,
                     sl_disable_late_mins:    float = 1.5,
                     sl_poll_interval_s:      float = 5.0,
                     # 2026-05-30: HC trades enter deep ITM (87-89Â¢ NO) where
                     # downside is asymmetric. Audit of all 3 HC SL exits showed
                     # the bid blew past BOTH the 30Â¢ and 25Â¢ triggers BETWEEN
                     # polls â€” the daemon never observed an intermediate price.
                     # Tightening the threshold doesn't help; faster polling does.
                     # HC positions now poll every 2s vs the standard 5s.
                     sl_poll_interval_hc_s:   float = 2.0,
                     sl_trigger_mode:         str   = "mid",
                     sl_aggressive_sell:      bool  = True,
                     # 2026-05-30 NEW: Futures FAST-EXIT signal.
                     # During the SL monitor, snapshot the lead-venue futures
                     # over `window_s` seconds. If futures moved against the
                     # position by `threshold_pct` or more, exit immediately
                     # (using same aggressive sell as SL). This catches fast
                     # reversals before the Polymarket bid catches up â€” leading
                     # rather than lagging the SL trigger.
                     # 2026-05-30 (initial): 0.30%/60s â€” too conservative,
                     #   ZERO fires in first 10h despite 3 SL losses.
                     # 2026-05-30 (tightened): 0.20%/30s â€” more sensitive
                     #   based on user feedback. Catches faster moves over
                     #   shorter windows. Trade-off: slightly higher chance
                     #   of false-exits on winners that briefly dip.
                     # Sanity guard: any |move|>5% treated as bad tick.
                     futures_fast_exit_enabled:    bool  = True,
                     futures_fast_exit_window_s:   float = 30.0,
                     futures_fast_exit_threshold_pct: float = 0.20,
                     futures_fast_exit_sanity_max_pct: float = 5.0,
                     # 2026-05-31 NEW: Safety Hedge feature.
                     # After a HC/STRONG-FLOOR YES position fills, attempt to
                     # buy NO contracts on the same ticker as a downside hedge.
                     # If YES wins â†’ NO settles 0 (cost = hedge premium).
                     # If YES SL fires â†’ hold NO to settlement; NO gain offsets
                     # YES loss. Variance reduction at modest EV cost.
                     # Phase 1 (this session): basic hedge attach with fixed
                     # ratio sizing. Phase 2 (next): synchronized chunking.
                     # Math fails for LATE-SURE (very deep ITM) â€” gate excludes.
                     hedge_enabled:             bool  = False,
                     hedge_tiers:               tuple = ("high_conv", "strong"),
                     hedge_min_yes_entry:       int   = 70,
                     hedge_max_yes_entry:       int   = 88,
                     hedge_max_no_cost:         int   = 28,
                     hedge_no_settle_assumed:   float = 0.95,
                     hedge_max_capital_mult:    float = 2.5,
                     hedge_widened_sl_cents:    int   = 50,
                     # 2026-05-31 Phase 5: smart NO sell. When primary SL has
                     # fired and we're holding the hedge, monitor the opposite-
                     # side bid. If it spikes above the target threshold, sell
                     # the hedge now (lock in profit, avoid cross-strike-twice
                     # tail risk). Trail logic: if bid drops X cents from peak,
                     # also sell.
                     hedge_no_sell_target:      int   = 97,
                     hedge_no_sell_trail:       int   = 10,
                     hedge_post_sl_poll_s:      float = 2.0,
                     # 2026-06-01: Smart Defensive Flip feature.
                     # After primary SL fires, if signals support continuation
                     # in the new direction, BUY opposite side as a small
                     # defensive position. Recover part of primary loss.
                     # Math: net positive at 50%+ continuation rate due to
                     # tight 15c flip SL + capital cap.
                     # Eligibility: HC/STRONG tier, opp_bid in [50,75]c band,
                     # >=5min remaining, no active hedge.
                     smart_flip_enabled:               bool  = False,
                     # 2026-06-01: LATE-SURE added. Original exclusion assumed
                     # late-sure entries were too deep-ITM for flip math, but
                     # the opp_bid band gate [50-75] already handles that case
                     # (deep-ITM SLs leave opp_bid >75, gate filters out).
                     smart_flip_tiers:                 tuple = ("high_conv", "strong", "late_sure"),
                     smart_flip_min_opp_entry:         int   = 50,
                     smart_flip_max_opp_entry:         int   = 75,
                     smart_flip_recovery_ratio:        float = 0.50,
                     smart_flip_sl_cents:              int   = 15,
                     smart_flip_sell_target:           int   = 89,
                     smart_flip_trail_cents:           int   = 10,
                     smart_flip_max_capital_usd:       float = 100.0,
                     smart_flip_min_mins_remaining:    float = 5.0,
                     smart_flip_require_futures_confirm: bool  = True,
                     smart_flip_futures_confirm_pct:   float = 0.10,
                     smart_flip_futures_window_s:      float = 30.0,
                     smart_flip_poll_s:                float = 2.0,
                     # 2026-06-01: Retry loop on flip skip. If first check
                     # fails any gate (opp_bid out of band, futures not
                     # confirming, etc.), wait `retry_sleep_s` and re-check.
                     # Catches "just-missed" cases where conditions clear
                     # within ~30-60s post-SL.
                     smart_flip_retry_attempts:        int   = 3,
                     smart_flip_retry_sleep_s:         float = 15.0,
                     # 2026-06-01: Hurst+TP disagreement veto (Gate B).
                     # When BOTH conditions hold:
                     #   Hurst >= min_hurst (strong-trending regime)
                     #   |TP - Markov| in adverse direction >= min_diff
                     # The trade is blocked. Catches "chasing a fading top"
                     # pattern: Markov sees recent momentum but TP (options
                     # market) already prices in the reversal.
                     #
                     # 7-day historical replay: catches 3 disaster losses
                     # (âˆ’$402 total), blocks 3 small wins (+$37), net +$365.
                     hurst_tp_veto_enabled:        bool   = False,
                     hurst_tp_veto_min_hurst:      float  = 0.80,
                     hurst_tp_veto_min_diff:       float  = 0.05,
                     hurst_tp_veto_tiers:          tuple  = ("high_conv", "strong"),
                     # 2026-06-01: Directional max-adverse-bar veto.
                     # Block trade direction X if any of the recent 5m bars
                     # moved >= threshold AGAINST X. Catches "catching a
                     # falling knife" / "chasing a dead-cat bounce" pattern
                     # where Markov sees the small reversal bar but ignores
                     # the much larger preceding adverse spike.
                     # 7-day backtest: catches 2 disasters (-$226), blocks
                     # 1 small win (+$1.28), net +$225/week.
                     max_adverse_bar_veto_enabled: bool   = False,
                     max_adverse_bar_veto_pct:     float  = 0.30,
                     max_adverse_bar_veto_tiers:   tuple  = ("high_conv", "strong"),
                     # 2026-06-02: Cumulative adverse momentum veto.
                     # Catches the "slow steady drift against the trade" pattern
                     # where no SINGLE 5m bar exceeds 0.30% but the cumulative
                     # 30-min drift in the adverse direction is significant.
                     # 2026-06-02 05:09 catastrophic NO trade (-$301) had no
                     # single bar above 0.176% but cumulative +0.359% adverse
                     # drift. The market never repriced (bid stayed 84-96Â¢) so
                     # SL didn't fire; BTC bounced to settle YES at the wire.
                     # 7-day simulation: thr=0.35 blocks 5 trades (4W/1L),
                     # saves $301 in losses, $28 in forgone wins, net +$273.
                     cum_adverse_momentum_veto_enabled: bool  = True,
                     cum_adverse_momentum_veto_pct:     float = 0.35,
                     cum_adverse_momentum_veto_tiers:   tuple = ("high_conv", "strong", "late_sure"),
                     # 2026-06-03: TIER 4 â€” LOW-HURST HC veto.
                     # Catches "extreme directional signal in mean-reverting
                     # regime" trap. Hurst < threshold means BTC is oscillating
                     # (mean-reverting), so extreme Markov readings are likely
                     # to be reversed by mean reversion. Default ENABLED.
                     # 7-day backtest: blocks 4 catastrophic trades, saves $450.
                     hc_low_hurst_veto_enabled:    bool  = True,
                     hc_low_hurst_threshold:       float = 0.30,
                     hc_low_hurst_markov_extremity: float = 0.35,
                     # 2026-06-03: HC distance-from-strike floor.
                     # When |dist_pct| < threshold, BTC is too close to strike
                     # for HC's extreme bet to be safe. 9-day backtest with
                     # strict HC (gapâ‰¥0.40, persistâ‰¥0.95, tpâ‰¥0.90) + |dist|â‰¥0.25
                     # gave 13/13 wins, +$127. Default 0.0 (no filter).
                     hc_dist_min:                  float = 0.0,
                     # 2026-06-03: tunable STANDARD price caps.
                     # Defaults match constants from run_backtest.py.
                     standard_price_cap_yes: Optional[int] = None,
                     standard_price_cap_no:  Optional[int] = None,
                     # 2026-06-03: FADE-BOUNCE tier.
                     # Hypothesis: when Polymarket market dips into the 40-55Â¢ band
                     # mid-window, the eventual outcome is dominated by the
                     # original direction (85% NO wins in our 7-day audit of 73
                     # such windows). This tier fires as a SECOND, smaller
                     # position at the discount price when the primary trade
                     # is active and signals still confirm the direction.
                     # 7-day counterfactual: replacing early ~85Â¢ entries with
                     # ~50Â¢ entries on the same 70 winning windows would have
                     # added +$1,091 in PnL.
                     # Default OFF â€” opt-in via --fade-bounce-enabled.
                     fade_bounce_enabled:        bool  = False,
                     fade_bounce_no_ask_min:     int   = 40,
                     fade_bounce_no_ask_max:     int   = 55,
                     fade_bounce_yes_side_enabled: bool = False,
                     fade_bounce_markov_no_max:  float = 0.45,
                     fade_bounce_markov_yes_min: float = 0.55,
                     fade_bounce_hurst_min:      float = 0.50,
                     fade_bounce_dist_min:       float = 0.03,
                     fade_bounce_min_mins:       float = 3.0,
                     fade_bounce_max_mins:       float = 12.0,
                     fade_bounce_min_stake_pct:  float = 0.015,
                     fade_bounce_kelly_frac:     float = 0.05,
                     fade_bounce_sl_cents:       int   = 20,
                     fade_bounce_max_capital_usd: float = 20.0,
                     ev_gate:    bool  = False,
                     ev_floor:   float = 0.05,
                     ev_ceiling: int   = 90,
                     ev_tp_weight: float = 0.5,
                     # â”€â”€ 2026-06-16: OKX-momentum confidence boost â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                     okx_boost_enabled: bool  = False,
                     okx_boost_weight:  float = 0.06,   # max ~Â±6pp nudge
                     okx_boost_scale:   float = 0.015,  # OKX 6s move(%) at which tanh~0.76
                     ev_strong_floor:     float = -0.03,
                     ev_strong_gap_min:   float = 0.13,
                     ev_strong_price_max: int   = 88,
                     ev_strong_tp_min:    float = 0.55,
                     ev_strong_max_mins:  float = 8.0,
                     ev_strong_max_adverse_momentum: float = 0.10,
                     last_bar_adverse_threshold: float = 0.10,
                     orderbook_signal_enabled: bool  = True,
                     trade_flow_signal_enabled: bool = True,
                     trade_flow_lookback_n:    int   = 20,
                     # 2026-05-28 PM: orderbook-confirmed lock-in bypasses.
                     # When all three hold (spreadâ‰¤max, top-of-book â‰¥ price_min,
                     # Markov gap â‰¥ gap_min, direction-aligned), Polymarket's own
                     # microstructure has confirmed the trade direction at
                     # extreme conviction. Used to (a) bypass HC's TP threshold
                     # and (b) raise the LATE-SURE cap by 1Â¢ (98 â†’ 99).
                     orderbook_lockin_enabled:        bool = True,
                     orderbook_lockin_spread_max:     int  = 2,
                     # 2026-05-28 PM3: lowered 90 â†’ 85. 90 was still strict
                     # in moderate markets (e.g., tonight's 22:30 window:
                     # NO @ 80Â¢ with no_top_bid=75Â¢ â€” clearly directional
                     # consensus but not "extreme"). At 85Â¢ market still
                     # implies 85% confidence; combined with gapâ‰¥0.20 and
                     # spreadâ‰¤2Â¢ provides cross-source confirmation.
                     orderbook_lockin_price_min:      int  = 85,
                     # 2026-05-28 PM2: lowered 0.30 â†’ 0.20 to match the new
                     # high_conv_gap_min. Symmetric so HC's lock-in bypass
                     # (requires lock-in confirmed) actually works for the
                     # 0.20-0.30 gap range that HC now accepts.
                     orderbook_lockin_gap_min:        float = 0.20,
                     late_window_price_max_lockin:    int  = 99,
                     # 2026-05-28 PM: when lock-in is confirmed, HC's EV floor
                     # relaxes to this value (matches LATE-SURE -$0.10). The
                     # orderbook is doing additional confirmation, so the
                     # tighter standard HC floor is too conservative.
                     hc_lockin_ev_floor:              float = -0.10,
                     # 2026-05-28 PM3: floor-stake for HC trades fired under
                     # the lock-in path (where Kelly returns 0 because the
                     # trade has slight negative EV by the floor). Without
                     # this, lock-in HC fires only 1 contract â€” defeats the
                     # purpose. Matches LATE-SURE's 1.5% default.
                     hc_lockin_min_stake_pct:         float = 0.015,
                     # 2026-05-28 PM3: NEW â€” allow STRONG-FLOOR tier to bypass
                     # the Hurst gate (previously only HC/LATE-SURE could).
                     # Justified when strong-floor's other gates (gapâ‰¥0.13,
                     # priceâ‰¤88, TP-meaningful, momentum, timing) hold.
                     # Caveat: strong-floor accepts modest negative EV
                     # (default -$0.06). In mean-reverting regimes (low
                     # Hurst), trades can flip direction. Lowest-risk usage
                     # combines this with --ev-strong-floor -0.10 to widen
                     # the EV cushion.
                     strong_floor_hurst_bypass:       bool  = True,
                     retry_walk_cents:    int   = 0,
                     max_window_fill_attempts: int = 5,
                     refill_retry_sleep_s:     float = 30.0,
                     late_window_mins:        float = 5.0,
                     # 2026-05-28: raised 96 â†’ 98 after 3-day audit showed
                     # EVERY late-window lock-in candidate (104 polls) was at
                     # 97-100Â¢ ask. Cap at 96 made the tier unreachable.
                     late_window_price_max:   int   = 98,
                     # 2026-05-28: lowered 0.85 â†’ 0.75. Deribit IV is for 13h+
                     # options applied to a 3-min binary â€” BS underestimates
                     # near-expiry probability. Late-window TPs cluster at
                     # 0.65-0.80 even when market consensus is 85-98%.
                     late_window_min_tp:      float = 0.75,
                     late_window_ev_floor:    float = -0.10,
                     # 2026-05-28: NEW â€” allow LATE-SURE to bypass the high-vol
                     # blocker the same way HIGH-CONV can. LATE-SURE's own
                     # qualifying conditions (mins<5 + TP-direction + Markov
                     # agreement + momentum check) provide enough directional
                     # confirmation to override vol-regime caution.
                     late_sure_vol_bypass:    bool  = True,
                     strong_floor_min_stake_pct: float = 0.010,
                     late_sure_min_stake_pct:    float = 0.015,
                     # â”€â”€ 2026-06-15: MIN ENTRY SIZE (standard tier) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                     # Skip standard-tier entries Kelly-sized below this many
                     # contracts instead of opening a token position (which
                     # locks out a better/high-conv re-entry for the window).
                     # Default 2 => skip 1-contract standard entries; the
                     # window stays open and is re-evaluated. Set 1 to disable.
                     standard_min_entry_contracts: int = 2,
                     # â”€â”€ 2026-06-15: GOLDEN-ZONE EXPANSION (backtest-validated) â”€â”€
                     golden_price_lo: int = 65,
                     golden_price_hi: int = 73,
                     golden_no_dist:  bool = False,
                     golden_no_hurst: bool = False,
                     # 2026-05-28 PM2: lowered 0.30 â†’ 0.20. HC's gap was
                     # historically 0.30 (strict â€” extreme Markov agreement).
                     # Today's audit shows many windows had sustained gap
                     # 0.20-0.30 with strong direction and orderbook lock-in
                     # but were blocked from HC. With orderbook + persistâ‰¥0.90
                     # + new lock-in bypass providing additional confirmation,
                     # 0.20 is sufficient signal strength.
                     high_conv_gap_min:      float = 0.20,
                     high_conv_persist_min:  float = 0.90,
                     # 2026-05-28 PM: lowered 0.85 â†’ 0.80. Same Deribit-IV
                     # undershoot bug as LATE-SURE: Deribit options are 13h+
                     # out, so BS-based TP applied to a 15min binary
                     # systematically reads ~5-10 percentage points low.
                     # Today's drought had multiple windows blocked at
                     # TP=0.83-0.84 (gap 0.49, p_yes 0.99) â€” clearly
                     # qualifying signals that the 0.85 floor missed.
                     high_conv_tp_strong:    float = 0.80,
                     high_conv_price_max:    int   = 97,
                     high_conv_ev_floor:     float = 0.005,
                     high_conv_max_mins:     float = 12.0,
                     high_conv_vol_bypass:           bool  = True,
                     high_conv_vol_bypass_momentum:  float = 0.10,
                     high_conv_vol_bypass_distance:  float = 0.15,
                     high_conv_vol_bypass_strong_distance: float = 0.25,
                     late_dir_enabled:        bool  = True,
                     late_dir_mins:           float = 5.0,
                     late_dir_gap_min:        float = 0.25,
                     late_dir_persist_min:    float = 0.95,
                     late_dir_distance_min:   float = 0.05,
                     late_dir_momentum_min:   float = 0.05,
                     late_dir_strong_distance:float = 0.08,
                     late_dir_ev_floor:       float = -0.05,
                     late_dir_price_max:      int   = 89,
                     order_lead_cents:        int   = 0,
                     okx_disabled:            bool  = False,
                     okx_poll_interval_s:     float = 3.0,
                     # â”€â”€ 2026-06-09: Rolling-WR adaptive throttle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                     # Detects regime shifts via observed outcomes (rolling WR
                     # over last N trades). When triggered, blocks weaker tiers
                     # (default standard+strong) until WR recovers or timeout.
                     # Off by default; opt-in via --rolling-wr-enabled.
                     rolling_wr_enabled:      bool  = False,
                     rolling_wr_window:       int   = 5,
                     rolling_wr_threshold:    float = 0.40,
                     rolling_wr_timeout_mins: float = 120.0,
                     rolling_wr_defensive_tiers: tuple = ("standard", "strong"),
                     # â”€â”€ 2026-06-11: ADAPTIVE BANKROLL (risk throttle) â”€â”€â”€â”€â”€â”€â”€
                     # Shrinks sizing to reduced_frac of bankroll on a loss
                     # cluster or weak rolling WR; recovers to 100% on
                     # demonstrated WR recovery. Keeps all tiers trading
                     # (live data preserved). Resets to NORMAL on restart.
                     adaptive_br_enabled:          bool  = False,
                     adaptive_br_reduced_frac:     float = 0.15,
                     adaptive_br_loss_trigger_usd: float = 300.0,
                     adaptive_br_wr_trigger:       float = 0.50,
                     adaptive_br_wr_window_h:      float = 3.0,
                     adaptive_br_wr_min_trades:    int   = 5,
                     adaptive_br_recover_wr:       float = 0.75,
                     adaptive_br_recover_window:   int   = 6,
                     adaptive_br_recover_min_wins: int   = 3,
                     # â”€â”€ 2026-06-12: PATIENT TOP-UP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                     # Complete the originally-intended position size when the
                     # ask returns to EV-valid prices later in the window.
                     # Every attempt re-runs the FULL signal (fresh direction,
                     # fresh EV at current ask) â€” the adverse-selection guard.
                     patient_topup_enabled:    bool  = False,
                     patient_topup_interval_s: float = 20.0,
                     patient_topup_min_mins:   float = 2.5,
                     # 2026-06-15: when True, the top-up can grow the position
                     # toward the FRESH Kelly size (recomputed at current price)
                     # rather than capping at the frozen entry-time intent.
                     # Only ever adds when fresh EV>0. Lets a trade that was
                     # Kelly-floored at entry (because EV was negative then)
                     # build up if it later becomes genuinely +EV.
                     patient_topup_dynamic_kelly: bool = False,
                     # â”€â”€ 2026-06-15: PERP-CONFIRMED TAKE-PROFIT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                     # Resting TP at entry+cents, gated to perp-momentum-
                     # confirmed entries (where backtest showed it's +EV).
                     take_profit_enabled:         bool  = False,
                     take_profit_cents:           int   = 10,
                     take_profit_perp_confirmed_only: bool = True,
                     take_profit_perp_min:        float = 0.0,
                     # â”€â”€ 2026-06-15: RESTING-ORDER TAKE-PROFIT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                     # Places a real resting GTC limit sell at entry+TP on fill,
                     # captured by Polymarket the instant the bid touches it (no
                     # poll latency). The SL monitor tracks fills + cancels the
                     # order before any other exit and near settlement. Startup
                     # cancels all resting orders (orphan cleanup). When on, the
                     # polled TP is bypassed (resting order handles profit-take).
                     resting_tp_enabled:          bool  = False,
                     # â”€â”€ 2026-06-16: RESTING-TP RE-ENTRY (live test) â”€â”€â”€â”€â”€â”€â”€â”€â”€
                     # When a resting-TP order FULLY sells (whole position exits
                     # in profit), re-open the window for a fresh signal
                     # evaluation so the daemon can re-enter IF signals still
                     # align (re-buy the same side, or flip to the new side).
                     # Re-entry only ever happens AFTER a full sell â€” partial
                     # fills, SL/RRM exits, flips and settlement never re-open
                     # the window. Each re-entry re-runs the full gate stack, so
                     # a trade only fires when the signal qualifies again.
                     tp_reentry_enabled:          bool  = False,
                     # â”€â”€ 2026-06-16: STOP-LOSS RE-ENTRY (live test) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                     # Independent of tp_reentry_enabled. When a position is
                     # FULLY closed by an adverse exit (stop-loss / RRM reversal
                     # / futures fast-exit), re-open the window for a fresh
                     # signal evaluation so the daemon may re-enter (re-buy or
                     # flip) IF signals still align. Skipped on partial fills,
                     # on the hedge-held path, and when a smart-flip position
                     # was already opened. Re-runs the full gate stack.
                     sl_reentry_enabled:          bool  = False,
                     # â”€â”€ 2026-06-15: HIGH-PRICE CEILING TP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                     # For high-price "sure" entries (>= min_cents, e.g. 89), a
                     # +take_profit_cents target would exceed 99Â¢ and never
                     # place. Instead rest a sell near the ceiling (target_cents,
                     # e.g. 98) so these lock in the last few cents early rather
                     # than waiting for settlement. Price-based, ALL tiers, NOT
                     # perp-gated. Requires --resting-tp-enabled.
                     high_price_tp_enabled:       bool  = False,
                     high_price_tp_min_cents:     int   = 89,
                     high_price_tp_target_cents:  int   = 98,
                     # â”€â”€ 2026-06-27: HIGH-RISK TIGHT-TP (conditional) â”€â”€â”€â”€â”€â”€â”€â”€
                     # A high_conv winner that (a) drew down >= dd_cents below
                     # entry AND (b) entered <= dist_max% from strike is a
                     # last-minute-reversal risk. On those (and ONLY those),
                     # rest a tight sell at entry+highrisk_tp_cents to lock a
                     # small win on the recovery bounce instead of riding into a
                     # possible settlement reversal. Unlike the high-price
                     # ceiling this fires on <1% of trades, so it barely caps
                     # winners. Survives the late-disable (it's a profit price).
                     # Requires --resting-tp-enabled. Default OFF.
                     highrisk_tp_enabled:         bool  = False,
                     highrisk_tp_dd_cents:        int   = 12,
                     highrisk_tp_dist_max:        float = 0.35,
                     highrisk_tp_cents:           int   = 5,
                     # â”€â”€ 2026-06-12: PERP-MOMENTUM ENTRY VETO â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                     # Skip entries when the Polymarket perp 30s tape moves against
                     # the position at fire time (signed bp <= threshold).
                     # Fail-open when the perp feed is down.
                     perp_veto_enabled:        bool  = False,
                     perp_veto_m30s_threshold: float = -3.0,
                     # 2026-07-02: POLYMARKET BID-STABILITY ENTRY VETO
                     bid_stab_veto_enabled:    bool  = False,
                     bid_stab_lookback_s:      float = 60.0,
                     bid_stab_max_fade_cents:  int   = 2,
                     bid_stab_min_samples:     int   = 3,
                     bid_stab_burst_samples:   int   = 3,
                     bid_stab_burst_interval_s: float = 2.0,
                     # 2026-06-29: TAKER-FLOW ENTRY VETO (near-strike trap)
                     taker_flow_veto_enabled:    bool  = False,
                     taker_flow_veto_agg_min:    float = 0.90,
                     taker_flow_veto_dist_max:   float = 0.25,
                     taker_flow_veto_min_trades: int   = 5,
                     # â”€â”€ 2026-06-14: BOOK-SKEW ENTRY VETO (golden-only) â”€â”€â”€â”€â”€â”€
                     # Binance futures depth imbalance (signed toward trade
                     # side). Golden-only by default (hurts other tiers).
                     book_skew_veto_enabled:   bool  = False,
                     book_skew_threshold:      float = -0.15,
                     book_skew_golden_only:    bool  = True,
                     # â”€â”€ 2026-06-15: PERP-IMB DEEP VETO (all tiers) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                     perp_imb_veto_enabled:    bool  = False,
                     perp_imb_veto_threshold:  float = -0.50,
                     # â”€â”€ 2026-06-17: GOLDEN NEAR + HIGH-VOL VETO â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                     # Skip golden-band entries that are BOTH near the strike
                     # (|dist%| < dist_max) AND in elevated short-term vol
                     # (GK_vol >= gk_min). The near+high-vol bucket loses ~36%
                     # vs ~20% for far entries (10-day study); distance alone is
                     # vol-confounded (corr +0.49) so it gates on both. Golden
                     # only. Opt-in; transient re-poll.
                     golden_near_vol_veto_enabled: bool  = False,
                     golden_near_vol_dist_max:     float = 0.08,
                     golden_near_vol_gk_min:       float = 0.0020,
                     # â”€â”€ 2026-06-22: HARD ENTRY-PRICE FLOOR (all tiers) â”€â”€â”€
                     # Final pre-order veto: never enter below this price.
                     min_entry_price:              int   = 0,
                     # â”€â”€ 2026-06-17: MAX PER-TRADE CAPITAL CAP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                     # Hard $ ceiling on a single ticket (contracts x entry),
                     # all tiers. Bounds single-ticket risk; clips only the
                     # oversized monster tickets. 0 = off.
                     max_trade_usd:                float = 0.0,
                     # â”€â”€ 2026-06-15: SURE-TRADE EV WALK-UP OVERRIDE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                     # Relax the EV floor (to ev_override_floor) during the
                     # price walk on high-conviction trades where book_skew
                     # agrees, to win fills on "sure" markets. Tight gate.
                     ev_walkup_override_enabled: bool  = False,
                     ev_override_pwin_min:       float = 0.70,
                     ev_override_price_max:      int   = 85,
                     ev_override_book_skew_min:  float = 0.0,
                     ev_override_floor:          float = -0.10,
                     # â”€â”€ 2026-06-12: RRM LIVE EXIT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                     # When the reversal-risk score fires (strike breach +
                     # confluence), SELL via the proven SL sell path instead of
                     # only logging. Validated 06-12: 3 shadow saves (+$341)
                     # vs 1 winner-cut (-$0.77) incl. catching the -$417 loss
                     # at 53\u00a2. Min-contracts guard keeps 1-lot noise log-only.
                     rrm_exit_enabled:         bool  = False,
                     rrm_exit_min_score:       int   = 6,
                     rrm_exit_min_contracts:   int   = 25,
                     # â”€â”€ 2026-06-18: PREDICT-CROSS-EXIT (drift-aware) â”€â”€â”€â”€â”€â”€â”€â”€
                     # In the final pcross_max_mins, estimate P(contract ends
                     # OTM) from the perp oracle (distance + projected 30s
                     # adverse drift, scaled by realized vol). When it exceeds
                     # pcross_prob for pcross_confirm_polls consecutive polls,
                     # sell pre-emptively via the proven SL sell path. The
                     # keep-alive holds that path open down to
                     # pcross_keep_alive_mins so a confirmed reversal can still
                     # be cut in the last 2 min (the cents-SL stays gated to
                     # its original disarm). Default OFF.
                     predict_cross_exit_enabled: bool  = False,
                     pcross_prob:                float = 0.40,
                     pcross_max_mins:            float = 2.0,
                     pcross_confirm_polls:       int   = 2,
                     pcross_min_contracts:       int   = 25,
                     pcross_keep_alive_mins:     float = 0.4,
                     # â”€â”€ 2026-06-24: HOLD-TO-WIN (cancel resting TP on conviction) â”€
                     holdwin_enabled:            bool  = False,
                     holdwin_tiers:              tuple = ("golden", "standard"),
                     holdwin_min_profit_cents:   int   = 8,
                     holdwin_min_dist_pct:       float = 0.10,
                     holdwin_max_potm:           float = 0.20,
                     holdwin_rearm_dist_pct:     float = 0.05,
                     holdwin_rearm_potm:         float = 0.40,
                     holdwin_trail_cents:        int   = 8,
                     holdwin_min_gap:            float = 0.30):
    session = Session(
        bankroll,
        rolling_wr_enabled=rolling_wr_enabled,
        rolling_wr_window=rolling_wr_window,
        rolling_wr_threshold=rolling_wr_threshold,
        rolling_wr_timeout_mins=rolling_wr_timeout_mins,
        rolling_wr_defensive_tiers=rolling_wr_defensive_tiers,
        adaptive_br_enabled=adaptive_br_enabled,
        adaptive_br_reduced_frac=adaptive_br_reduced_frac,
        adaptive_br_loss_trigger_usd=adaptive_br_loss_trigger_usd,
        adaptive_br_wr_trigger=adaptive_br_wr_trigger,
        adaptive_br_wr_window_h=adaptive_br_wr_window_h,
        adaptive_br_wr_min_trades=adaptive_br_wr_min_trades,
        adaptive_br_recover_wr=adaptive_br_recover_wr,
        adaptive_br_recover_window=adaptive_br_recover_window,
        adaptive_br_recover_min_wins=adaptive_br_recover_min_wins,
    )
    log.info("=" * 65)
    log.info(f"  Polymarket BTC 15m Daemon (ChainVector)  |  dry_run={dry_run}  |  bankroll=${bankroll:.0f}")
    if term_prob_disabled:
        log.info("  Terminal probability: DISABLED")
    else:
        log.info(f"  Terminal probability: enabled (logs only"
                 f"{', GATE RELAX' if term_prob_relax else ''})  "
                 f"@ 12/7/3 min checkpoints")
    if futures_lead_disabled:
        log.info("  ChainVector futures lead: DISABLED")
    else:
        log.info(f"  ChainVector futures lead: enabled  (binance_futures via /momentum, "
                 f"lookback={futures_lead_lookback_s:.0f}s, "
                 f"veto if futures move > {futures_lead_veto_bps:.0f}bps against direction)")
    log.info(f"  CV momentum boost: {'ENABLED' if cv_mom_boost_enabled else 'DISABLED'}  "
             f"(combined_p nudge \u2264\u00b1{cv_mom_boost_weight:.2f}, "
             f"saturates ~agg\u00b1{cv_mom_boost_scale:.0f})")
    log.info(f"  CV momentum veto: {'ENABLED' if cv_mom_veto_enabled else 'DISABLED'}  "
             f"(block when signed agg \u2264 -{cv_mom_veto_score:.0f} with breadth "
             f"\u2265{cv_mom_veto_breadth:.2f} against)")
    log.info(f"  CV probability floor veto: {'ENABLED' if cv_prob_veto_enabled else 'DISABLED'}  "
             f"(block when ensemble P(our side) \u2264 {cv_prob_veto_max:.2f} at exact TTE)")
    log.info(f"  CV quote-stability veto: {'ENABLED' if cv_stab_veto_enabled else 'DISABLED'}  "
             f"(block when Polymarket repricing momentum \u2264 -{cv_stab_mom_against:.0f} against side)")
    log.info(f"  CV cascade-risk veto: {'ENABLED' if cv_cascade_veto_enabled else 'DISABLED'}  "
             f"(block when risk \u2265 {cv_cascade_veto_score:.0f} and cascade side against)")
    log.info(f"  CV recorded signals: edge + regime/volatility/risk-index context "
             f"logged per fire (promotable after outcome validation)")
    if consensus_veto_enabled:
        log.info(f"  Consensus veto: ENABLED  (binance_futures+OKX both must show "
                 f"|move|â‰¥{consensus_min_move_pct:.3f}% opposing direction)")
        if consensus_smart_bypass:
            log.info(f"    Smart bypass: ENABLED â€” veto skipped when ANY of:")
            log.info(f"      (1) longer-window futures (â‰¥{consensus_long_window_s:.0f}s) "
                     f"favors trade by â‰¥{consensus_long_min_pct:.3f}%  (6s blip is noise)")
            log.info(f"      (2) recent 5m bars sum favors trade by â‰¥{consensus_5m_favor_pct:.3f}%  "
                     f"(30-min trend overrides blip)")
            log.info(f"      (3) |dist_pct|â‰¥{consensus_far_dist_pct:.3f}% AND "
                     f"both moves <{consensus_far_max_move_pct:.3f}%  "
                     f"(BTC too far for tiny wiggle)")
        else:
            log.info(f"    Smart bypass: DISABLED  (strict 6s consensus)")
    else:
        log.info("  Consensus veto: DISABLED")
    log.info(f"  HIGH-CONV sizing boost: frac {high_conv_confirmed_frac:.2f} "
             f"({high_conv_confirmed_frac/0.05:.1f}x baseline) when "
             f"binance_futures+OKX consensus confirms direction")
    if hc_block_on_split_externals:
        log.info(f"  HIGH-CONV split-externals block: ENABLED â€” rejects HIGH-CONV "
                 f"tier when one external opposes the trade AND the other doesn't "
                 f"(STANDARD/strong/late-* tiers unaffected)")
    else:
        log.info(f"  HIGH-CONV split-externals block: DISABLED")
    if standard_confirmed_boost > 1.0:
        log.info(f"  STANDARD-confirmed boost: {standard_confirmed_boost:.1f}x Kelly fraction "
                 f"when active tier is NOT high-conv AND both externals confirm direction")
    else:
        log.info(f"  STANDARD-confirmed boost: disabled (multiplier {standard_confirmed_boost:.1f})")
    if last_bar_adverse_threshold > 0:
        log.info(f"  Last-bar adverse gate: ENABLED â€” block trades when LAST 5m bar opposes "
                 f"direction by >{last_bar_adverse_threshold:.2f}%. Catches late-stage "
                 f"reversal patterns the multi-bar Markov misses. EXTREME-SIGNAL BYPASS: "
                 f"skip when gapâ‰¥0.30 AND TP-direction â‰¥{high_conv_tp_strong:.2f} "
                 f"(or â‰¤{1-high_conv_tp_strong:.2f}) â€” in lock-in markets the very 5m bar "
                 f"that created the consensus often trips this gate.")
    else:
        log.info(f"  Last-bar adverse gate: DISABLED")
    if orderbook_signal_enabled:
        log.info(f"  Polymarket orderbook signal: RECORDING (audit-only, 5s cache). "
                 f"Captures depth, spread, imbalance per poll for retrospective analysis.")
    if trade_flow_signal_enabled:
        log.info(f"  Polymarket trade-flow signal: RECORDING (audit-only, last {trade_flow_lookback_n} "
                 f"trades per poll). Captures taker aggression direction.")
    if taker_flow_veto_enabled:
        log.info(f"  TAKER-FLOW veto: ENABLED - block entry when >= "
                 f"{taker_flow_veto_agg_min*100:.0f}% of recent taker volume aggresses "
                 f"AGAINST the bet AND |dist|<{taker_flow_veto_dist_max:.2f} from strike "
                 f"(min {taker_flow_veto_min_trades} trades). Near-strike reversal-trap guard.")
    if sl_enabled:
        agg_str = " (aggressive sell: fill against any remaining bid)" if sl_aggressive_sell else ""
        log.info(f"  Stop-loss: ENABLED  exit if {sl_trigger_mode}-price drops "
                 f"{sl_loss_cents}Â¢ below entry (STANDARD/strong/late-dir), "
                 f"{sl_loss_cents_high_conv}Â¢ (HIGH-CONV + LATE-SURE â€” high-price tiers) â€” "
                 f"grace={sl_grace_mins:.1f}min after entry, "
                 f"disabled in final {sl_disable_late_mins:.1f}min, "
                 f"poll every {sl_poll_interval_s:.1f}s standard / "
                 f"{sl_poll_interval_hc_s:.1f}s HC{agg_str}")
        _ds_gs = f"${max_loss_per_trade:.0f}" if max_loss_per_trade > 0 else "OFF"
        _ds_hc = f"${max_loss_per_trade_high_conv:.0f}" if max_loss_per_trade_high_conv > 0 else "OFF"
        log.info(f"  Per-trade DOLLAR stop: golden/standard={_ds_gs}, high_conv={_ds_hc} "
                 f"(fires via SL path until pcross disarm floor)")
        if sl_reentry_enabled:
            log.warning("  Stop-loss RE-ENTRY: ENABLED \u2014 after an adverse "
                        "FULL exit (stop-loss / RRM reversal / futures fast-"
                        "exit), the window re-opens for a fresh signal "
                        "evaluation; the daemon may re-enter (re-buy or flip) "
                        "IF signals still align. Skipped on partial fills, "
                        "hedge-held positions, and when a smart-flip already "
                        "fired. Independent of TP re-entry. LIVE-TEST FEATURE "
                        "\u2014 re-evaluates immediately after a losing exit.")
        if futures_fast_exit_enabled:
            log.info(f"  Futures FAST-EXIT: ENABLED  exit immediately if lead futures "
                     f"moved >{futures_fast_exit_threshold_pct:.2f}% against position over "
                     f"{futures_fast_exit_window_s:.0f}s (sanity-reject |move|>{futures_fast_exit_sanity_max_pct:.1f}%)")
        else:
            log.info("  Futures FAST-EXIT: DISABLED")
    else:
        log.info("  Stop-loss: DISABLED â€” positions hold until settlement")
    if hedge_enabled:
        log.info(f"  Safety HEDGE: ENABLED  tiers={list(hedge_tiers)} "
                 f"entry={hedge_min_yes_entry}-{hedge_max_yes_entry}\u00a2 "
                 f"max_no={hedge_max_no_cost}\u00a2  no_settle_assumed={hedge_no_settle_assumed:.2f} "
                 f"capital_mult\u2264{hedge_max_capital_mult:.1f}x  widened_sl={hedge_widened_sl_cents}\u00a2")
    else:
        log.info("  Safety HEDGE: DISABLED")
    if hurst_tp_veto_enabled:
        log.info(f"  Hurst+TP veto: ENABLED  tiers={list(hurst_tp_veto_tiers)} "
                 f"\u2014 block when Hurst\u2265{hurst_tp_veto_min_hurst:.2f} AND "
                 f"|TP\u2212Markov| adverse \u2265{hurst_tp_veto_min_diff:.2f}")
    else:
        log.info("  Hurst+TP veto: DISABLED")
    if max_adverse_bar_veto_enabled:
        log.info(f"  Max-adverse-bar veto: ENABLED  tiers={list(max_adverse_bar_veto_tiers)} "
                 f"\u2014 block direction X if any of last 5m bars moved "
                 f"\u2265{max_adverse_bar_veto_pct:.3f}% against X "
                 f"(catches dead-cat-bounce / falling-knife)")
    else:
        log.info("  Max-adverse-bar veto: DISABLED")
    if cum_adverse_momentum_veto_enabled:
        log.info(f"  Cumulative-adverse-momentum veto: ENABLED  "
                 f"tiers={list(cum_adverse_momentum_veto_tiers)} "
                 f"\u2014 block direction X when sum(recent_5m_pct) adverse to X "
                 f"\u2265{cum_adverse_momentum_veto_pct:.3f}% "
                 f"(catches slow steady drift against trade)")
    else:
        log.info("  Cumulative-adverse-momentum veto: DISABLED")
    if hc_low_hurst_veto_enabled:
        log.info(f"  Low-Hurst HC veto (Tier 4): ENABLED  "
                 f"\u2014 block HC when Hurst<{hc_low_hurst_threshold:.2f} AND "
                 f"|Markov-0.5|\u2265{hc_low_hurst_markov_extremity:.2f}  "
                 f"(mean-reverting regime + extreme directional = trap)")
    else:
        log.info("  Low-Hurst HC veto (Tier 4): DISABLED")
    if hc_dist_min > 0:
        log.info(f"  HC distance floor: ENABLED  "
                 f"\u2014 require |dist_pct|\u2265{hc_dist_min:.3f}% for HC "
                 f"(blocks near-strike HC trades; backtest: 13/13 wins at 0.25%)")
    else:
        log.info("  HC distance floor: DISABLED (no minimum |dist_pct|)")
    if rolling_wr_enabled:
        log.info(f"  Rolling-WR throttle: ENABLED  "
                 f"\u2014 if last {rolling_wr_window} trades WR \u2264 "
                 f"{rolling_wr_threshold*100:.0f}%, block tiers "
                 f"{sorted(set(rolling_wr_defensive_tiers))} until next win "
                 f"or {rolling_wr_timeout_mins:.0f} min timeout")
    else:
        log.info("  Rolling-WR throttle: DISABLED")
    if adaptive_br_enabled:
        log.info(f"  Adaptive bankroll: ENABLED  "
                 f"\u2014 REDUCE to {adaptive_br_reduced_frac*100:.0f}% sizing on "
                 f"drawdown \u2265${adaptive_br_loss_trigger_usd:.0f} OR "
                 f"{adaptive_br_wr_window_h:.0f}h WR < {adaptive_br_wr_trigger*100:.0f}% "
                 f"(\u2265{adaptive_br_wr_min_trades} trades); RECOVER on "
                 f"\u2265{adaptive_br_recover_min_wins} wins and last-"
                 f"{adaptive_br_recover_window} WR \u2265 {adaptive_br_recover_wr*100:.0f}%. "
                 f"Daily hard cap extended to $700. Resets on restart.")
    else:
        log.info("  Adaptive bankroll: DISABLED")
    if patient_topup_enabled:
        _dk = " [dynamic-Kelly: grows toward fresh Kelly when EV turns +]" if patient_topup_dynamic_kelly else ""
        log.info(f"  Patient top-up: ENABLED  \u2014 complete under-filled positions "
                 f"when ask improves below blended entry; full fresh-signal + EV "
                 f"re-check each attempt (every {patient_topup_interval_s:.0f}s, "
                 f"stops <{patient_topup_min_mins:.1f}min left){_dk}")
    else:
        log.info("  Patient top-up: DISABLED")
    if holdwin_enabled:
        log.warning(
            f"  Hold-to-win: ENABLED â€” tiers={','.join(holdwin_tiers)}, "
            f"entry-arm gap>={holdwin_min_gap} & cushion>={holdwin_min_dist_pct}%; "
            f"cancel TP when profit>=+{holdwin_min_profit_cents}Â¢ & "
            f"dist>={holdwin_min_dist_pct}% & P(OTM)<={holdwin_max_potm}; "
            f"re-arm on dist<{holdwin_rearm_dist_pct}% / P(OTM)>="
            f"{holdwin_rearm_potm} / trail {holdwin_trail_cents}Â¢. "
            f"$-stop/pcross/RRM remain active as the reversal backstop.")
    if take_profit_enabled:
        _scope = (f"perp-confirmed only (entry m30s>{take_profit_perp_min:+.1f})"
                  if take_profit_perp_confirmed_only else "ALL trades")
        _mode = "RESTING limit order" if resting_tp_enabled else "polled (SL monitor)"
        log.info(f"  Take-profit: ENABLED  \u2014 exit at entry +{take_profit_cents}\u00a2 "
                 f"[{_scope}] via {_mode}. Backtest: +10\u00a2 on perp-confirmed "
                 f"was +EV; on non-confirmed it is -EV (hence the gate).")
        if resting_tp_enabled:
            log.warning("  Resting-TP: LIVE order management active \u2014 cancels "
                        "before other exits + near settlement; startup orphan "
                        "cleanup done. VERIFY WITH SMALL BANKROLL FIRST.")
            if tp_reentry_enabled:
                log.warning("  Resting-TP RE-ENTRY: ENABLED \u2014 after a resting-TP "
                            "order FULLY sells, the window re-opens for a fresh "
                            "signal evaluation; the daemon may re-enter (re-buy "
                            "or flip) IF signals still align. Full sells only; "
                            "partial fills / SL / RRM / flip / settlement never "
                            "re-open the window. LIVE-TEST FEATURE.")
            if high_price_tp_enabled:
                log.info(f"  High-price ceiling TP: entries \u2265"
                         f"{high_price_tp_min_cents}\u00a2 rest a sell at "
                         f"{high_price_tp_target_cents}\u00a2 (all tiers, not "
                         f"perp-gated).")
    elif resting_tp_enabled and high_price_tp_enabled:
        # Ceiling TP works without the +NÂ¢ TP being enabled.
        log.info(f"  Take-profit: ceiling-only \u2014 entries \u2265"
                 f"{high_price_tp_min_cents}\u00a2 rest a sell at "
                 f"{high_price_tp_target_cents}\u00a2 (resting order).")
    else:
        log.info("  Take-profit: DISABLED")
    if highrisk_tp_enabled and resting_tp_enabled:
        log.info(f"  High-risk tight TP: ENABLED \u2014 high_conv trades that draw "
                 f"down \u2265{highrisk_tp_dd_cents}\u00a2 AND entered \u2264"
                 f"{highrisk_tp_dist_max:.2f}% from strike rest a sell at "
                 f"entry+{highrisk_tp_cents}\u00a2 (kept alive to close).")
    elif highrisk_tp_enabled and not resting_tp_enabled:
        log.info("  High-risk tight TP: requested but INERT (needs --resting-tp-enabled)")
    else:
        log.info("  High-risk tight TP: DISABLED")
    if perp_veto_enabled:
        log.info(f"  Perp-momentum entry veto: ENABLED  \u2014 skip entry when perp "
                 f"30s tape \u2264 {perp_veto_m30s_threshold:+.1f}bp against the "
                 f"position (fail-open if perp feed down)")
    else:
        log.info("  Perp-momentum entry veto: DISABLED (log-only verdicts still recorded)")
    if bid_stab_veto_enabled:
        log.info(f"  Bid-stability entry veto: ENABLED  â€” skip entry unless our "
                 f"side's Polymarket bid is stable/rising over the last "
                 f"{bid_stab_lookback_s:.0f}s (net >= 0 and <= "
                 f"{bid_stab_max_fade_cents}c off its peak; >= "
                 f"{bid_stab_min_samples} samples, fail-open; transient re-poll)")
    else:
        log.info("  Bid-stability entry veto: DISABLED")
    if bid_stab_veto_enabled and bid_stab_burst_samples > 0:
        log.info(f"  Bid-stability BURST confirm: {bid_stab_burst_samples} live "
                 f"re-samples @ {bid_stab_burst_interval_s:.1f}s at the entry "
                 f"moment (~{bid_stab_burst_samples * bid_stab_burst_interval_s:.0f}s "
                 f"real-time window) â€” any downtick vetoes + re-polls")
    if book_skew_veto_enabled:
        _scope = "golden 65-73\u00a2 only" if book_skew_golden_only else "ALL tiers"
        log.info(f"  Book-skew entry veto: ENABLED ({_scope})  \u2014 skip entry when "
                 f"Binance futures depth skew \u2264 {book_skew_threshold:+.2f} against "
                 f"the position (fail-open if ChainVector feed down)")
    else:
        log.info("  Book-skew entry veto: DISABLED (log-only verdicts still recorded)")
    if perp_imb_veto_enabled:
        log.info(f"  Perp-imb deep veto: ENABLED (all tiers)  \u2014 skip entry when "
                 f"Polymarket perp book imbalance \u2264 {perp_imb_veto_threshold:+.2f} "
                 f"against the position (transient re-poll; fail-open if perp down)")
    else:
        log.info("  Perp-imb deep veto: DISABLED (log-only verdicts still recorded)")
    if golden_near_vol_veto_enabled:
        log.info(f"  Golden near+high-vol veto: ENABLED  \u2014 skip golden entries "
                 f"with |dist%| < {golden_near_vol_dist_max:.3f}% AND GK_vol "
                 f"\u2265 {golden_near_vol_gk_min:.5f} (knife-edge whip risk; "
                 f"transient re-poll; fail-open if dist/GK missing).")
    else:
        log.info("  Golden near+high-vol veto: DISABLED")
    if min_entry_price > 0:
        log.warning(f"  Entry-price FLOOR: ENABLED â€” never enter below "
                    f"{min_entry_price}Â¢ (all tiers). Keeps >=89Â¢ deep/high-conv "
                    f"band; cuts golden/standard cheap entries.")
    else:
        log.info("  Entry-price floor: DISABLED")
    if max_trade_usd > 0:
        log.warning(f"  Max per-trade capital: ${max_trade_usd:.0f} \u2014 single-ticket "
                    f"size capped (contracts x entry), all tiers. Clips oversized "
                    f"monster tickets; small/medium book unaffected.")
    else:
        log.info("  Max per-trade capital: UNCAPPED")
    if ev_walkup_override_enabled:
        log.info(f"  Sure-trade EV walk-up override: ENABLED  \u2014 relax walk floor "
                 f"to ${ev_override_floor:+.2f}/c when p_win\u2265{ev_override_pwin_min:.2f} "
                 f"AND book_skew\u2265{ev_override_book_skew_min:+.2f} AND price\u2264"
                 f"{ev_override_price_max}\u00a2 (win fills on 'sure' book-confirmed markets)")
    else:
        log.info("  Sure-trade EV walk-up override: DISABLED")
    if rrm_exit_enabled:
        log.info(f"  RRM live exit: ENABLED  \u2014 sell on reversal score "
                 f"\u2265{rrm_exit_min_score}/10 (strike breach + confluence), "
                 f"positions \u2265{rrm_exit_min_contracts}c only; smaller fires stay "
                 f"log-only. Late-window (<2.5min) coverage stays log-only.")
    else:
        log.info("  RRM live exit: DISABLED (reversal monitor logs would-exits only)")
    if predict_cross_exit_enabled:
        log.warning(
            f"  Predict-cross-exit: ENABLED \u2014 sell when P(end OTM)\u2265"
            f"{pcross_prob:.2f} for {pcross_confirm_polls} consecutive polls in "
            f"the final {pcross_max_mins:.1f}min (drift-aware perp oracle); "
            f"positions \u2265{pcross_min_contracts}c only (smaller stay log-only). "
            f"SL sell path kept alive to {pcross_keep_alive_mins:.1f}min; "
            f"cents-SL disarm unchanged ({sl_disable_late_mins:.1f}min).")
    else:
        log.info("  Predict-cross-exit: DISABLED")
    _yes_cap_disp = standard_price_cap_yes if standard_price_cap_yes is not None else MAX_ENTRY_PRICE_YES
    _no_cap_disp  = standard_price_cap_no  if standard_price_cap_no  is not None else MAX_ENTRY_PRICE_NO
    _yes_default = MAX_ENTRY_PRICE_YES
    _no_default  = MAX_ENTRY_PRICE_NO
    if _yes_cap_disp != _yes_default or _no_cap_disp != _no_default:
        log.info(f"  Standard price caps: YES\u2264{_yes_cap_disp}\u00a2  NO\u2264{_no_cap_disp}\u00a2  "
                 f"(defaults: YES\u2264{_yes_default}\u00a2  NO\u2264{_no_default}\u00a2)")
    else:
        log.info(f"  Standard price caps: YES\u2264{_yes_cap_disp}\u00a2  NO\u2264{_no_cap_disp}\u00a2  (defaults)")
    if fade_bounce_enabled:
        log.info(f"  FADE-BOUNCE tier: ENABLED  (dual-entry on bid dip)")
        log.info(f"    NO side: bid \u2208 [{fade_bounce_no_ask_min},{fade_bounce_no_ask_max}]\u00a2 AND "
                 f"Markov\u2264{fade_bounce_markov_no_max:.2f} AND "
                 f"dist\u2264-{fade_bounce_dist_min:.3f}%")
        if fade_bounce_yes_side_enabled:
            log.info(f"    YES side: bid \u2208 [{fade_bounce_no_ask_min},{fade_bounce_no_ask_max}]\u00a2 AND "
                     f"Markov\u2265{fade_bounce_markov_yes_min:.2f} AND "
                     f"dist\u2265+{fade_bounce_dist_min:.3f}%")
        else:
            log.info(f"    YES side: DISABLED (data showed 0% WR in this band)")
        log.info(f"    Common gates: Hurst\u2265{fade_bounce_hurst_min:.2f}, "
                 f"mins\u2208[{fade_bounce_min_mins:.1f},{fade_bounce_max_mins:.1f}]")
        log.info(f"    Sizing: min_stake={fade_bounce_min_stake_pct:.3f}\u00d7bankroll, "
                 f"kelly_frac={fade_bounce_kelly_frac:.3f}, "
                 f"max_capital=${fade_bounce_max_capital_usd:.0f}")
        log.info(f"    SL: {fade_bounce_sl_cents}\u00a2 below entry (lighter than other tiers)")
    else:
        log.info("  FADE-BOUNCE tier: DISABLED")
    if smart_flip_enabled:
        fut_tag = (f" + futures_confirm\u2265{smart_flip_futures_confirm_pct:.3f}%/"
                   f"{smart_flip_futures_window_s:.0f}s"
                   if smart_flip_require_futures_confirm else "")
        log.info(f"  Smart FLIP: ENABLED  tiers={list(smart_flip_tiers)} "
                 f"opp_band={smart_flip_min_opp_entry}-{smart_flip_max_opp_entry}\u00a2 "
                 f"recovery={smart_flip_recovery_ratio:.0%}  "
                 f"flip_sl={smart_flip_sl_cents}\u00a2  "
                 f"target={smart_flip_sell_target}\u00a2  "
                 f"trail={smart_flip_trail_cents}\u00a2  "
                 f"max_cap=${smart_flip_max_capital_usd:.0f}  "
                 f"retries={smart_flip_retry_attempts}\u00d7{smart_flip_retry_sleep_s:.0f}s"
                 f"{fut_tag}")
        if dry_run:
            log.info("  Smart FLIP: DRY-RUN simulation enabled \u2014 each trade "
                     "logs hypothetical flip decision (if primary SL had fired). "
                     "Writes FLIP_ATTACH_SIM / FLIP_SKIP_SIM audit records.")
    else:
        log.info("  Smart FLIP: DISABLED")
    if okx_disabled:
        log.info("  OKX lead view: DISABLED")
    else:
        log.info(f"  OKX lead view: enabled (ChainVector /momentum okx venue, poll every "
                 f"{okx_poll_interval_s:.1f}s, snapshotted in audit on every poll)")
    if ev_gate:
        log.info(f"  EV gate: ENABLED  (floor=${ev_floor:.2f}/c, ceiling={ev_ceiling}Â¢, "
                 f"TP weight={ev_tp_weight:.2f})")
        strong_hurst_tag = " + HURST-BYPASS" if strong_floor_hurst_bypass else ""
        log.info(f"           strong-floor=${ev_strong_floor:+.2f}/c when gapâ‰¥{ev_strong_gap_min:.2f} "
                 f"AND priceâ‰¤{ev_strong_price_max}Â¢ AND TPâ‰¥{ev_strong_tp_min:.2f}/â‰¤{1-ev_strong_tp_min:.2f} (directional) "
                 f"AND mins_leftâ‰¤{ev_strong_max_mins:.1f} AND recent 6-bar momentum not >{ev_strong_max_adverse_momentum:.2f}% adverse"
                 f"{strong_hurst_tag}")
        late_sure_vol_tag = (" + VOL-BYPASS" if late_sure_vol_bypass else "")
        ob_lockin_tag = ""
        if orderbook_lockin_enabled:
            ob_lockin_tag = (f"  [ORDERBOOK-LOCKIN: when spread\u2264"
                             f"{orderbook_lockin_spread_max}\u00a2 AND top-bid\u2265"
                             f"{orderbook_lockin_price_min}\u00a2 AND gap\u2265"
                             f"{orderbook_lockin_gap_min:.2f}: HC+LATE-SURE bypass TP, "
                             f"use Markov-only p_win, "
                             f"LATE-SURE cap rises to {late_window_price_max_lockin}\u00a2, "
                             f"HC floor relaxes to ${hc_lockin_ev_floor:+.2f}/c, "
                             f"HC kelly=0 \u2192 min-stake {hc_lockin_min_stake_pct*100:.1f}%]")
        log.info(f"           LATE-WINDOW SURE: floor=${late_window_ev_floor:+.2f}/c, "
                 f"priceâ‰¤{late_window_price_max}Â¢ "
                 f"when mins_leftâ‰¤{late_window_mins:.1f} AND TPâ‰¥{late_window_min_tp:.2f} "
                 f"AND Markov+momentum agree{late_sure_vol_tag}{ob_lockin_tag}")
        log.info(f"           HIGH-CONVICTION: floor=${high_conv_ev_floor:+.3f}/c (positive-EV only), "
                 f"priceâ‰¤{high_conv_price_max}Â¢, mins_leftâ‰¤{high_conv_max_mins:.1f} "
                 f"when gapâ‰¥{high_conv_gap_min:.2f} AND persistâ‰¥{high_conv_persist_min:.2f} "
                 f"AND TP extremeâ‰¥{high_conv_tp_strong:.2f}  (pure Kelly; can bypass timing)")
        if high_conv_vol_bypass:
            log.info(f"           HIGH-CONV vol bypass: enabled when "
                     f"|dist from strike|â‰¥{high_conv_vol_bypass_distance:.2f}% AND "
                     f"(5m returnâ‰¥{high_conv_vol_bypass_momentum:.2f}% in dir OR "
                     f"|dist|â‰¥{high_conv_vol_bypass_strong_distance:.2f}% strong)")
        else:
            log.info(f"           HIGH-CONV vol bypass: DISABLED (vol stays hard gate)")
        if late_dir_enabled:
            log.info(f"           LATE-DIR (late-window directional): floor=${late_dir_ev_floor:+.2f}/c, "
                     f"priceâ‰¤{late_dir_price_max}Â¢ when mins<{late_dir_mins:.1f}, gapâ‰¥{late_dir_gap_min:.2f}, "
                     f"persistâ‰¥{late_dir_persist_min:.2f}, TP confirms direction, "
                     f"|dist|â‰¥{late_dir_strong_distance:.2f}% (or momentum confirms)")
        else:
            log.info(f"           LATE-DIR tier: DISABLED")
        log.info(f"           Min stake on negative-EV overrides: "
                 f"strong={strong_floor_min_stake_pct*100:.2f}%, "
                 f"late-sure={late_sure_min_stake_pct*100:.2f}% of bankroll")
        if standard_min_entry_contracts > 1:
            log.info(f"           Standard min entry: \u2265"
                     f"{standard_min_entry_contracts}c (smaller standard sizes "
                     f"skip + keep window open for a better/high-conv entry)")
        _gx = []
        if (golden_price_lo, golden_price_hi) != (65, 73):
            _gx.append(f"band {golden_price_lo}-{golden_price_hi}\u00a2")
        if golden_no_dist:  _gx.append("no-dist")
        if golden_no_hurst: _gx.append("no-hurst")
        if _gx:
            log.info(f"           Golden-zone EXPANDED: {', '.join(_gx)} "
                     f"(backtest: ~2x volume, 80%+ WR held-to-settle)")
        if okx_boost_enabled:
            log.info(f"           OKX-momentum boost: ON (weight \u00b1"
                     f"{okx_boost_weight:.2f}, scale {okx_boost_scale:.3f}) "
                     f"\u2014 nudges combined_p toward OKX 6s momentum. A/B live.")
    else:
        log.info("  EV gate: disabled (flat 72Â¢ YES / 65Â¢ NO price caps in effect)")
    if retry_walk_cents > 0:
        log.info(f"  Order retry: walk up to +{retry_walk_cents}Â¢ on zero-fill "
                 f"(stops on fill, EV-floor breach, or ceiling)")
    else:
        log.info(f"  Order retry: disabled (single attempt at limit price only)")
    if order_lead_cents > 0:
        log.info(f"  Order lead: submit initial order at +{order_lead_cents}Â¢ above "
                 f"observed price (EV-checked vs strong-floor)")
    else:
        log.info(f"  Order lead: disabled (initial order at observed orderbook ask)")
    log.info(f"  Per-window NO_FILL retries: up to {max_window_fill_attempts}, "
             f"sleeping {refill_retry_sleep_s:.0f}s between attempts")
    log.info("=" * 65)
    _log_active_params()

    # Per-window state for terminal-probability cache. Each entry stores the
    # most recent TP result keyed by window_id; the `_computed_at_unix` field
    # drives the 60-second refresh logic.
    window_tp_cache: dict[str, dict] = {}

    # Per-window count of failed IOC submissions (zero-fill / chunked + walked).
    # If signal still approves on next poll, we re-attempt â€” but bounded by
    # max_window_fill_attempts to avoid spamming Polymarket if the orderbook is
    # persistently moving away from us.
    window_no_fills: dict[str, int] = {}
    # 2026-06-12: PATIENT TOP-UP â€” track the LARGEST intended size across all
    # fire attempts in a window. When a later attempt fills smaller (price ran
    # away, Kelly shrank), the top-up watcher keeps trying to reach this size
    # at EV-valid prices for the rest of the window.
    window_max_intent: dict[str, int] = {}

    # Structured audit log â€” one JSONL file per UTC day in logs/audit_*.jsonl
    audit = AuditLogger(_log_dir)
    log.info(f"  Audit log: {_log_dir}/audit_*.jsonl")
    session._audit = audit   # adaptive-bankroll transition records

    # Captured daemon configuration for inclusion in every audit row
    audit_params = {
        "term_prob_relax":      term_prob_relax,
        "futures_lead_enabled": not futures_lead_disabled,
        "futures_lead_lookback_s": futures_lead_lookback_s,
        "futures_lead_veto_bps":   futures_lead_veto_bps,
        "cv_mom_boost_enabled":  cv_mom_boost_enabled,
        "cv_mom_boost_weight":   cv_mom_boost_weight,
        "cv_mom_boost_scale":    cv_mom_boost_scale,
        "cv_mom_veto_enabled":   cv_mom_veto_enabled,
        "cv_mom_veto_score":     cv_mom_veto_score,
        "cv_mom_veto_breadth":   cv_mom_veto_breadth,
        "cv_prob_veto_enabled":  cv_prob_veto_enabled,
        "cv_prob_veto_max":      cv_prob_veto_max,
        "cv_stab_veto_enabled":  cv_stab_veto_enabled,
        "cv_stab_mom_against":   cv_stab_mom_against,
        "cv_cascade_veto_enabled": cv_cascade_veto_enabled,
        "cv_cascade_veto_score":   cv_cascade_veto_score,
        "consensus_veto_enabled":   consensus_veto_enabled,
        "consensus_min_move_pct":   consensus_min_move_pct,
        "consensus_okx_lookback_s": consensus_okx_lookback_s,
        "consensus_smart_bypass":     consensus_smart_bypass,
        "consensus_long_window_s":    consensus_long_window_s,
        "consensus_long_min_pct":     consensus_long_min_pct,
        "consensus_5m_favor_pct":     consensus_5m_favor_pct,
        "consensus_far_dist_pct":     consensus_far_dist_pct,
        "consensus_far_max_move_pct": consensus_far_max_move_pct,
        "high_conv_confirmed_frac": high_conv_confirmed_frac,
        "hc_block_on_split_externals": hc_block_on_split_externals,
        "standard_confirmed_boost":   standard_confirmed_boost,
        "sl_enabled":               sl_enabled,
        "sl_loss_cents":            sl_loss_cents,
        "sl_loss_cents_high_conv":  sl_loss_cents_high_conv,
        "max_loss_per_trade":       max_loss_per_trade,
        "max_loss_per_trade_high_conv": max_loss_per_trade_high_conv,
        "sl_grace_mins":            sl_grace_mins,
        "sl_disable_late_mins":     sl_disable_late_mins,
        "sl_poll_interval_s":       sl_poll_interval_s,
        "sl_poll_interval_hc_s":    sl_poll_interval_hc_s,
        "futures_fast_exit_enabled":     futures_fast_exit_enabled,
        "futures_fast_exit_window_s":    futures_fast_exit_window_s,
        "futures_fast_exit_threshold_pct": futures_fast_exit_threshold_pct,
        "futures_fast_exit_sanity_max_pct": futures_fast_exit_sanity_max_pct,
        "hedge_enabled":                 hedge_enabled,
        "hedge_tiers":                   list(hedge_tiers),
        "hedge_min_yes_entry":           hedge_min_yes_entry,
        "hedge_max_yes_entry":           hedge_max_yes_entry,
        "hedge_max_no_cost":             hedge_max_no_cost,
        "hedge_no_settle_assumed":       hedge_no_settle_assumed,
        "hedge_max_capital_mult":        hedge_max_capital_mult,
        "hedge_widened_sl_cents":        hedge_widened_sl_cents,
        "hedge_no_sell_target":          hedge_no_sell_target,
        "hedge_no_sell_trail":           hedge_no_sell_trail,
        "hedge_post_sl_poll_s":          hedge_post_sl_poll_s,
        "smart_flip_enabled":            smart_flip_enabled,
        "smart_flip_tiers":              list(smart_flip_tiers),
        "smart_flip_min_opp_entry":      smart_flip_min_opp_entry,
        "smart_flip_max_opp_entry":      smart_flip_max_opp_entry,
        "smart_flip_recovery_ratio":     smart_flip_recovery_ratio,
        "smart_flip_sl_cents":           smart_flip_sl_cents,
        "smart_flip_sell_target":        smart_flip_sell_target,
        "smart_flip_trail_cents":        smart_flip_trail_cents,
        "smart_flip_max_capital_usd":    smart_flip_max_capital_usd,
        "smart_flip_min_mins_remaining": smart_flip_min_mins_remaining,
        "smart_flip_require_futures_confirm": smart_flip_require_futures_confirm,
        "smart_flip_futures_confirm_pct": smart_flip_futures_confirm_pct,
        "smart_flip_retry_attempts":     smart_flip_retry_attempts,
        "smart_flip_retry_sleep_s":      smart_flip_retry_sleep_s,
        "hurst_tp_veto_enabled":         hurst_tp_veto_enabled,
        "hurst_tp_veto_min_hurst":       hurst_tp_veto_min_hurst,
        "hurst_tp_veto_min_diff":        hurst_tp_veto_min_diff,
        "hurst_tp_veto_tiers":           list(hurst_tp_veto_tiers),
        "max_adverse_bar_veto_enabled":  max_adverse_bar_veto_enabled,
        "max_adverse_bar_veto_pct":      max_adverse_bar_veto_pct,
        "max_adverse_bar_veto_tiers":    list(max_adverse_bar_veto_tiers),
        "cum_adverse_momentum_veto_enabled": cum_adverse_momentum_veto_enabled,
        "cum_adverse_momentum_veto_pct":     cum_adverse_momentum_veto_pct,
        "cum_adverse_momentum_veto_tiers":   list(cum_adverse_momentum_veto_tiers),
        "hc_low_hurst_veto_enabled":         hc_low_hurst_veto_enabled,
        "hc_low_hurst_threshold":            hc_low_hurst_threshold,
        "hc_low_hurst_markov_extremity":     hc_low_hurst_markov_extremity,
        "hc_dist_min":                       hc_dist_min,
        "rolling_wr_enabled":                rolling_wr_enabled,
        "rolling_wr_window":                 rolling_wr_window,
        "rolling_wr_threshold":              rolling_wr_threshold,
        "rolling_wr_timeout_mins":           rolling_wr_timeout_mins,
        "rolling_wr_defensive_tiers":        list(rolling_wr_defensive_tiers),
        "adaptive_br_enabled":               adaptive_br_enabled,
        "adaptive_br_reduced_frac":          adaptive_br_reduced_frac,
        "adaptive_br_loss_trigger_usd":      adaptive_br_loss_trigger_usd,
        "adaptive_br_wr_trigger":            adaptive_br_wr_trigger,
        "adaptive_br_wr_window_h":           adaptive_br_wr_window_h,
        "adaptive_br_recover_wr":            adaptive_br_recover_wr,
        "patient_topup_enabled":             patient_topup_enabled,
        "patient_topup_interval_s":          patient_topup_interval_s,
        "patient_topup_min_mins":            patient_topup_min_mins,
        "patient_topup_dynamic_kelly":       patient_topup_dynamic_kelly,
        "take_profit_enabled":               take_profit_enabled,
        "take_profit_cents":                 take_profit_cents,
        "take_profit_perp_confirmed_only":   take_profit_perp_confirmed_only,
        "take_profit_perp_min":              take_profit_perp_min,
        "resting_tp_enabled":                resting_tp_enabled,
        "tp_reentry_enabled":                tp_reentry_enabled,
        "sl_reentry_enabled":                sl_reentry_enabled,
        "high_price_tp_enabled":             high_price_tp_enabled,
        "high_price_tp_min_cents":           high_price_tp_min_cents,
        "high_price_tp_target_cents":        high_price_tp_target_cents,
        "highrisk_tp_enabled":               highrisk_tp_enabled,
        "highrisk_tp_dd_cents":              highrisk_tp_dd_cents,
        "highrisk_tp_dist_max":              highrisk_tp_dist_max,
        "highrisk_tp_cents":                 highrisk_tp_cents,
        "perp_veto_enabled":                 perp_veto_enabled,
        "perp_veto_m30s_threshold":          perp_veto_m30s_threshold,
        "taker_flow_veto_enabled":           taker_flow_veto_enabled,
        "taker_flow_veto_agg_min":           taker_flow_veto_agg_min,
        "taker_flow_veto_dist_max":          taker_flow_veto_dist_max,
        "taker_flow_veto_min_trades":        taker_flow_veto_min_trades,
        "book_skew_veto_enabled":            book_skew_veto_enabled,
        "book_skew_threshold":               book_skew_threshold,
        "book_skew_golden_only":             book_skew_golden_only,
        "perp_imb_veto_enabled":             perp_imb_veto_enabled,
        "perp_imb_veto_threshold":           perp_imb_veto_threshold,
        "golden_near_vol_veto_enabled":      golden_near_vol_veto_enabled,
        "golden_near_vol_dist_max":          golden_near_vol_dist_max,
        "golden_near_vol_gk_min":            golden_near_vol_gk_min,
        "min_entry_price":                   min_entry_price,
        "max_trade_usd":                     max_trade_usd,
        "ev_walkup_override_enabled":        ev_walkup_override_enabled,
        "ev_override_pwin_min":              ev_override_pwin_min,
        "ev_override_price_max":             ev_override_price_max,
        "ev_override_book_skew_min":         ev_override_book_skew_min,
        "ev_override_floor":                 ev_override_floor,
        "rrm_exit_enabled":                  rrm_exit_enabled,
        "rrm_exit_min_score":                rrm_exit_min_score,
        "predict_cross_exit_enabled":        predict_cross_exit_enabled,
        "pcross_prob":                       pcross_prob,
        "pcross_max_mins":                   pcross_max_mins,
        "pcross_confirm_polls":              pcross_confirm_polls,
        "pcross_min_contracts":              pcross_min_contracts,
        "pcross_keep_alive_mins":            pcross_keep_alive_mins,
        "rrm_exit_min_contracts":            rrm_exit_min_contracts,
        "standard_price_cap_yes":            (standard_price_cap_yes if standard_price_cap_yes is not None else MAX_ENTRY_PRICE_YES),
        "standard_price_cap_no":             (standard_price_cap_no  if standard_price_cap_no  is not None else MAX_ENTRY_PRICE_NO),
        "fade_bounce_enabled":              fade_bounce_enabled,
        "fade_bounce_no_ask_min":           fade_bounce_no_ask_min,
        "fade_bounce_no_ask_max":           fade_bounce_no_ask_max,
        "fade_bounce_yes_side_enabled":     fade_bounce_yes_side_enabled,
        "fade_bounce_markov_no_max":        fade_bounce_markov_no_max,
        "fade_bounce_markov_yes_min":       fade_bounce_markov_yes_min,
        "fade_bounce_hurst_min":            fade_bounce_hurst_min,
        "fade_bounce_dist_min":             fade_bounce_dist_min,
        "fade_bounce_min_mins":             fade_bounce_min_mins,
        "fade_bounce_max_mins":             fade_bounce_max_mins,
        "fade_bounce_min_stake_pct":        fade_bounce_min_stake_pct,
        "fade_bounce_kelly_frac":           fade_bounce_kelly_frac,
        "fade_bounce_sl_cents":             fade_bounce_sl_cents,
        "fade_bounce_max_capital_usd":      fade_bounce_max_capital_usd,
        "sl_trigger_mode":          sl_trigger_mode,
        "sl_aggressive_sell":       sl_aggressive_sell,
        "ev_gate":              ev_gate,
        "ev_floor":             ev_floor,
        "ev_ceiling":           ev_ceiling,
        "ev_tp_weight":         ev_tp_weight,
        "ev_strong_floor":      ev_strong_floor,
        "ev_strong_gap_min":    ev_strong_gap_min,
        "ev_strong_price_max":  ev_strong_price_max,
        "ev_strong_tp_min":     ev_strong_tp_min,
        "ev_strong_max_mins":   ev_strong_max_mins,
        "ev_strong_max_adverse_momentum": ev_strong_max_adverse_momentum,
        "last_bar_adverse_threshold":     last_bar_adverse_threshold,
        "orderbook_signal_enabled":       orderbook_signal_enabled,
        "trade_flow_signal_enabled":      trade_flow_signal_enabled,
        "trade_flow_lookback_n":          trade_flow_lookback_n,
        "orderbook_lockin_enabled":       orderbook_lockin_enabled,
        "orderbook_lockin_spread_max":    orderbook_lockin_spread_max,
        "orderbook_lockin_price_min":     orderbook_lockin_price_min,
        "orderbook_lockin_gap_min":       orderbook_lockin_gap_min,
        "late_window_price_max_lockin":   late_window_price_max_lockin,
        "hc_lockin_ev_floor":             hc_lockin_ev_floor,
        "hc_lockin_min_stake_pct":        hc_lockin_min_stake_pct,
        "strong_floor_hurst_bypass":      strong_floor_hurst_bypass,
        "retry_walk_cents":     retry_walk_cents,
        "max_window_fill_attempts": max_window_fill_attempts,
        "refill_retry_sleep_s":     refill_retry_sleep_s,
        "late_window_mins":      late_window_mins,
        "late_window_price_max": late_window_price_max,
        "late_window_min_tp":    late_window_min_tp,
        "late_window_ev_floor":  late_window_ev_floor,
        "late_sure_vol_bypass":  late_sure_vol_bypass,
        "strong_floor_min_stake_pct": strong_floor_min_stake_pct,
        "late_sure_min_stake_pct":    late_sure_min_stake_pct,
        "standard_min_entry_contracts": standard_min_entry_contracts,
        "golden_price_lo":       golden_price_lo,
        "golden_price_hi":       golden_price_hi,
        "golden_no_dist":        golden_no_dist,
        "golden_no_hurst":       golden_no_hurst,
        "okx_boost_enabled":     okx_boost_enabled,
        "okx_boost_weight":      okx_boost_weight,
        "okx_boost_scale":       okx_boost_scale,
        "high_conv_gap_min":     high_conv_gap_min,
        "high_conv_persist_min": high_conv_persist_min,
        "high_conv_tp_strong":   high_conv_tp_strong,
        "high_conv_price_max":   high_conv_price_max,
        "high_conv_ev_floor":    high_conv_ev_floor,
        "high_conv_max_mins":    high_conv_max_mins,
        "high_conv_vol_bypass":           high_conv_vol_bypass,
        "high_conv_vol_bypass_momentum":  high_conv_vol_bypass_momentum,
        "high_conv_vol_bypass_distance":  high_conv_vol_bypass_distance,
        "high_conv_vol_bypass_strong_distance": high_conv_vol_bypass_strong_distance,
        "late_dir_enabled":         late_dir_enabled,
        "late_dir_mins":            late_dir_mins,
        "late_dir_gap_min":         late_dir_gap_min,
        "late_dir_persist_min":     late_dir_persist_min,
        "late_dir_distance_min":    late_dir_distance_min,
        "late_dir_momentum_min":    late_dir_momentum_min,
        "late_dir_strong_distance": late_dir_strong_distance,
        "late_dir_ev_floor":        late_dir_ev_floor,
        "late_dir_price_max":       late_dir_price_max,
    }

    # ChainVector futures lead-lag feed (background thread). ONE /momentum
    # poller feeds BOTH lead venues: futures_lead (binance_futures â€” primary,
    # replaces the old CME/Databento feed) and okx_feed (okx â€” secondary,
    # consensus veto partner). Started here so by the time the first entry
    # window opens the buffer has 60+ seconds of mids. Missing API key is a
    # soft failure â€” all lead vetoes fail-open.
    cv_lead_hub: Optional[CVLeadFeed] = None
    futures_lead = None   # CVLeadVenueView (binance_futures) or None
    okx_feed = None       # CVLeadVenueView (okx) or None
    if not (futures_lead_disabled and okx_disabled):
        if os.environ.get("CHAINVECTOR_API_KEY", ""):
            try:
                cv_lead_hub = CVLeadFeed(poll_interval_s=okx_poll_interval_s)
                cv_lead_hub.start()
                attach_lead_feed(cv_lead_hub)   # signal_feeds reuses the buffer
                if not futures_lead_disabled:
                    futures_lead = cv_lead_hub.view("binance_futures")
                if not okx_disabled:
                    okx_feed = cv_lead_hub.view("okx")
            except Exception as e:
                log.warning(f"Failed to start ChainVector lead feed: {e}. Continuing without it.")
                cv_lead_hub = None
                futures_lead = None
                okx_feed = None
        else:
            log.warning("CHAINVECTOR_API_KEY not set â€” futures-lead/OKX vetoes and "
                        "momentum EV weight disabled (fail-open)")

    # Try fetching live balance at startup
    live_bal = await get_live_bankroll()
    if live_bal and live_bal > 0:
        session.bankroll = live_bal
        log.info(f"Live balance: ${live_bal:.2f}")

    # Recover any open positions left over from a previous run (crash, restart,
    # power loss). Without this, a position held in Polymarket but not in
    # session.pending would silently miss its WIN/LOSS log line and Daily P&L.
    if not dry_run:
        n_rec = await recover_open_positions(session)
        if n_rec:
            log.info(f"Position recovery: rebuilt {n_rec} pending settlement(s).")
        # â”€â”€ 2026-06-15: RESTING-TP ORPHAN CLEANUP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # The ONLY resting (non-IOC) orders this daemon ever places are TP
        # orders. On startup, cancel ALL live resting orders so a crash/restart
        # can never leave an orphaned sell order on the book (which could fill
        # later and put us net short). This is the critical safety net for the
        # resting-TP system. Safe even if resting-TP is disabled (no-op then).
        if resting_tp_enabled:
            try:
                _open = await _kget("/portfolio/orders",
                                    {"status": "resting", "limit": 1000})
                _orders = _open.get("orders", []) if isinstance(_open, dict) else []
                _cancelled = 0
                for _o in _orders:
                    _oid = _o.get("order_id")
                    if _oid:
                        try:
                            await _kdelete(f"/portfolio/events/orders/{_oid}")
                            _cancelled += 1
                        except Exception:
                            pass
                log.warning(f"[RESTING-TP] startup orphan cleanup: cancelled "
                            f"{_cancelled} pre-existing resting order(s).")
            except Exception as _e:
                log.warning(f"[RESTING-TP] startup orphan cleanup failed "
                            f"(non-fatal): {_e!r}")
            # â”€â”€ RE-PLACE TPs FOR RECOVERED POSITIONS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # The cleanup above cancelled the TP orders of any positions that
            # were open before this restart. Re-place a fresh resting TP for
            # each recovered position based on its blended entry price, so a
            # restart no longer strips the TP off an open position.
            for _wid, _trade in list(session.pending.items()):
                await _place_resting_tp_for_trade(
                    _trade,
                    take_profit_cents=take_profit_cents,
                    high_price_tp_enabled=high_price_tp_enabled,
                    high_price_tp_min_cents=high_price_tp_min_cents,
                    high_price_tp_target_cents=high_price_tp_target_cents,
                )

        # â”€â”€ TIGHT SL MONITOR FOR RECOVERED POSITIONS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # If stop-loss is enabled, give each recovered position the same tight
        # poll-based SL + resting-TP monitor a fresh position gets (the inline
        # monitor can't be reused, so monitor_recovered_position mirrors its
        # core). Runs regardless of resting-TP (SL is independent). Blocks until
        # each resolves (SL / full TP) or nears settlement â€” consistent with the
        # daemon's one-position-at-a-time design. Settlement of any remainder is
        # handled by the main loop once these return.
        if sl_enabled and session.pending:
            for _wid, _trade in list(session.pending.items()):
                try:
                    await monitor_recovered_position(
                        session, audit, _trade, _wid,
                        sl_loss_cents=sl_loss_cents,
                        sl_trigger_mode=sl_trigger_mode,
                        sl_poll_interval_s=sl_poll_interval_s,
                        sl_disable_late_mins=sl_disable_late_mins,
                        sl_aggressive_sell=sl_aggressive_sell,
                        take_profit_cents=take_profit_cents,
                    )
                except Exception as _me:
                    log.warning(f"[RECOVERED-SL] monitor error (non-fatal): {_me!r}")

    last_balance_refresh = time.time()

    while True:
        session.new_day_check()
        session.check_defensive_timeout()

        # Check pending settlements first
        settled_windows = []
        for wid, trade in list(session.pending.items()):
            # â”€â”€ 2026-06-15: RECOVERED-POSITION RESTING-TP MONITOR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Recovered positions aren't SL-monitored, so their re-placed TP is
            # tracked here: reconcile any fills (records PnL, reduces position)
            # and detect a full TP exit. Only touches positions carrying a
            # resting_tp_id (set by _place_resting_tp_for_trade on restart).
            if trade.get("resting_tp_id"):
                try:
                    _rst = await _korder_status(trade["resting_tp_id"])
                    if _rst:
                        _rnew = max(0, _order_fill_count(_rst)
                                    - trade.get("resting_tp_filled", 0))
                        if _rnew > 0:
                            _reconcile_recovered_tp_fill(trade, session, audit, _rnew, wid)
                        if (trade["contracts"] <= 0
                                or (_rst.get("status") or "").lower() == "executed"
                                or _order_remaining_count(_rst) == 0):
                            log.warning(f"  [RESTING-TP] recovered {trade['ticker']} "
                                        f"fully filled via TP â€” closed.")
                            trade["resting_tp_id"] = None
                            session.pending.pop(wid, None)
                            continue
                except Exception:
                    pass
            result = await check_settlement(trade["ticker"])
            if result is not None:
                # Cancel the recovered position's resting TP before settling
                # (reconcile any final fills first) so it can't linger.
                if trade.get("resting_tp_id"):
                    try:
                        _rst = await _korder_status(trade["resting_tp_id"])
                        if _rst:
                            _rnew = max(0, _order_fill_count(_rst)
                                        - trade.get("resting_tp_filled", 0))
                            if _rnew > 0:
                                _reconcile_recovered_tp_fill(trade, session, audit, _rnew, wid)
                        await _kdelete(f"/portfolio/events/orders/{trade['resting_tp_id']}")
                    except Exception:
                        pass
                    trade["resting_tp_id"] = None
                    if trade["contracts"] <= 0:
                        session.pending.pop(wid, None)
                        continue
                # â”€â”€ Primary leg PnL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # If a safety hedge fired its SL on the primary, the primary
                # PnL was already recorded by the SL exit. We don't double-
                # count here; only the hedge leg needs settlement processing.
                primary_already_recorded = bool(trade.get("primary_sold"))
                if primary_already_recorded:
                    primary_pnl = float(trade.get("primary_realized_pnl") or 0)
                    log.info(
                        f"  Primary already SL-exited at "
                        f"{trade.get('primary_exit_cents')}\u00a2 "
                        f"(pnl=${primary_pnl:+.2f}); resolving hedge only."
                    )
                else:
                    won = (result == trade["side"].upper())
                    primary_pnl = trade["net_win"] if won else -trade["cost"]
                    emoji = "WIN +" if won else "LOSS -"
                    log.info(
                        f"SETTLED {emoji}${abs(primary_pnl):.2f} | {trade['ticker']} | "
                        f"BUY {trade['side']} @ {trade['limit_price']}\u00a2 | "
                        f"Result={result} | Daily P&L: ${session.daily_pnl + primary_pnl:+.2f}"
                    )
                    session.record(primary_pnl)
                    audit.write(build_settlement_record(
                        window_id=wid, ticker=trade["ticker"], side=trade["side"],
                        result=result, contracts=trade["contracts"],
                        pnl_usd=round(primary_pnl, 4),
                        daily_pnl_usd=round(session.daily_pnl, 4),
                        limit_price_cents=trade["limit_price"],
                    ))
                # â”€â”€ Hedge leg PnL (2026-05-31) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # The hedge contract is on the OPPOSITE side of the primary.
                # NO contract wins (pays $1) when the YES side LOSES.
                # If primary side = YES and result = NO: hedge_won = True
                # If primary side = NO and result = YES: hedge_won = True
                # Otherwise hedge settles at $0.
                hedge_state = trade.get("hedge")
                if hedge_state and hedge_state.get("active"):
                    hedge_won = (result != trade["side"].upper())
                    h_contracts = hedge_state.get("contracts_filled", 0)
                    h_entry_yes_leg = hedge_state.get("entry_yes_leg", 0)
                    # For NO hedge: cost = (1 - entry_yes_leg); payout if won = $1
                    # For YES hedge: cost = entry_yes_leg; payout if won = $1
                    if hedge_state.get("side") == "NO":
                        h_cost_per = (1 - h_entry_yes_leg)
                    else:
                        h_cost_per = h_entry_yes_leg
                    h_cost_total = h_cost_per * h_contracts
                    h_net_win = (1 - h_cost_per) * h_contracts  # net if hedge wins
                    if hedge_won:
                        hedge_pnl = h_net_win
                        # Subtract fees (rough estimate: 2Ã— maker fee Ã— 0.5)
                        hedge_pnl -= 2 * MAKER_FEE_RATE * 0.5 * h_contracts
                    else:
                        hedge_pnl = -h_cost_total
                    hedge_pnl = round(hedge_pnl, 4)
                    hedge_state["status"] = "settled"
                    hedge_state["exit_cents"] = 100 if hedge_won else 0
                    hedge_state["exit_ts"] = datetime.now(timezone.utc).isoformat()
                    hedge_state["pnl_usd"] = hedge_pnl
                    log.info(
                        f"  HEDGE SETTLED {'WIN' if hedge_won else 'LOSS'}: "
                        f"{hedge_state['side']} {h_contracts}c @ "
                        f"{hedge_state['entry_cents']}\u00a2 -> "
                        f"{hedge_state['exit_cents']}\u00a2  hedge_pnl=${hedge_pnl:+.2f}"
                    )
                    session.record(hedge_pnl)
                    # Combined PnL log line for visibility
                    combined = primary_pnl + hedge_pnl
                    log.info(
                        f"  COMBINED (primary+hedge): primary=${primary_pnl:+.2f}  "
                        f"hedge=${hedge_pnl:+.2f}  total=${combined:+.2f}"
                    )
                    audit.write({
                        "type":                "HEDGE_EXIT",
                        "window_id":           wid,
                        "ticker":              trade["ticker"],
                        "primary_side":        trade["side"],
                        "primary_pnl":         round(primary_pnl, 4),
                        "primary_sl_fired":    primary_already_recorded,
                        "hedge_side":          hedge_state.get("side"),
                        "hedge_contracts":     h_contracts,
                        "hedge_entry_cents":   hedge_state.get("entry_cents"),
                        "hedge_exit_cents":    hedge_state.get("exit_cents"),
                        "hedge_won":           hedge_won,
                        "hedge_pnl":           hedge_pnl,
                        "combined_pnl":        round(combined, 4),
                        "settlement_result":   result,
                        "exit_reason":         "settled",
                        "ts":                  datetime.now(timezone.utc).isoformat(),
                        "ts_ms":               int(time.time() * 1000),
                    })
                # â”€â”€ 2026-06-01: flip leg settlement resolution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # If a smart flip is still active at settlement (monitor didn't
                # sell mid-window), resolve PnL based on actual settlement.
                flip_state = trade.get("flip")
                if flip_state and flip_state.get("active"):
                    # Flip side wins iff settlement result == flip side
                    flip_won = (result == flip_state.get("side", "").upper())
                    f_contracts = flip_state.get("contracts_filled", 0)
                    f_entry_yes_leg = flip_state.get("entry_yes_leg", 0)
                    if flip_state.get("side") == "NO":
                        f_cost_per = (1 - f_entry_yes_leg)
                    else:
                        f_cost_per = f_entry_yes_leg
                    f_cost_total = f_cost_per * f_contracts
                    f_net_win = (1 - f_cost_per) * f_contracts
                    if flip_won:
                        flip_pnl = f_net_win - 2 * MAKER_FEE_RATE * 0.5 * f_contracts
                    else:
                        flip_pnl = -f_cost_total
                    flip_pnl = round(flip_pnl, 4)
                    flip_state["status"] = "settled"
                    flip_state["exit_cents"] = 100 if flip_won else 0
                    flip_state["exit_ts"] = datetime.now(timezone.utc).isoformat()
                    flip_state["pnl_usd"] = flip_pnl
                    flip_state["active"] = False
                    log.info(
                        f"  FLIP SETTLED {'WIN' if flip_won else 'LOSS'}: "
                        f"{flip_state['side']} {f_contracts}c @ "
                        f"{flip_state.get('entry_cents')}\u00a2 \u2192 "
                        f"{flip_state['exit_cents']}\u00a2  flip_pnl=${flip_pnl:+.2f}"
                    )
                    session.record(flip_pnl)
                    combined = primary_pnl + flip_pnl
                    log.info(
                        f"  COMBINED (primary + flip held to settle): "
                        f"primary=${primary_pnl:+.2f}  flip=${flip_pnl:+.2f}  "
                        f"total=${combined:+.2f}"
                    )
                    audit.write({
                        "type":              "FLIP_EXIT",
                        "window_id":         wid,
                        "ticker":            trade["ticker"],
                        "primary_side":      trade["side"],
                        "primary_pnl":       round(primary_pnl, 4),
                        "flip_side":         flip_state.get("side"),
                        "flip_contracts":    f_contracts,
                        "flip_entry_cents":  flip_state.get("entry_cents"),
                        "flip_exit_cents":   flip_state.get("exit_cents"),
                        "flip_peak_bid":     flip_state.get("peak_bid"),
                        "flip_won":          flip_won,
                        "flip_pnl":          flip_pnl,
                        "combined_pnl":      round(combined, 4),
                        "settlement_result": result,
                        "exit_reason":       "settled",
                        "ts":                datetime.now(timezone.utc).isoformat(),
                        "ts_ms":             int(time.time() * 1000),
                    })
                settled_windows.append(wid)
                # 2026-05-31: bid trajectory observed during SL monitor for
                # this position. Lets us answer "would a tighter SL have
                # caught winners?" without re-running the daemon. min_bid is
                # the worst bid we saw; max_drawdown is entry minus min_bid.
                _won_for_audit = (not primary_already_recorded
                                  and result == trade["side"].upper())
                if any(trade.get(k) is not None for k in
                       ("min_bid_cents", "max_drawdown_cents", "n_sl_polls")):
                    audit.write({
                        "type":                     "BID_TRAJECTORY",
                        "window_id":                wid,
                        "ticker":                   trade["ticker"],
                        "side":                     trade["side"],
                        "entry_cents":              trade["limit_price"],
                        "settlement_result":        result,
                        "won":                      _won_for_audit,
                        "pnl_usd":                  round(primary_pnl, 4),
                        "min_bid_cents":            trade.get("min_bid_cents"),
                        "max_bid_cents":            trade.get("max_bid_cents"),
                        "max_drawdown_cents":       trade.get("max_drawdown_cents", 0),
                        "min_bid_secs_from_entry":  trade.get("min_bid_secs_from_entry", 0),
                        "n_sl_polls":               trade.get("n_sl_polls", 0),
                    })
        for wid in settled_windows:
            del session.pending[wid]

        # Refresh bankroll every 30 min
        if time.time() - last_balance_refresh > 1800:
            live_bal = await get_live_bankroll()
            if live_bal and live_bal > 0:
                session.bankroll = live_bal
                log.info(f"Balance refresh: ${live_bal:.2f}")
            last_balance_refresh = time.time()

        # Session limit check
        stop = session.limit_hit()
        if stop:
            log.warning(f"SESSION PAUSED â€” {stop}  (daily P&L: ${session.daily_pnl:+.2f})")
            await asyncio.sleep(300)
            continue

        # Timing
        close_dt  = next_window_close()
        now       = datetime.now(timezone.utc)
        mins_left = (close_dt - now).total_seconds() / 60
        window_id = close_dt.strftime("%Y%m%d%H%M")

        # Sleep until 12 min before close (entry window opens)
        if mins_left > 12.5:
            sleep_s = (mins_left - 12) * 60
            log.info(
                f"Next window {window_id} closes in {mins_left:.1f} min "
                f"â€” sleeping {fmt(sleep_s)}"
            )
            await asyncio.sleep(sleep_s)
            continue

        # Window is expiring â€” skip it
        if mins_left < 2.5:
            skip_s = (mins_left + 1.5) * 60
            if window_id not in session.traded:
                log.info(f"Window {window_id} closing ({mins_left:.1f} min) â€” skip")
                session.traded.add(window_id)
            await asyncio.sleep(skip_s)
            continue

        # Already traded this window
        if window_id in session.traded:
            await asyncio.sleep(min(30, mins_left * 60 * 0.5))
            continue

        # â”€â”€ Active window: fetch market + signal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        log.info(f"Window {window_id} | {mins_left:.1f} min to close â€” running signal...")

        try:
            market = await fetch_market()
        except Exception as e:
            log.error(f"Market fetch error: {e}")
            await asyncio.sleep(30)
            continue

        if not market:
            log.warning("No active market found â€” retrying in 30s")
            await asyncio.sleep(30)
            continue

        # 2026-07-02: feed the bid-stability history on EVERY poll so the
        # entry gate has bid context by the time a candidate fires.
        _bidstab_update(window_id, market)

        # 2026-05-28 PM: feed last poll's cached TP into run_signal so the
        # last-bar-adverse gate can detect extreme-signal lock-ins and bypass.
        # Cache is module-scope (persists across polls per window); TP that's
        # 5-15s stale is fine for the extreme check (we're testing >=0.85 / <=0.15,
        # well above the noise threshold).
        prev_tp_for_bypass = None
        prev_tp_entry = window_tp_cache.get(window_id)
        if prev_tp_entry and not prev_tp_entry.get("error"):
            prev_tp_for_bypass = prev_tp_entry.get("bs_p_above")

        try:
            signal = await run_signal(
                market, session.effective_bankroll(),
                last_bar_adverse_threshold=last_bar_adverse_threshold,
                tp_bs_p_above=prev_tp_for_bypass,
                last_bar_extreme_gap_min=0.30,
                last_bar_extreme_tp=high_conv_tp_strong,  # share HC's TP threshold
                standard_price_cap_yes=standard_price_cap_yes,
                standard_price_cap_no=standard_price_cap_no,
                golden_price_lo=golden_price_lo,
                golden_price_hi=golden_price_hi,
                golden_no_dist=golden_no_dist,
                golden_no_hurst=golden_no_hurst,
            )
        except Exception as e:
            log.error(f"Signal error: {e}", exc_info=True)
            await asyncio.sleep(30)
            continue

        approved    = signal["approved"]
        rec         = signal["recommendation"]
        contracts   = signal["contracts"]
        limit_price = signal["limit_price"]
        ticker      = signal["ticker"]
        reasons     = signal["rejection_reasons"]
        sig         = signal["signal"]
        mkt         = signal["market"]
        recent_5m   = signal.get("recent_5m_pct", []) or []
        # 2026-05-29: track the floor that approved this trade. Initialized
        # to ev_floor (standard floor for run_signal direct approvals which
        # require positive Kelly = positive EV). EV gate paths override this
        # with their tier-specific floor (late-sure -0.10, hc-lockin -0.10,
        # high-conv user-set, strong-floor -0.06). The lead-cents check below
        # uses this same floor instead of hardcoded strong-floor.
        approved_floor = ev_floor

        # â”€â”€ Polymarket orderbook + trade-flow snapshot (audit-only, 5s cached) â”€â”€
        # Fetched ONCE per poll iteration; both modules have their own caches
        # so subsequent calls within 5s reuse the data. Record-only for now;
        # promote to gates after we have a day of correlated outcomes.
        ob_snap = None
        tr_snap = None
        if orderbook_signal_enabled:
            try:
                ob_snap = await fetch_orderbook(ticker, _kget)
            except Exception:
                ob_snap = None
        if trade_flow_signal_enabled:
            try:
                tr_snap = await fetch_recent_trades(ticker, _kget,
                                                     limit=trade_flow_lookback_n)
            except Exception:
                tr_snap = None

        # â”€â”€ Terminal Probability (ChainVector six-estimator ensemble) â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Recompute every TP_REFRESH_INTERVAL_S seconds so the signal tracks
        # fast-moving markets. The engine is called with target=strike and
        # close_ts=this contract's exact close time, so the returned P(above)
        # is bucket-free. The ChainVector client caches /probability for
        # TTL_PROB (12s) per (strike, close_ts), so the 15s refresh cadence
        # costs ~1 API call per refresh and gives fresh TP during signal
        # development.
        #
        # The `checkpoint` label is a freshness tag: "@-0.1m" means the
        # cached TP value is 0.1 minutes (6s) old.
        TP_REFRESH_INTERVAL_S = 15.0

        mins_now = sig["minutes_left"]
        prev_tp  = window_tp_cache.get(window_id)
        last_compute_ts = (prev_tp or {}).get("_computed_at_unix", 0.0)
        seconds_since_compute = time.time() - last_compute_ts
        should_recompute = (
            (prev_tp is None)
            or (seconds_since_compute >= TP_REFRESH_INTERVAL_S)
        )
        tp_result: Optional[dict] = None
        if should_recompute and not term_prob_disabled:
            try:
                loop = asyncio.get_event_loop()
                tp_result = await loop.run_in_executor(
                    None,
                    lambda: compute_terminal_prob(
                        mkt["btc_price"], mkt["strike"], close_dt
                    ),
                )
                tp_result["_computed_at_unix"] = time.time()
                tp_result["checkpoint"]    = "fresh"
                tp_result["window_id"]     = window_id
                tp_result["markov_p_yes"]  = sig["p_yes"]
                tp_result["markov_gap"]    = sig["gap"]
                window_tp_cache[window_id] = tp_result
                save_snapshot(tp_result, ticker, _log_dir)
            except Exception as e:
                log.warning(f"Terminal prob compute failed: {e}")
                tp_result = None

        # Use the most recent TP result for this window (always one â€” fresh if
        # we just computed, or up to ~60s old if not).
        tp_cached = window_tp_cache.get(window_id)

        # Status line â€” add term_p_yes with freshness tag
        tp_str = ""
        if tp_cached and not tp_cached.get("error"):
            t_p     = tp_cached.get("bs_p_above", 0.5)
            t_iv    = tp_cached.get("deribit_iv_at_strike", 0)
            t_age_s = time.time() - tp_cached.get("_computed_at_unix", time.time())
            t_tag   = "fresh" if t_age_s < 5 else f"-{t_age_s/60:.1f}m"
            tp_str  = (f" | p_term={t_p:.1%} iv={t_iv*100:.1f}% @{t_tag}")
        log.info(
            f"BTC ${mkt['btc_price']:,.0f} | Strike ${mkt['strike']:,.0f} | "
            f"Î”{mkt['dist_pct']:+.3f}% | {sig['minutes_left']:.1f}min | "
            f"p(YES)={sig['p_yes']:.1%} gap={sig['gap']:.3f} persist={sig['persist']:.2f} "
            f"{'[GOLDEN]' if sig['is_golden'] else ''}"
            f"{tp_str}"
        )

        # If this poll just computed a new TP, also log a 1-line cross-method check
        if tp_result and not tp_result.get("error"):
            log.info(
                f"   TermProb (fresh, cv-ensemble): "
                f"ens={tp_result['bs_p_above']*100:.1f}% "
                f"gauss={tp_result['bl_p_above']*100:.1f}% "
                f"mc={tp_result['mc_p_above']*100:.1f}% "
                f"(tte={tp_result['hours_to_deribit_expiry']*60:.1f}min exact, "
                f"spread {tp_result['cross_method_max_disagreement']*100:.1f}pp)"
            )

        # â”€â”€ (Optional) gate-relaxation logic â€” OFF by default â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # When term_prob_relax is enabled AND TP agrees with Markov direction,
        # we allow a relaxed gap floor (0.08 instead of MARKOV_MIN_GAP) for
        # this poll only. Cleans only the gap blocker â€” other gates unchanged.
        if (term_prob_relax and tp_cached and not tp_cached.get("error")
                and not approved):
            t_p_yes = tp_cached.get("bs_p_above", 0.5)
            markov_yes_lean = sig["p_yes"] > 0.5
            tp_yes_lean     = t_p_yes      > 0.55
            tp_no_lean      = t_p_yes      < 0.45
            confirms = (markov_yes_lean and tp_yes_lean) or \
                       (not markov_yes_lean and tp_no_lean)
            if confirms and sig["gap"] >= 0.08 and "Markov gap" in " | ".join(reasons):
                # Strip the Markov-gap blocker from the rejection list and re-evaluate
                new_reasons = [r for r in reasons if "Markov gap" not in r]
                if not new_reasons:
                    log.info(
                        f"   TermProb CONFIRMS direction "
                        f"(markov={sig['p_yes']:.1%}, term={t_p_yes:.1%}) "
                        f"â†’ relaxing gap gate (was 0.11, allowing 0.08)"
                    )
                    # Re-derive recommendation
                    rec         = "YES" if sig["p_yes"] > 0.5 else "NO"
                    approved    = True
                    reasons     = []
                    # Re-compute contracts at relaxed sizing (use existing Kelly logic
                    # by reusing the signal's contracts field â€” already > 0 if Markov
                    # passed everything else; fall back to a minimum if not).
                    if signal.get("contracts", 0) <= 0:
                        contracts = 1
                    limit_price = mkt["yes_ask"] if rec == "YES" else mkt["no_ask"]

        # â”€â”€ (Optional) EV gate override â€” OFF by default â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # When --ev-gate is enabled AND the only remaining blocker is the
        # price cap, we recompute the trade's expected value using a
        # COMBINED Markov+TP probability. If EV per contract clears the
        # floor (default $0.05) AND we're below the safety ceiling (90Â¢
        # regardless of EV), we override the price-cap rejection. This
        # captures high-confidence trades that the flat 72Â¢/65Â¢ cap misses
        # when both forward-looking signals agree.
        #
        # Direction safety: if combined-p points OPPOSITE Markov, we abort
        # the override â€” never trade against the deterministic call.
        # Initialize active_tier so downstream gates (e.g. hurst_tp_veto) can
        # always read it. When EV-gate override doesn't run (signal approved
        # cleanly or ev_gate disabled), the trade is implicitly "standard".
        # 2026-06-03: fixes UnboundLocalError when hurst_tp_veto runs on a
        # clean-approved signal that never entered the EV-override path.
        active_tier = "standard"

        if ev_gate and not approved and reasons:
            is_price_only = len(reasons) == 1 and ("price " in reasons[0] and "cap" in reasons[0])
            # Extended qualification: high-conv tier (the strictest tier, with
            # extreme signal requirements) is allowed to also bypass the timing
            # gate. This catches markets that priced in BEFORE the 10-min entry
            # window opens (e.g., 10.8 min left with gap=0.500). Restricted to
            # high-conv only â€” other tiers stay locked to the standard window.
            def _r_is_price(r: str) -> bool:
                return "price " in r and "cap" in r
            def _r_is_timing(r: str) -> bool:
                return "timing" in r or "min outside" in r
            def _r_is_vol(r: str) -> bool:
                return "high vol" in r
            def _r_is_hurst(r: str) -> bool:
                return "mean-reverting" in r

            # Hoist tp_cached_now lookup so directional-confirmation helpers
            # (computed BEFORE qualifies_for_eval) can use it. The same value
            # is reused inside the qualified-eval block below.
            tp_cached_now = window_tp_cache.get(window_id)

            # Directional confirmation for vol bypass â€” independent of any
            # tier decision. Vol is the most-protected gate so we require
            # both BTC distance from strike AND recent 5m momentum to confirm
            # direction. Compute once here so we can use it for both
            # qualification and tier-selection.
            # CRITICAL BUG FIX (2026-05-28 PM): run_signal sets recommendation
            # = "NO_TRADE" (a truthy string!) when the trade is rejected â€” not
            # None. The old `signal.get("recommendation") or fallback` pattern
            # short-circuited on the truthy "NO_TRADE" string, leaving
            # rec_for_dir == "NO_TRADE" instead of the Markov-implied direction.
            # Downstream, `rec_yes = (rec_for_dir == "YES")` evaluated False
            # for ALL rejected trades. The direction-match check
            # `(sig["p_yes"] > 0.5) == rec_yes` then silently FAILED for YES
            # setups (True == False), while NO setups masked the bug (False ==
            # False). Result: zero HC fires on YES-direction lock-ins all day.
            sig_rec = signal.get("recommendation")
            if sig_rec and sig_rec not in ("NO_TRADE", "NO TRADE"):
                rec_for_dir = sig_rec
            else:
                rec_for_dir = "YES" if sig["p_yes"] > 0.5 else "NO"
            dir_confirmed = False
            dir_reason: Optional[str] = None
            if high_conv_vol_bypass:
                dir_confirmed, dir_reason = _is_high_conv_directional(
                    btc_price=mkt["btc_price"],
                    strike=mkt["strike"],
                    recent_5m=recent_5m,
                    rec=rec_for_dir,
                    momentum_min_pct=high_conv_vol_bypass_momentum,
                    distance_min_pct=high_conv_vol_bypass_distance,
                    strong_distance_pct=high_conv_vol_bypass_strong_distance,
                )

            # Late-window directional vol bypass (new tier â€” looser than
            # high-conv but only late in window with directional signals).
            late_dir_vol_ok = False
            late_dir_reason: Optional[str] = None
            if late_dir_enabled:
                # Need tp_p_yes for direction check
                tp_for_check = None
                if tp_cached_now is not None and not tp_cached_now.get("error"):
                    tp_for_check = tp_cached_now.get("bs_p_above")
                late_dir_vol_ok, late_dir_reason = _is_late_window_directional(
                    mins_left=sig["minutes_left"],
                    gap=sig.get("gap", 0.0),
                    persist=sig.get("persist", 0.0),
                    tp_p_yes=tp_for_check,
                    new_rec=rec_for_dir,
                    btc_price=mkt["btc_price"],
                    strike=mkt["strike"],
                    recent_5m=recent_5m,
                    late_dir_mins=late_dir_mins,
                    late_dir_gap_min=late_dir_gap_min,
                    late_dir_persist_min=late_dir_persist_min,
                    late_dir_distance_min=late_dir_distance_min,
                    late_dir_momentum_min=late_dir_momentum_min,
                    late_dir_strong_distance=late_dir_strong_distance,
                )

            # Hurst (mean-reverting regime) is bypass-able ONLY when high-conv
            # or late-sure conditions are satisfied. These tiers have very
            # strict requirements (gapâ‰¥0.30 + TPâ‰¥0.85 for high-conv, or
            # mins<5 + TP-direction-extreme for late-sure) â€” when met, the
            # signal extremes justify overriding the regime indicator.
            hurst_bypass_ok = False
            # 2026-05-28: late_sure_qualifies is hoisted here so it can also
            # authorize vol bypass below.
            late_sure_qualifies = False
            # Pre-compute orderbook lock-in flag once so it can authorize
            # bypasses in the qualification checks below.
            ob_lockin_confirmed_early = False
            if orderbook_lockin_enabled and ob_snap:
                ob_lockin_confirmed_early, _ = _is_orderbook_lockin(
                    orderbook=ob_snap,
                    new_rec=rec_for_dir,
                    spread_max=orderbook_lockin_spread_max,
                    price_min=orderbook_lockin_price_min,
                    gap=sig.get("gap", 0.0),
                    gap_min=orderbook_lockin_gap_min,
                )
            if tp_cached_now is not None and not tp_cached_now.get("error"):
                tp_p = tp_cached_now.get("bs_p_above")
                if tp_p is not None:
                    rec_yes = (rec_for_dir == "YES")
                    tp_extreme_hc = (tp_p >= high_conv_tp_strong if rec_yes
                                     else tp_p <= 1 - high_conv_tp_strong)
                    # NEW (2026-05-28): late-sure uses its OWN, looser TP
                    # threshold (default 0.75). Deribit's 13h IV undershoots
                    # 3-min binary probability; insisting on TPâ‰¥0.85 made
                    # late-sure unreachable in 5 days of live data.
                    tp_extreme_ls = (tp_p >= late_window_min_tp if rec_yes
                                     else tp_p <= 1 - late_window_min_tp)
                    # 2026-05-28 PM: when orderbook lock-in confirmed, the TP
                    # extreme requirement is satisfied automatically (TP is
                    # being bypassed; the orderbook substitutes for it).
                    tp_extreme_hc_or_lockin = tp_extreme_hc or ob_lockin_confirmed_early
                    tp_extreme_ls_or_lockin = tp_extreme_ls or ob_lockin_confirmed_early
                    high_conv_qualifies = (
                        sig.get("gap", 0) >= high_conv_gap_min
                        and sig.get("persist", 0) >= high_conv_persist_min
                        and tp_extreme_hc_or_lockin
                        and sig["minutes_left"] <= high_conv_max_mins
                        and (sig["p_yes"] > 0.5) == rec_yes
                    )
                    late_sure_qualifies = (
                        sig["minutes_left"] <= late_window_mins  # 2026-05-28 PM: < â†’ <=
                        and tp_extreme_ls_or_lockin
                        and (sig["p_yes"] > 0.5) == rec_yes
                    )
                    # 2026-05-28 PM3: strong-floor can ALSO bypass Hurst
                    # when its own gates (gapâ‰¥0.13, priceâ‰¤88, TP-meaningful,
                    # minsâ‰¤8, direction match) all hold AND the user opts in.
                    # Note: strong-floor's EV floor (-$0.06 default) still
                    # applies â€” won't fire unless EV passes. Combine with
                    # --ev-strong-floor -0.10 for wider cushion on Hurst-blocked
                    # moderate setups.
                    strong_floor_inline = False
                    if strong_floor_hurst_bypass:
                        # Inline TP-meaningful check (matches the full check
                        # in the main strong-floor evaluation below)
                        tp_meaningful_inline = (
                            tp_p >= ev_strong_tp_min if rec_yes
                            else tp_p <= 1 - ev_strong_tp_min
                        )
                        strong_floor_inline = (
                            sig.get("gap", 0) >= ev_strong_gap_min
                            and limit_price <= ev_strong_price_max
                            and tp_meaningful_inline
                            and sig["minutes_left"] <= ev_strong_max_mins
                            and (sig["p_yes"] > 0.5) == rec_yes
                        )
                    hurst_bypass_ok = (
                        high_conv_qualifies
                        or late_sure_qualifies
                        or strong_floor_inline
                    )

            # Vol is bypass-able when EITHER high-conv-directional OR late-dir
            # confirms direction. 2026-05-28: ALSO bypass-able when LATE-SURE
            # qualifies (mins<5 + TP-direction + Markov agreement). LATE-SURE's
            # own gates are strict enough that vol-regime caution is redundant.
            # Tier selection still enforces which tier ultimately fires.
            vol_bypass_ok = (
                dir_confirmed
                or late_dir_vol_ok
                or (late_sure_vol_bypass and late_sure_qualifies)
            )

            blockers_subset_acceptable = (
                bool(reasons)
                and all(_r_is_price(r) or _r_is_timing(r)
                        or (_r_is_vol(r) and vol_bypass_ok)
                        or (_r_is_hurst(r) and hurst_bypass_ok)
                        for r in reasons)
                and (any(_r_is_timing(r) for r in reasons)
                     or any(_r_is_vol(r) for r in reasons)
                     or any(_r_is_hurst(r) for r in reasons))
            )
            qualifies_for_eval = is_price_only or blockers_subset_acceptable
            if qualifies_for_eval:
                # tp_cached_now was hoisted above
                if tp_cached_now and not tp_cached_now.get("error"):
                    tp_p = tp_cached_now["bs_p_above"]
                    combined_p = ev_tp_weight * tp_p + (1 - ev_tp_weight) * sig["p_yes"]
                    src = f"M={sig['p_yes']:.3f} TP={tp_p:.3f} w={ev_tp_weight:.2f}"
                else:
                    combined_p = sig["p_yes"]
                    src = f"M={sig['p_yes']:.3f} (TP unavailable)"

                # â”€â”€ 2026-06-16: OKX-MOMENTUM CONFIDENCE BOOST â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # Residual analysis (369 windows): OKX 6s futures momentum is
                # the ONE auxiliary signal that stably predicts the model's
                # residual (corr +0.09, +5pp confirm/oppose split, consistent in
                # BOTH temporal halves â€” unlike the book/flow/liq composites,
                # which were 15m-noise, and mkt_dev, which was regime-unstable).
                # Nudge combined_p in OKX's direction (a small, capped boost), so
                # the EV gate / Kelly get the extra confidence when the lead
                # venue is moving with the trade. If a flip results, the
                # existing direction-conflict check below safely skips it.
                if okx_boost_enabled and okx_feed is not None:
                    try:
                        _okxb = okx_feed.get_recent_move(lookback_s=futures_lead_lookback_s)
                        _okx_mv = (_okxb or {}).get("move_pct")
                        if _okx_mv is not None and okx_boost_scale > 0:
                            _boost = okx_boost_weight * math.tanh(_okx_mv / okx_boost_scale)
                            _cp_old = combined_p
                            combined_p = min(0.99, max(0.01, combined_p + _boost))
                            src += f" +okx{_boost:+.3f}"
                            if abs(combined_p - _cp_old) >= 0.005:
                                log.info(f"   OKX-BOOST: combined_p {_cp_old:.3f}\u2192"
                                         f"{combined_p:.3f} (okx 6s={_okx_mv:+.4f}%, "
                                         f"boost {_boost:+.3f})")
                    except Exception as _okxe:
                        log.debug(f"   OKX-boost failed (non-fatal): {_okxe!r}")

                # â”€â”€ ChainVector MOMENTUM-SCORECARD EV BOOST â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # /momentum aggregates vol-normalized momentum across venues
                # into one scorecard (aggregate_score âˆ’100..+100, breadth_up,
                # dispersion). When the cross-venue scorecard agrees with the
                # trade direction, nudge combined_p the same way the OKX boost
                # does â€” capped, tanh-saturated, direction-conflict-safe (the
                # check below skips any flip). Zero extra API cost: the lead
                # feed already holds the latest scorecard.
                if cv_mom_boost_enabled and cv_lead_hub is not None:
                    try:
                        _mom = cv_lead_hub.latest_momentum()
                        _agg = None if _mom is None else _mom.get("aggregate_score")
                        if _agg is not None and cv_mom_boost_scale > 0:
                            _cvb = cv_mom_boost_weight * math.tanh(
                                float(_agg) / cv_mom_boost_scale)
                            _cp_old = combined_p
                            combined_p = min(0.99, max(0.01, combined_p + _cvb))
                            src += f" +cvmom{_cvb:+.3f}"
                            if abs(combined_p - _cp_old) >= 0.005:
                                log.info(f"   CV-MOM BOOST: combined_p {_cp_old:.3f}\u2192"
                                         f"{combined_p:.3f} (agg={float(_agg):+.1f}, "
                                         f"breadth_up={_mom.get('breadth_up')}, "
                                         f"boost {_cvb:+.3f})")
                    except Exception as _cvme:
                        log.debug(f"   CV-mom boost failed (non-fatal): {_cvme!r}")

                new_side_is_yes = combined_p > 0.5
                new_rec         = "YES" if new_side_is_yes else "NO"
                # Markov's IMPLIED direction (from p_yes) â€” `rec` is "NO_TRADE"
                # whenever deterministic gates fail, regardless of which way
                # the model leans. We compare against the underlying signal.
                markov_implied  = "YES" if sig["p_yes"] > 0.5 else "NO"

                if new_rec != markov_implied:
                    log.info(
                        f"   EV-gate: combined_p={combined_p:.3f} points {new_rec} "
                        f"but Markov leans {markov_implied} (p_yes={sig['p_yes']:.3f}) "
                        f"â€” direction conflict, skip override"
                    )
                else:
                    p_win   = combined_p if new_side_is_yes else 1 - combined_p
                    p_d     = limit_price / 100
                    fee_c   = MAKER_FEE_RATE * p_d * (1 - p_d)
                    cost_c  = p_d + fee_c
                    net_win = (1 - p_d) - fee_c
                    ev_per  = p_win * net_win - (1 - p_win) * cost_c

                    # Strong-signal relaxation (Option A):
                    # When all of these hold, use the lower `ev_strong_floor`
                    # instead of the standard `ev_floor`. Bets that high
                    # consensus across signals + reasonable price = the
                    # model is systematically conservative.
                    # TP must lean MEANINGFULLY in trade direction, not just
                    # above/below 50%. Loss on 2026-05-27 21:24 (YES @ 68Â¢,
                    # -$43): TP was 52.7% â€” barely above coin-flip â€” and the
                    # trade was a dead-cat bounce. Requiring TP â‰¥ 0.55 (or
                    # â‰¤ 0.45 for NO) filters out these marginal cases.
                    yes_lean = sig["p_yes"] > 0.5
                    tp_p = (tp_cached_now or {}).get("bs_p_above")
                    tp_meaningful = (
                        tp_p is not None
                        and (
                            (yes_lean and tp_p >= ev_strong_tp_min)
                            or (not yes_lean and tp_p <= (1 - ev_strong_tp_min))
                        )
                    )
                    # Strong-floor mins_left cap (2026-05-27 PM update): only
                    # allow strong-floor in the LATE part of the entry window.
                    # Today's -$162 cluster (two losses 16 min apart) both
                    # fired at 11-12 min left â€” classic "mid-market reversal"
                    # pattern. Capping strong-floor to mins_left â‰¤ 8 forces
                    # signal to develop further before we commit.
                    strong_timing_ok = sig["minutes_left"] <= ev_strong_max_mins
                    # Recent 6-bar net momentum check (2026-05-27 PM update):
                    # block strong-floor when the cumulative recent_5m_pct
                    # is strongly opposite to trade direction. Today's two
                    # losses both had net -0.21% over 6 bars but bought YES
                    # (Markov caught a brief bounce in last 2 bars). The
                    # threshold filters out "dead-cat bounce" entries.
                    recent_sum = sum(recent_5m or [])
                    if yes_lean:
                        momentum_ok = recent_sum >= -ev_strong_max_adverse_momentum
                    else:
                        momentum_ok = recent_sum <= ev_strong_max_adverse_momentum
                    is_strong = (
                        sig["gap"] >= ev_strong_gap_min
                        and limit_price <= ev_strong_price_max
                        and tp_cached_now is not None
                        and not tp_cached_now.get("error")
                        and tp_meaningful
                        and strong_timing_ok
                        and momentum_ok
                    )

                    # Late-window-sure tier (Option B): late in the entry
                    # window, when TP strongly confirms direction AND Markov
                    # leans the same way AND recent 5m momentum doesn't
                    # disagree, we allow bidding much higher (default 89Â¢) at
                    # a more lenient EV floor (default -$0.10/c). When this
                    # fires we also boost p_win using TP (which is the most
                    # forward-looking signal at this stage).
                    # â”€â”€ Orderbook lock-in confirmation (2026-05-28 PM) â”€â”€â”€â”€â”€
                    # Compute once here so it can drive both LATE-SURE's
                    # effective cap AND HC's TP-bypass below.
                    ob_lockin_confirmed = False
                    ob_lockin_reason: Optional[str] = None
                    if orderbook_lockin_enabled and ob_snap:
                        ob_lockin_confirmed, ob_lockin_reason = _is_orderbook_lockin(
                            orderbook=ob_snap,
                            new_rec=new_rec,
                            spread_max=orderbook_lockin_spread_max,
                            price_min=orderbook_lockin_price_min,
                            gap=sig.get("gap", 0.0),
                            gap_min=orderbook_lockin_gap_min,
                        )

                    # When the orderbook confirms lock-in in our direction,
                    # raise LATE-SURE's effective cap by 1Â¢ (98 â†’ 99). The
                    # orderbook is orthogonal evidence of market consensus
                    # at the absolute extreme â€” justifies the 1Â¢ extra risk.
                    effective_late_window_price_max = (
                        late_window_price_max_lockin if ob_lockin_confirmed
                        else late_window_price_max
                    )

                    is_late_sure, late_reason = _is_late_window_sure(
                        mins_left=sig["minutes_left"],
                        limit_price=limit_price,
                        tp_cached=tp_cached_now,
                        new_rec=new_rec,
                        markov_p_yes=sig["p_yes"],
                        recent_5m=recent_5m,
                        late_window_mins=late_window_mins,
                        late_window_price_max=effective_late_window_price_max,
                        late_window_min_tp=late_window_min_tp,
                        # 2026-05-28 PM: lock-in confirmed â†’ bypass TP
                        # threshold (mirrors HC behavior).
                        orderbook_lockin_bypass=ob_lockin_confirmed,
                    )

                    # High-conviction signal-based tier: independent of time,
                    # activated when gap+persist+TP all hit extremes. Lifts
                    # the ceiling to 97Â¢ but REQUIRES positive EV at that
                    # price (no negative-EV tolerance â€” risk:reward is 30+:1
                    # at high prices, no margin for probability miscalibration).
                    # 2026-05-28 PM: HC's TP threshold is BYPASSED when the
                    # orderbook confirms lock-in in our direction. Other HC
                    # checks (gap, persist, mins, price, momentum) still apply.
                    is_high_conv, hc_reason = _is_high_conviction(
                        mins_left=sig["minutes_left"],
                        limit_price=limit_price,
                        tp_cached=tp_cached_now,
                        new_rec=new_rec,
                        sig=sig,
                        recent_5m=recent_5m,
                        high_conv_gap_min=high_conv_gap_min,
                        high_conv_persist_min=high_conv_persist_min,
                        high_conv_tp_strong=high_conv_tp_strong,
                        high_conv_price_max=high_conv_price_max,
                        high_conv_max_mins=high_conv_max_mins,
                        orderbook_lockin_bypass=ob_lockin_confirmed,
                        hc_low_hurst_veto_enabled=hc_low_hurst_veto_enabled,
                        hc_low_hurst_threshold=hc_low_hurst_threshold,
                        hc_low_hurst_markov_extremity=hc_low_hurst_markov_extremity,
                        dist_pct=mkt.get("dist_pct"),
                        hc_dist_min=hc_dist_min,
                    )

                    # â”€â”€ HIGH-CONV split-externals block â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    # 2026-05-27 PM analysis: today's only HIGH-CONV loss
                    # caught by SL had lead-venue -0.01% / OKX +0.003% â€”
                    # one opposed, one weakly agreed. The HIGH-CONV tier
                    # ALREADY requires gap+persist+TP at extremes; when
                    # externals send a mixed signal on TOP of that, the
                    # internal "consensus" is fragile.
                    #
                    # Block when: one external directionally opposes the
                    # trade (any non-zero magnitude) AND the other doesn't
                    # also oppose (i.e., is neutral OR weakly agrees).
                    # XOR logic â€” both-oppose is caught by consensus veto;
                    # both-agree or both-neutral pass through.
                    #
                    # Only applies to HIGH-CONV tier â€” STANDARD/strong/
                    # late-* tiers have lower price ceilings and more EV
                    # cushion, so split externals don't kill them.
                    #
                    # 2026-05-29 BUG FIX: skip when orderbook lock-in is
                    # confirmed. The lock-in pattern (spreadâ‰¤2, top-of-book
                    # â‰¥85Â¢ in trade direction, gapâ‰¥0.20) is itself a third-
                    # source orthogonal confirmation â€” exactly what the split
                    # rule was added to compensate for. Stacking another
                    # opposition check on top blocks clean lock-in trades
                    # like 2026-05-28 23:38 NO @ 87Â¢ (db=+0.0102% = 1bp of
                    # noise, okx=0). With ob-lockin confirmed + Markov-only
                    # p_win, that trade would fire HC with EV=-$0.08 within
                    # the -$0.10 lock-in floor.
                    if (is_high_conv and hc_block_on_split_externals
                            and not ob_lockin_confirmed
                            and futures_lead is not None and okx_feed is not None):
                        try:
                            hc_db_now = futures_lead.get_recent_move(
                                lookback_s=futures_lead_lookback_s)
                            hc_okx_now = okx_feed.get_recent_move(
                                lookback_s=consensus_okx_lookback_s)
                            hc_db_move = (hc_db_now or {}).get("move_pct")
                            hc_okx_move = (hc_okx_now or {}).get("move_pct")
                            hc_side_yes = (new_rec == "YES")
                            def _hc_opposes_trade(m, side_yes):
                                # Any non-zero opposing move counts (much
                                # stricter than the 0.005% consensus veto)
                                if m is None or m == 0: return False
                                return (m < 0) if side_yes else (m > 0)
                            db_opp = _hc_opposes_trade(hc_db_move, hc_side_yes)
                            okx_opp = _hc_opposes_trade(hc_okx_move, hc_side_yes)
                            # Split if XOR: exactly one opposes. Both-oppose
                            # is handled by consensus veto; both-non-oppose
                            # is fine.
                            is_split = (db_opp != okx_opp)
                            # Require BOTH feeds present (don't block on
                            # one-source data â€” too easy to false-trigger)
                            both_present = (hc_db_move is not None
                                            and hc_okx_move is not None)
                            if is_split and both_present:
                                log.info(
                                    f"   HIGH-CONV REJECTED â€” split externals: "
                                    f"db={hc_db_move:+.4f}% okx={hc_okx_move:+.4f}% "
                                    f"(one opposes {new_rec}, other doesn't). "
                                    f"Falling through to lower tiers."
                                )
                                is_high_conv = False
                                hc_reason = "split_externals_block"
                        except Exception as e:
                            log.debug(f"   HC split-externals check failed: {e}")

                    # Late-window directional tier: looser TP than high-conv
                    # but only late in window. Captures developing directional
                    # moves where signals confirm direction but TP isn't yet
                    # at the high-conv extreme. Uses combined_p directly (no
                    # TP boost) for conservatism.
                    is_late_dir = False
                    late_dir_disqual_reason: Optional[str] = None
                    if late_dir_enabled:
                        tp_for_check2 = None
                        if tp_cached_now is not None and not tp_cached_now.get("error"):
                            tp_for_check2 = tp_cached_now.get("bs_p_above")
                        is_late_dir, late_dir_disqual_reason = _is_late_window_directional(
                            mins_left=sig["minutes_left"],
                            gap=sig.get("gap", 0.0),
                            persist=sig.get("persist", 0.0),
                            tp_p_yes=tp_for_check2,
                            new_rec=new_rec,
                            btc_price=mkt["btc_price"],
                            strike=mkt["strike"],
                            recent_5m=recent_5m,
                            late_dir_mins=late_dir_mins,
                            late_dir_gap_min=late_dir_gap_min,
                            late_dir_persist_min=late_dir_persist_min,
                            late_dir_distance_min=late_dir_distance_min,
                            late_dir_momentum_min=late_dir_momentum_min,
                            late_dir_strong_distance=late_dir_strong_distance,
                        )

                    # Pick the most-permissive applicable tier:
                    #   late-sure   â†’ TP-leaning p_win, lenient floor, cap 89
                    #   high-conv   â†’ TP-leaning p_win, positive-EV only, cap 97
                    #   strong      â†’ standard combined_p, mild lenient floor, cap 80
                    #   default     â†’ standard combined_p, standard floor, cap 90
                    # Ordering rationale: late-sure has the most lenient floor
                    # (allows -$0.10 EV) so it wins below 89Â¢ if applicable;
                    # high-conv takes over for 89-97Â¢ when signals are extreme;
                    # strong-floor handles the strong-consensus-at-moderate-price
                    # band; default catches everything else.
                    # When timing was a blocker, ONLY high-conv can use the
                    # bypass. When vol was a blocker, high-conv, late-dir, OR
                    # late-sure (new 2026-05-28) can bypass. When Hurst was
                    # blocked, high-conv or late-sure can bypass. Other tiers
                    # stay locked to the standard 3-10min entry window,
                    # standard vol gate, and Hurst gate.
                    timing_was_blocked = any(_r_is_timing(r) for r in reasons)
                    vol_was_blocked    = any(_r_is_vol(r) for r in reasons)
                    hurst_was_blocked  = any(_r_is_hurst(r) for r in reasons)
                    # `needs_high_conv` (original semantics): trade requires a
                    # tier that can bypass timing OR vol. HIGH-CONV bypasses
                    # both; LATE-DIR bypasses vol only. 2026-05-28: LATE-SURE
                    # also bypasses vol now (see vol_bypass_ok wiring above).
                    needs_high_conv    = timing_was_blocked or vol_was_blocked
                    needs_strong_tier  = needs_high_conv or hurst_was_blocked

                    # 2026-05-28: LATE-SURE can fire when:
                    #   â€¢ timing was NOT blocked (timing still requires HC), AND
                    #   â€¢ either no vol-block OR vol-block was already authorized
                    #     by late_sure_qualifies via vol_bypass_ok.
                    # Equivalent: late-sure fires when timing is fine and the
                    # poll reached the EV gate in the first place. We restate
                    # via late_sure_vol_authorized for clarity.
                    late_sure_vol_authorized = (
                        not vol_was_blocked
                        or (late_sure_vol_bypass and late_sure_qualifies)
                    )
                    late_sure_can_fire = (
                        is_late_sure
                        and not timing_was_blocked
                        and late_sure_vol_authorized
                    )

                    # late-sure can bypass Hurst AND vol (per 2026-05-28
                    # design), but still can't bypass timing â€” that requires
                    # high-conv. The vol-bypass authorization is captured in
                    # late_sure_can_fire above.
                    # Compute Markov-only p_win for use when lock-in bypass
                    # is active. Why: when TP is structurally undershooting
                    # (Deribit 13h IV applied to 15min binary) and we bypass
                    # the TP threshold via orderbook lock-in, we shouldn't
                    # then USE that low TP to compute p_win. Markov + the
                    # orderbook itself are the trustworthy signals here.
                    markov_p_dir = (sig["p_yes"] if new_side_is_yes
                                    else 1.0 - sig["p_yes"])
                    if late_sure_can_fire:
                        if ob_lockin_confirmed:
                            # 2026-05-28 PM: lock-in path â€” use Markov-only
                            # (or combined, whichever is HIGHER). TP-boost is
                            # skipped because TP is what we're bypassing.
                            p_win = max(p_win, markov_p_dir)
                        else:
                            # Standard LATE-SURE path: boost with TP since
                            # TP-extreme is what got us here.
                            tp_p_dir = (tp_cached_now["bs_p_above"]
                                        if new_side_is_yes
                                        else 1.0 - tp_cached_now["bs_p_above"])
                            p_win = max(p_win, tp_p_dir)
                        ev_per = p_win * net_win - (1 - p_win) * cost_c
                        floor_to_use = late_window_ev_floor
                        floor_label  = "late-sure-floor"
                        # 2026-05-28 PM: when orderbook lock-in is confirmed,
                        # use the elevated cap (98 â†’ 99 by default).
                        effective_ceiling = effective_late_window_price_max
                        active_tier       = "late_sure"
                    elif is_high_conv:
                        # 2026-05-28 PM:
                        #   â€¢ Standard HC path: boost p_win using TP (since
                        #     TP-extreme is required).
                        #   â€¢ Lock-in HC path: use Markov-only (or max with
                        #     combined). TP is being bypassed so we don't
                        #     trust it for p_win either. Floor also relaxes
                        #     to hc_lockin_ev_floor (default -$0.10, matches
                        #     LATE-SURE) since the orderbook confirmation
                        #     warrants the same EV cushion.
                        if ob_lockin_confirmed:
                            p_win = max(p_win, markov_p_dir)
                            floor_to_use = hc_lockin_ev_floor
                            floor_label  = "hc-lockin-floor"
                        else:
                            if tp_cached_now and not tp_cached_now.get("error"):
                                tp_p_dir = (tp_cached_now["bs_p_above"]
                                            if new_side_is_yes
                                            else 1.0 - tp_cached_now["bs_p_above"])
                                p_win = max(p_win, tp_p_dir)
                            floor_to_use = high_conv_ev_floor
                            floor_label  = "high-conv-floor"
                        ev_per = p_win * net_win - (1 - p_win) * cost_c
                        effective_ceiling = high_conv_price_max
                        active_tier       = "high_conv"
                    elif is_late_dir and not hurst_was_blocked:
                        # Late-window-directional: uses combined_p directly
                        # (no TP boost) for conservatism. Tier can bypass
                        # vol/timing like high-conv but NOT Hurst (late-dir
                        # has looser TP requirements than late-sure, and
                        # mean-reverting regimes are too risky for it).
                        floor_to_use = late_dir_ev_floor
                        floor_label  = "late-dir-floor"
                        effective_ceiling = late_dir_price_max
                        active_tier       = "late_dir"
                    elif is_strong and (
                        not needs_strong_tier
                        or (strong_floor_hurst_bypass
                            and not timing_was_blocked
                            and not vol_was_blocked)
                    ):
                        # 2026-05-28 PM3: strong-floor can fire when ONLY Hurst
                        # was blocked (timing/vol still excluded). Markov-only
                        # p_win isn't applied here â€” strong-floor keeps its
                        # combined p_win since gapâ‰¥0.13 isn't "extreme enough"
                        # to trust Markov alone over TP+Markov blend.
                        floor_to_use = ev_strong_floor
                        floor_label  = "strong-floor"
                        effective_ceiling = ev_ceiling
                        active_tier       = "strong"
                    elif needs_strong_tier:
                        # Only high-conv (or late-sure for Hurst) may bypass
                        # timing/vol/Hurst. If we got here, one of those was
                        # a blocker but no eligible tier qualified.
                        which_parts = []
                        if timing_was_blocked: which_parts.append("timing")
                        if vol_was_blocked:    which_parts.append("vol")
                        if hurst_was_blocked:  which_parts.append("hurst")
                        which = "+".join(which_parts)
                        log.info(
                            f"   EV-gate: {which}-bypass requires high-conv "
                            f"or late-sure tier (blocked by: {hc_reason or 'n/a'}) "
                            f"â€” skip override"
                        )
                        active_tier       = "none"
                    else:
                        floor_to_use = ev_floor
                        floor_label  = "floor"
                        effective_ceiling = ev_ceiling
                        active_tier       = "standard"

                    if active_tier == "none":
                        pass  # already logged above, fall through to NO TRADE
                    elif limit_price > effective_ceiling:
                        ceiling_tag = ({"late_sure":"late-sure",
                                        "high_conv":"high-conv",
                                        "late_dir":"late-dir",
                                        "strong":"safety",
                                        "standard":"safety"}[active_tier])
                        log.info(
                            f"   EV-gate: price {limit_price}Â¢ exceeds "
                            f"{ceiling_tag} ceiling "
                            f"{effective_ceiling}Â¢ â€” skip "
                            f"(combined_p={combined_p:.3f}, EV=${ev_per:+.4f}/c)"
                        )
                        # Tell the user when a higher tier was ALMOST eligible.
                        # This surfaces "would have qualified except price"
                        # for high-conv (the most common case at 98+Â¢).
                        if active_tier == "standard":
                            # Check if high-conv conditions would have held at
                            # a hypothetical price within the high-conv cap.
                            hc_ok_except_price, hc_why_not = _is_high_conviction(
                                mins_left=sig["minutes_left"],
                                limit_price=min(limit_price, high_conv_price_max),
                                tp_cached=tp_cached_now,
                                new_rec=new_rec,
                                sig=sig,
                                recent_5m=recent_5m,
                                high_conv_gap_min=high_conv_gap_min,
                                high_conv_persist_min=high_conv_persist_min,
                                high_conv_tp_strong=high_conv_tp_strong,
                                high_conv_price_max=high_conv_price_max,
                                high_conv_max_mins=high_conv_max_mins,
                                hc_low_hurst_veto_enabled=hc_low_hurst_veto_enabled,
                                hc_low_hurst_threshold=hc_low_hurst_threshold,
                                hc_low_hurst_markov_extremity=hc_low_hurst_markov_extremity,
                                dist_pct=mkt.get("dist_pct"),
                                hc_dist_min=hc_dist_min,
                            )
                            if hc_ok_except_price and limit_price > high_conv_price_max:
                                log.info(
                                    f"   (high-conv conditions ALL met â€” "
                                    f"only price blocked: {limit_price}Â¢ > {high_conv_price_max}Â¢. "
                                    f"Tune --high-conv-price-max to raise.)"
                                )
                    elif ev_per < floor_to_use:
                        if is_late_sure:
                            tier_detail = (f"  [late-sure: mins={sig['minutes_left']:.1f}, "
                                          f"TP={tp_cached_now['bs_p_above']:.2f}, p_win={p_win:.3f}]")
                        elif is_high_conv:
                            tp_str = (f"{tp_cached_now['bs_p_above']:.3f}"
                                      if tp_cached_now and not tp_cached_now.get("error")
                                      else "n/a")
                            bypass_str = " ob-bypass" if ob_lockin_confirmed else ""
                            tier_detail = (f"  [high-conv{bypass_str}: gap={sig['gap']:.3f}â‰¥{high_conv_gap_min:.2f}, "
                                          f"persist={sig.get('persist',0):.2f}, "
                                          f"TP={tp_str}, p_win={p_win:.3f}]")
                        elif is_late_dir:
                            tier_detail = (f"  [late-dir: mins={sig['minutes_left']:.1f}, "
                                          f"gap={sig['gap']:.3f}â‰¥{late_dir_gap_min:.2f}, "
                                          f"persist={sig.get('persist',0):.2f}, "
                                          f"TP={tp_cached_now['bs_p_above']:.3f}]")
                        elif is_strong:
                            tier_detail = (f"  [strong: gap={sig['gap']:.3f}â‰¥{ev_strong_gap_min:.2f}, "
                                          f"priceâ‰¤{ev_strong_price_max}Â¢, TP confirms]")
                        else:
                            tier_detail = ""
                        log.info(
                            f"   EV-gate: EV ${ev_per:+.4f}/c below {floor_label} "
                            f"${floor_to_use:+.3f}/c at {limit_price}Â¢  ({src})"
                            + tier_detail
                        )
                    else:
                        # Kelly sizing tiers (same as run_signal) â€” anti-Kelly at high prices
                        if   65 <= limit_price <= 73: frac = 0.35
                        elif 73 <  limit_price <= 79: frac = 0.12
                        elif 79 <  limit_price <= 85: frac = 0.08
                        else:                         frac = 0.05

                        # Externals-confirmed sizing boost. Two paths:
                        #   â€¢ HIGH-CONV tier: set frac to high_conv_confirmed_frac
                        #     (default 0.10 = 2x baseline 0.05)
                        #   â€¢ Non-HC tiers (standard/strong/late-sure/late-dir):
                        #     multiply frac by standard_confirmed_boost (default
                        #     2.0). Historical 5/27 data: standard tier produced
                        #     +$87 vs high-conv's -$27 on the day. The standard
                        #     tier deserves the same external-confirm sizing
                        #     respect we already gave HIGH-CONV.
                        # Both paths require: BOTH externals available AND neither
                        # opposes the trade direction.
                        hc_size_tag = ""
                        if okx_feed is not None:
                            try:
                                hc_db_now = (futures_lead.get_recent_move(
                                    lookback_s=futures_lead_lookback_s)
                                    if futures_lead is not None else None)
                                hc_okx_now = okx_feed.get_recent_move(
                                    lookback_s=consensus_okx_lookback_s)
                                hc_db_move = (hc_db_now or {}).get("move_pct")
                                hc_okx_move = (hc_okx_now or {}).get("move_pct")
                                hc_side_yes = (new_rec == "YES")
                                def _hc_dir(m, thr=consensus_min_move_pct):
                                    return (m is not None and abs(m) >= thr)
                                def _hc_opp(m, side_yes):
                                    if m is None: return False
                                    return (m < 0) if side_yes else (m > 0)
                                def _hc_fmt(m):
                                    return f"{m:+.4f}%" if m is not None else "stale"
                                hc_db_opp  = _hc_dir(hc_db_move) and _hc_opp(hc_db_move,  hc_side_yes)
                                hc_okx_opp = _hc_dir(hc_okx_move) and _hc_opp(hc_okx_move, hc_side_yes)
                                # 2026-06-27: OKX-as-primary fallback. Normally
                                # BOTH externals must be present and neither oppose.
                                # When CME (Databento) is stale â€” e.g. Fri-Sun
                                # close â€” fall back to OKX-only confirmation so
                                # HIGH-CONV/STD trades still earn the externals-
                                # confirmed sizing boost on weekends. OKX BTC-USDT-
                                # SWAP is 24/7. We never fall back to CME-only (CME
                                # is the one that disappears), and a single
                                # OPPOSING source still blocks the boost.
                                db_present  = hc_db_move is not None
                                okx_present = hc_okx_move is not None
                                if db_present and okx_present:
                                    externals_confirm = (not hc_db_opp and not hc_okx_opp)
                                    _confirm_src = "db+okx"
                                elif okx_present and not db_present:
                                    externals_confirm = (not hc_okx_opp)
                                    _confirm_src = "okx-only(CME stale)"
                                else:
                                    externals_confirm = False
                                    _confirm_src = "none"

                                if externals_confirm:
                                    if active_tier == "high_conv":
                                        # HC-confirmed: set frac to fixed value
                                        if high_conv_confirmed_frac > frac:
                                            frac = high_conv_confirmed_frac
                                            hc_size_tag = (f" (HC-confirmed boost [{_confirm_src}]: "
                                                           f"frac={frac:.2f}, db={_hc_fmt(hc_db_move)}, "
                                                           f"okx={_hc_fmt(hc_okx_move)})")
                                    elif active_tier in ("standard", "strong", "late_sure", "late_dir"):
                                        # STANDARD-confirmed: multiply existing
                                        # frac by configurable multiplier. MAX_TRADE_PCT
                                        # cap (20%) protects against extreme sizing
                                        # for golden zone (frac 0.35 Ã— 2 = 0.70).
                                        if standard_confirmed_boost > 1.0:
                                            old_frac = frac
                                            frac = old_frac * standard_confirmed_boost
                                            hc_size_tag = (f" (STD-confirmed boost [{_confirm_src}]: "
                                                           f"frac {old_frac:.2f}â†’{frac:.2f} "
                                                           f"({standard_confirmed_boost:.1f}x), "
                                                           f"db={_hc_fmt(hc_db_move)}, "
                                                           f"okx={_hc_fmt(hc_okx_move)})")
                            except Exception as e:
                                log.debug(f"  Externals-confirmed sizing boost check failed: {e}")

                        b          = net_win / cost_c if cost_c > 0 else 1.0
                        kelly_full = max(0.0, (b * p_win - (1 - p_win)) / b)
                        risk_pct   = min(MAX_TRADE_PCT, frac * kelly_full)

                        # Floor-stake for negative-EV overrides:
                        # Kelly gives 0 when EV < 0, so without this the trade
                        # would default to the `max(1, â€¦)` floor of one contract.
                        # Relaxed-floor tiers (strong-floor, late-sure) explicitly
                        # accept some negative EV â€” they should size to a small,
                        # fixed % of bankroll instead of 1c. We take the MAX of
                        # Kelly and the floor-stake so a strongly +EV signal can
                        # still scale up normally.
                        # 2026-05-28 PM4 BUG FIX: floor-stake selection now
                        # gates on `active_tier` (what actually fires) rather
                        # than the is_X helper bools (which can both be True
                        # when a signal qualifies for multiple tiers). Pre-fix,
                        # an HC-bypass:ob-lockin trade would get strong-floor's
                        # 1.0% min-stake because `is_strong` is also True for
                        # those signals â€” the elif `is_strong` matched first.
                        floor_stake_label = ""
                        if active_tier == "late_sure" and kelly_full <= 0.0:
                            risk_pct = max(risk_pct, late_sure_min_stake_pct)
                            floor_stake_label = f" (late-sure min-stake {late_sure_min_stake_pct*100:.2f}%)"
                        elif (active_tier == "high_conv" and ob_lockin_confirmed
                              and kelly_full <= 0.0):
                            risk_pct = max(risk_pct, hc_lockin_min_stake_pct)
                            floor_stake_label = f" (hc-lockin min-stake {hc_lockin_min_stake_pct*100:.2f}%)"
                        elif active_tier == "strong" and kelly_full <= 0.0:
                            risk_pct = max(risk_pct, strong_floor_min_stake_pct)
                            floor_stake_label = f" (strong-floor min-stake {strong_floor_min_stake_pct*100:.2f}%)"
                        floor_stake_label = floor_stake_label + hc_size_tag

                        dyn_cap    = max(25, round(session.effective_bankroll() / 200 * 25))
                        new_contracts = min(
                            max(1, round(session.effective_bankroll() * risk_pct / cost_c)),
                            dyn_cap,
                        )

                        # Add bypass annotation for tiers that bypassed vol/timing/Hurst
                        bypass_tags = []
                        if active_tier in ("high_conv", "late_dir"):
                            if timing_was_blocked:
                                bypass_tags.append("timing")
                            if vol_was_blocked:
                                bypass_tags.append("vol")
                        if active_tier in ("high_conv", "late_sure"):
                            if hurst_was_blocked:
                                bypass_tags.append("hurst")
                        # 2026-05-28 PM3: strong-floor can also bypass hurst
                        if active_tier == "strong" and hurst_was_blocked:
                            bypass_tags.append("hurst")
                        # 2026-05-28 PM: tag orderbook-lockin contribution
                        if ob_lockin_confirmed and active_tier in ("high_conv", "late_sure"):
                            bypass_tags.append("ob-lockin")
                        bypass_str = (f" bypass:{'+'.join(bypass_tags)}" if bypass_tags else "")
                        tier_tag = (f" [LATE-SURE{bypass_str}]" if active_tier == "late_sure"
                                    else f" [HIGH-CONV{bypass_str}]" if active_tier == "high_conv"
                                    else f" [LATE-DIR{bypass_str}]"  if active_tier == "late_dir"
                                    else " [STRONG-FLOOR]" if active_tier == "strong"
                                    else "")
                        log.info(
                            f"   EV-GATE OVERRIDE{tier_tag} "
                            f"â€” price cap {limit_price}Â¢ relaxed "
                            f"(was hard cap 72Â¢ YES / 65Â¢ NO). "
                            f"combined_p={combined_p:.3f} ({src}), "
                            f"p_win={p_win:.3f}, EV=${ev_per:+.4f}/c "
                            f"({floor_label} ${floor_to_use:+.3f}, "
                            f"ceiling {effective_ceiling}Â¢), "
                            f"sizingâ†’{new_contracts}c{floor_stake_label}"
                        )
                        approved  = True
                        reasons   = []
                        rec       = new_rec
                        contracts = new_contracts
                        # 2026-05-29 BUG FIX: remember WHICH floor approved
                        # this trade. Downstream lead-cents check needs to
                        # use the same floor (was hardcoded to strong-floor,
                        # which blocked LATE-SURE lead-walks incorrectly).
                        approved_floor = floor_to_use

        if not approved:
            reason_str = " | ".join(reasons) if reasons else "no edge"
            log.info(f"NO TRADE â€” {reason_str}")

            # Audit: capture the full state of this rejection
            # Snapshot both lead venues (binance_futures + OKX) on EVERY poll
            # so we can analyze their
            # signal value vs trade outcomes (not just on trade-attempts).
            # No API cost â€” both get_recent_move() calls read from in-memory
            # deques maintained by background threads.
            db_snapshot = None
            if futures_lead is not None:
                try:
                    move_data = futures_lead.get_recent_move(
                        lookback_s=futures_lead_lookback_s
                    )
                    db_snapshot = {
                        "consistent":  True,  # not vetoing on NO_TRADE polls
                        "reason":      "snapshot_only",
                        "connected":   move_data.get("connected"),
                        "front_month": move_data.get("front_month"),
                        "stale_s":     move_data.get("stale_s"),
                        "n_ticks":     move_data.get("n_ticks"),
                        "move_pct":    move_data.get("move_pct"),
                        "current_mid": move_data.get("current_mid"),
                        "past_mid":    move_data.get("past_mid"),
                    }
                except Exception:
                    db_snapshot = None  # don't let snapshot failure break audit

            okx_snapshot = None
            if okx_feed is not None:
                try:
                    okx_snapshot = okx_feed.get_recent_move(
                        lookback_s=futures_lead_lookback_s  # same lookback for apples-to-apples
                    )
                except Exception:
                    okx_snapshot = None

            audit.write(build_poll_record(
                window_id=window_id, ticker=ticker, close_dt=close_dt,
                decision="NO_TRADE", rec=None, reasons=reasons,
                signal=sig, market=mkt, recent_5m_pct=recent_5m,
                tp_cached=window_tp_cache.get(window_id),
                db_check=db_snapshot,
                ev_calc=None,
                params=audit_params,
                okx_check=okx_snapshot,
                orderbook=ob_snap,
                trade_flow=tr_snap,
            ))

            # Re-poll if every blocker is "transient" â€” depends on instantaneous
            # market state and may flip before the entry window closes:
            #   â€¢ timing       â€” by definition ticks down each second
            #   â€¢ price cap    â€” orderbook moves continuously
            #   â€¢ near-strike  â€” BTC oscillation around the strike
            # Anything else (Markov gap/persist, vol, Hurst, blocked hour,
            # building history) takes many minutes to shift, so it's terminal
            # for this window.
            def _is_transient(r: str) -> bool:
                # Always-transient: depend on price or clock and tick continuously
                if ("timing" in r
                        or "min outside" in r
                        or ("price " in r and "cap" in r)
                        or "near-strike" in r):
                    return True
                # Last-bar adverse: the "last 5m bar" rolls forward every
                # 5 minutes. A new bar can shift this from adverse to OK.
                # Treat as transient while there's enough time left.
                if "last-bar adverse" in r:
                    return mins_left > 4.0
                # Markov gap/persist develop as new 1-min returns arrive.
                # GK volatility recomputes as new 5-min candles arrive â€” every
                # ~5 min the rolling window shifts and a calmer/noisier candle
                # can flip the gate. Treat as transient while there's still
                # meaningful time left in the window (default: > 5 min).
                if ("Markov gap" in r
                        or "persist" in r
                        or "high vol" in r):
                    return mins_left > 5.0
                # Hurst is mathematically invariant within a single 15min window
                # (computed from 15min candle history), but the daemon should
                # KEEP POLLING because the high-conv / late-sure tiers can
                # bypass Hurst when extreme signal alignment justifies it.
                # Without this, Hurst-blocked windows never get a second look,
                # even if TP later swings to 0.85+ in the final minutes.
                if "mean-reverting" in r:
                    return mins_left > 5.0
                # blocked UTC hour, building history â†’ terminal (truly invariant)
                return False
            transient_only = bool(reasons) and all(_is_transient(r) for r in reasons)
            if transient_only and mins_left > 3.5:
                # In-window re-poll interval. Was 20s; reduced to 10s to catch
                # brief orderbook dips and signal developments that the 20s
                # cadence would miss. API budget at 6 polls/min is still ~6%
                # of Polymarket's 100/min limit â€” plenty of headroom.
                await asyncio.sleep(10)
            else:
                session.traded.add(window_id)
                await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
            continue

        # â”€â”€ ChainVector FUTURES LEAD VETO (binance_futures venue) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Last check before placing the order: did CME MBT futures just move
        # AGAINST our direction? If so, skip â€” Polymarket is likely to follow.
        # Veto is soft: stale/missing data â†’ don't veto (assume signal not
        # available, defer to Markov decision).
        # 2026-06-25: ITM lock-in veto exemption (see module constants).
        itm_lockin_exempt = False
        if ITM_LOCKIN_VETO_EXEMPT_ENABLED:
            _cp_now = combined_p if 'combined_p' in locals() else None
            _dp_now = mkt.get('dist_pct') if mkt else None
            if _cp_now is not None and _dp_now is not None:
                _signed_cushion = _dp_now if rec == 'YES' else -_dp_now
                if (_signed_cushion >= ITM_LOCKIN_EXEMPT_CUSHION_PCT
                        and _cp_now >= ITM_LOCKIN_EXEMPT_MIN_COMBINED_P):
                    itm_lockin_exempt = True
                    log.info(
                        f"   ITM-LOCKIN EXEMPT \u2014 signed cushion {_signed_cushion:+.3f}% "
                        f"\u2265{ITM_LOCKIN_EXEMPT_CUSHION_PCT:.2f}% AND combined_p={_cp_now:.3f}"
                        f"\u2265{ITM_LOCKIN_EXEMPT_MIN_COMBINED_P:.2f}: skip futures-lead & Hurst+TP vetoes."
                    )

        db_check: Optional[dict] = None
        if futures_lead is not None and not itm_lockin_exempt:
            db_check = futures_lead.is_signal_consistent(
                rec, lookback_s=futures_lead_lookback_s,
                veto_threshold_bps=futures_lead_veto_bps,
            )
            db_summary = (
                f"front={db_check.get('front_month','n/a')} "
                f"n_ticks={db_check.get('n_ticks',0)} "
                f"stale_s={db_check.get('stale_s')} "
                f"move={db_check.get('move_pct')}"
            )
            if not db_check["consistent"]:
                log.warning(
                    f"FUTURES-LEAD VETO â€” skipping {rec} trade. {db_check['reason']}. "
                    f"({db_summary})"
                )
                # Audit: record the veto with full context
                # Snapshot OKX for record too (consistent across audit rows)
                okx_snap_veto = None
                if okx_feed is not None:
                    try:
                        okx_snap_veto = okx_feed.get_recent_move(
                            lookback_s=futures_lead_lookback_s
                        )
                    except Exception:
                        okx_snap_veto = None
                audit.write(build_poll_record(
                    window_id=window_id, ticker=ticker, close_dt=close_dt,
                    decision="NO_TRADE", rec=rec,
                    reasons=[f"futures_lead_veto: {db_check['reason']}"],
                    signal=sig, market=mkt, recent_5m_pct=recent_5m,
                    tp_cached=window_tp_cache.get(window_id),
                    db_check=db_check, ev_calc=None,
                    params=audit_params,
                    okx_check=okx_snap_veto,
                    orderbook=ob_snap,
                    trade_flow=tr_snap,
                ))
                # 2026-05-29 BUG FIX: futures-lead veto is TRANSIENT, not terminal.
                # The opposing futures move can revert in 6-30 seconds. Re-poll
                # so we don't abandon the window after a single snapshot of
                # adverse externals. Same logic as consensus veto.
                if mins_left > 3.5:
                    await asyncio.sleep(10)
                else:
                    session.traded.add(window_id)
                    await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
                continue
            else:
                log.info(f"   Futures-lead OK â€” {db_check['reason']}  ({db_summary})")

        # â”€â”€ ChainVector MOMENTUM-SCORECARD VETO (strong-against) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # The /momentum aggregate is a cross-venue, vol-normalized read; when
        # it is STRONGLY against the trade direction AND breadth confirms
        # (most venues moving the same adverse way), entering is fighting the
        # whole tape, not one venue's blip. Softer disagreement is left to the
        # EV boost (which already nudged combined_p down). Fail-open on
        # missing/stale scorecard; transient re-poll like the lead veto.
        if (cv_mom_veto_enabled and cv_lead_hub is not None
                and not itm_lockin_exempt):
            try:
                _momv = cv_lead_hub.latest_momentum()
            except Exception:
                _momv = None
            _aggv = None if _momv is None else _momv.get("aggregate_score")
            _brv = None if _momv is None else _momv.get("breadth_up")
            if _aggv is not None:
                _signed_mom = float(_aggv) if rec == "YES" else -float(_aggv)
                _breadth_against = True
                if _brv is not None:
                    _breadth_against = ((float(_brv) <= 1.0 - cv_mom_veto_breadth)
                                        if rec == "YES"
                                        else (float(_brv) >= cv_mom_veto_breadth))
                if _signed_mom <= -cv_mom_veto_score and _breadth_against:
                    log.warning(
                        f"CV-MOMENTUM VETO â€” scorecard strongly against {rec}: "
                        f"agg={float(_aggv):+.1f} (signed {_signed_mom:+.1f} \u2264 "
                        f"-{cv_mom_veto_score:.0f}), breadth_up={_brv}. "
                        f"Skipping {rec} trade."
                    )
                    audit.write(build_poll_record(
                        window_id=window_id, ticker=ticker, close_dt=close_dt,
                        decision="NO_TRADE", rec=rec,
                        reasons=[f"cv_momentum_veto: agg={float(_aggv):+.1f} "
                                 f"breadth_up={_brv} against {rec}"],
                        signal=sig, market=mkt, recent_5m_pct=recent_5m,
                        tp_cached=window_tp_cache.get(window_id),
                        db_check=db_check, ev_calc=None,
                        params=audit_params,
                        orderbook=ob_snap, trade_flow=tr_snap,
                        extra={"cv_momentum": _momv},
                    ))
                    # Transient: cross-venue momentum can flip within a minute.
                    if mins_left > 3.5:
                        await asyncio.sleep(10)
                    else:
                        session.traded.add(window_id)
                        await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
                    continue

        # â”€â”€ ChainVector PROBABILITY-ENGINE VETO (losing-market floor) â”€â”€â”€â”€â”€â”€â”€â”€
        # The probability engine is already blended into combined_p as the EV
        # weight (--ev-tp-weight). This veto is the hard floor under that
        # blend: if the six-estimator ensemble says our side has â‰¤ N%
        # probability at the EXACT time-to-close, the market is priced to
        # lose regardless of what Markov sees. Fail-open when TP unavailable.
        if cv_prob_veto_enabled and not itm_lockin_exempt:
            _tpv = window_tp_cache.get(window_id)
            if _tpv and not _tpv.get("error"):
                _p_above = _tpv.get("bs_p_above")
                if isinstance(_p_above, (int, float)):
                    _p_side = float(_p_above) if rec == "YES" else 1.0 - float(_p_above)
                    if _p_side <= cv_prob_veto_max:
                        log.warning(
                            f"CV-PROB VETO â€” probability engine gives {rec} only "
                            f"{_p_side:.3f} (\u2264{cv_prob_veto_max:.2f}) at exact TTE. "
                            f"Ensemble says this market loses; skipping."
                        )
                        audit.write(build_poll_record(
                            window_id=window_id, ticker=ticker, close_dt=close_dt,
                            decision="NO_TRADE", rec=rec,
                            reasons=[f"cv_prob_veto: p_side={_p_side:.3f} "
                                     f"<= {cv_prob_veto_max:.2f}"],
                            signal=sig, market=mkt, recent_5m_pct=recent_5m,
                            tp_cached=_tpv, db_check=db_check, ev_calc=None,
                            params=audit_params,
                            orderbook=ob_snap, trade_flow=tr_snap,
                        ))
                        # Transient: TP refreshes at the 12/7/3-min checkpoints.
                        if mins_left > 3.5:
                            await asyncio.sleep(10)
                        else:
                            session.traded.add(window_id)
                            await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
                        continue

        # â”€â”€ 2026-06-01: HURST + TP DISAGREEMENT VETO (Gate B) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Catches the "chasing a fading top" pattern where:
        #   - Hurst is HIGH (strong-trending regime, prone to violent reversal)
        #   - TermProb (options market) significantly DISAGREES with Markov
        #     in the adverse direction (TP sees less confidence on our side)
        # Both must hold. Markov sees recent momentum but TP prices in the
        # reversal already.
        #
        # 7-day backtest justification:
        #   Hurst >= 0.80, |TP-M adverse| >= 0.05:
        #     catches 3 losses (-$402)  blocks 3 wins (+$37)   net +$365/week
        #
        # Gated on HC/STRONG tiers (the high-stake trades where this matters).
        if (hurst_tp_veto_enabled and active_tier in hurst_tp_veto_tiers
                and not itm_lockin_exempt):
            _h = sig.get('hurst')
            _tp_p = tp_p if 'tp_p' in locals() else (tp_cached.get('bs_p_above')
                                                     if tp_cached else None)
            if (_h is not None and isinstance(_tp_p, (int, float))):
                _markov_pyes = sig.get('p_yes', 0)
                # Adverse direction: TP says less of our direction than Markov
                if rec == 'YES':
                    _adv_diff = _markov_pyes - _tp_p
                else:
                    _adv_diff = _tp_p - _markov_pyes
                if _h >= hurst_tp_veto_min_hurst and _adv_diff >= hurst_tp_veto_min_diff:
                    log.warning(
                        f"HURST+TP VETO â€” Hurst={_h:.2f}\u2265{hurst_tp_veto_min_hurst:.2f} "
                        f"AND TP-Markov adverse_diff={_adv_diff:+.3f}\u2265{hurst_tp_veto_min_diff:.3f}. "
                        f"Strong-trend regime + signal disagreement; skipping {rec} {active_tier} trade."
                    )
                    audit.write(build_poll_record(
                        window_id=window_id, ticker=ticker, close_dt=close_dt,
                        decision="NO_TRADE", rec=rec,
                        reasons=[
                            f"hurst_tp_veto: hurst={_h:.2f}>={hurst_tp_veto_min_hurst:.2f} "
                            f"AND adverse_diff={_adv_diff:+.3f}>={hurst_tp_veto_min_diff:.3f} "
                            f"(tier={active_tier})"
                        ],
                        signal=sig, market=mkt, recent_5m_pct=recent_5m,
                        tp_cached=window_tp_cache.get(window_id),
                        db_check=db_check, ev_calc=None,
                        params=audit_params,
                        okx_check=okx_snap_pre if 'okx_snap_pre' in locals() else None,
                    ))
                    session.traded.add(window_id)
                    await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
                    continue

        # â”€â”€ 2026-06-01: DIRECTIONAL MAX-ADVERSE-BAR VETO â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Block trade direction X if any of the recent 5m bars moved
        # >= threshold AGAINST direction X. Catches the "catching a falling
        # knife" / "chasing a dead-cat bounce" pattern where Markov gives
        # too much weight to the most recent (small) bar while ignoring a
        # much larger preceding adverse spike.
        #
        # Directional: a -0.40% bar blocks YES trades (BTC dropped, don't
        # chase a bounce) but does NOT block NO trades (NO is going WITH
        # the recent crash).
        #
        # 7-day backtest: at threshold 0.30%, catches 2 disasters (-$226),
        # blocks 1 small win (+$1.28), net +$225/week.
        if (max_adverse_bar_veto_enabled and active_tier in max_adverse_bar_veto_tiers):
            _rec5m_for_veto = sig.get('recent_5m_pct', []) or []
            if _rec5m_for_veto:
                if rec == 'YES':
                    # Adverse bars are negative (BTC dropped)
                    _adverse_bars = [b for b in _rec5m_for_veto if b < 0]
                    _max_adv_mag = (abs(min(_adverse_bars)) if _adverse_bars else 0.0)
                else:
                    # NO trade: adverse bars are positive (BTC rose)
                    _adverse_bars = [b for b in _rec5m_for_veto if b > 0]
                    _max_adv_mag = (max(_adverse_bars) if _adverse_bars else 0.0)
                if _max_adv_mag >= max_adverse_bar_veto_pct:
                    log.warning(
                        f"MAX-ADVERSE-BAR VETO â€” recent 5m bar moved "
                        f"{_max_adv_mag:.3f}% against {rec} "
                        f"(\u2265{max_adverse_bar_veto_pct:.3f}% threshold). "
                        f"Skipping {rec} {active_tier} trade to avoid "
                        f"catching a falling knife."
                    )
                    audit.write(build_poll_record(
                        window_id=window_id, ticker=ticker, close_dt=close_dt,
                        decision="NO_TRADE", rec=rec,
                        reasons=[
                            f"max_adverse_bar_veto: max_adv_mag={_max_adv_mag:.3f}% "
                            f">= {max_adverse_bar_veto_pct:.3f}% (tier={active_tier})"
                        ],
                        signal=sig, market=mkt, recent_5m_pct=recent_5m,
                        tp_cached=window_tp_cache.get(window_id),
                        db_check=db_check, ev_calc=None,
                        params=audit_params,
                        okx_check=okx_snap_pre if 'okx_snap_pre' in locals() else None,
                    ))
                    session.traded.add(window_id)
                    await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
                    continue

        # â”€â”€ CUMULATIVE-ADVERSE-MOMENTUM VETO â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # 2026-06-02: catches the "slow steady drift against the trade"
        # pattern where no single 5m bar exceeds the per-bar veto threshold
        # (e.g. 0.30%) but the CUMULATIVE 30-min drift in the adverse
        # direction is large enough to threaten the position at settlement.
        #
        # 2026-06-02 05:09 NO trade (-$301): no single bar above 0.176%,
        # but cumulative +0.359% adverse drift over 30 min. The orderbook
        # never repriced (bid stayed 84-96Â¢) so the SL never fired; BTC
        # bounced through strike right at the settle window.
        #
        # 7-day backtest: threshold 0.35% blocks 5 trades (4W/1L), saves
        # $301 in losses, $28 in forgone wins, net +$273.
        if (cum_adverse_momentum_veto_enabled
                and active_tier in cum_adverse_momentum_veto_tiers):
            _rec5m_for_cum = sig.get('recent_5m_pct', []) or []
            if _rec5m_for_cum:
                _sum5m = sum(_rec5m_for_cum)
                if rec == 'YES':
                    # Adverse to YES = BTC down = negative sum
                    _adv_cum = -_sum5m
                else:
                    # Adverse to NO = BTC up = positive sum
                    _adv_cum = _sum5m
                if _adv_cum >= cum_adverse_momentum_veto_pct:
                    log.warning(
                        f"CUM-ADVERSE-MOMENTUM VETO â€” recent 5m sum drifted "
                        f"{_adv_cum:+.3f}% against {rec} over ~30 min "
                        f"(â‰¥{cum_adverse_momentum_veto_pct:.3f}% threshold). "
                        f"Skipping {rec} {active_tier} trade to avoid "
                        f"a slow grind reversal at settlement."
                    )
                    audit.write(build_poll_record(
                        window_id=window_id, ticker=ticker, close_dt=close_dt,
                        decision="NO_TRADE", rec=rec,
                        reasons=[
                            f"cum_adverse_momentum_veto: adv_cum={_adv_cum:+.3f}% "
                            f">= {cum_adverse_momentum_veto_pct:.3f}% (tier={active_tier})"
                        ],
                        signal=sig, market=mkt, recent_5m_pct=recent_5m,
                        tp_cached=window_tp_cache.get(window_id),
                        db_check=db_check, ev_calc=None,
                        params=audit_params,
                        okx_check=okx_snap_pre if 'okx_snap_pre' in locals() else None,
                    ))
                    session.traded.add(window_id)
                    await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
                    continue

        # â”€â”€ Cross-venue CONSENSUS VETO (binance_futures + OKX) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Catches the failure mode where the primary venue's per-source check passes
        # because its move is below veto_threshold_bps (default 5bps), but
        # BOTH external feeds are independently showing a small directional
        # move AGAINST our trade. The 2026-05-27 overnight session (14
        # settled trades, 92.9% WR overall) had ONE loss with this exact
        # signature:
        #
        #   YES @ 90Â¢ (HIGH-CONV)  Markov=0.95  TP=0.88
        #   binance_futures -0.01% (under 5bps threshold)
        #   OKX       -0.025% (record-only at the time)
        #   â†’ BTC dropped through strike in 11 min, lost $21.40
        #
        # All 13 winners had either: (a) both externals aligned with the
        # trade direction, OR (b) only one source available. NO winner had
        # both sources opposing. Adding this consensus check would have
        # blocked exactly that 1 loss while not blocking any wins.
        if (consensus_veto_enabled and okx_feed is not None
                and db_check is not None and db_check.get("consistent")):
            try:
                okx_snap = okx_feed.get_recent_move(
                    lookback_s=consensus_okx_lookback_s
                )
            except Exception:
                okx_snap = None
            okx_move = (okx_snap or {}).get("move_pct")
            db_move  = db_check.get("move_pct")
            # Directional helpers: small |move| < threshold treated as
            # neutral (no fresh information). Both must be DIRECTIONAL
            # AND OPPOSE the trade for the consensus veto to fire.
            def _is_directional(m, threshold):
                return (m is not None and abs(m) >= threshold)
            def _opposes(m, side_is_yes):
                if m is None: return False
                # Trade side YES = we want BTC up = positive move agrees
                if side_is_yes:
                    return m < 0
                return m > 0
            side_yes = (rec == "YES")
            db_dir = _is_directional(db_move, consensus_min_move_pct)
            ok_dir = _is_directional(okx_move, consensus_min_move_pct)
            db_opp = db_dir and _opposes(db_move,  side_yes)
            ok_opp = ok_dir and _opposes(okx_move, side_yes)
            if db_opp and ok_opp:
                # â”€â”€ 2026-06-01: Smart bypass checks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # 24h audit (17 vetoes): 65% had recent 5m bars strongly
                # favoring the trade direction. 6s blips were noise inside
                # the broader trend. Three independent bypasses below.
                bypass_active = False
                bypass_reason = None
                bypass_kind   = None
                if consensus_smart_bypass:
                    # Bypass 1: longer-window futures-lead contradicts the 6s blip
                    db_long_move = None
                    if futures_lead is not None:
                        try:
                            db_long_snap = futures_lead.get_recent_move(
                                lookback_s=consensus_long_window_s
                            )
                            db_long_move = (db_long_snap or {}).get("move_pct")
                        except Exception:
                            db_long_move = None
                    if db_long_move is not None:
                        # Favoring trade: YES â†’ positive move, NO â†’ negative move
                        if side_yes:
                            long_favors = db_long_move > 0
                        else:
                            long_favors = db_long_move < 0
                        if long_favors and abs(db_long_move) >= consensus_long_min_pct:
                            bypass_active = True
                            bypass_kind   = "longer_window"
                            bypass_reason = (f"longer-window futures "
                                             f"{db_long_move:+.4f}% over "
                                             f"{consensus_long_window_s:.0f}s favors {rec} "
                                             f"(6s blip is noise inside broader trend)")
                    # Bypass 2: recent 5m bars strongly favor trade
                    if not bypass_active and recent_5m:
                        sum_5m = sum(recent_5m)
                        if side_yes:
                            favors_5m = sum_5m >= consensus_5m_favor_pct
                        else:
                            favors_5m = sum_5m <= -consensus_5m_favor_pct
                        if favors_5m:
                            bypass_active = True
                            bypass_kind   = "5m_trend"
                            bypass_reason = (f"recent 5m bars sum={sum_5m:+.3f}% "
                                             f"strongly favors {rec} (30-min trend "
                                             f"overrides 6s blip)")
                    # Bypass 3: BTC far from strike AND consensus moves are tiny
                    if not bypass_active:
                        _dist_pct = mkt.get('dist_pct') if mkt else None
                        if _dist_pct is not None:
                            max_move = max(abs(db_move or 0), abs(okx_move or 0))
                            if (abs(_dist_pct) >= consensus_far_dist_pct
                                    and max_move < consensus_far_max_move_pct):
                                bypass_active = True
                                bypass_kind   = "far_from_strike"
                                bypass_reason = (f"|dist|={abs(_dist_pct):.3f}% â‰¥ "
                                                 f"{consensus_far_dist_pct:.3f}% AND "
                                                 f"max consensus move {max_move:.4f}% < "
                                                 f"{consensus_far_max_move_pct:.3f}% "
                                                 f"(too far for tiny wiggle to matter)")
                if bypass_active:
                    log.info(
                        f"CONSENSUS VETO BYPASSED ({bypass_kind}) â€” {bypass_reason}. "
                        f"Original veto: db={db_move:+.4f}%, okx={okx_move:+.4f}% "
                        f"both oppose {rec} (â‰¥{consensus_min_move_pct:.3f}%)."
                    )
                    audit.write(build_poll_record(
                        window_id=window_id, ticker=ticker, close_dt=close_dt,
                        decision="VETO_BYPASS", rec=rec,
                        reasons=[
                            f"consensus_veto_bypassed[{bypass_kind}]: {bypass_reason}"
                        ],
                        signal=sig, market=mkt, recent_5m_pct=recent_5m,
                        tp_cached=window_tp_cache.get(window_id),
                        db_check=db_check, ev_calc=None,
                        params=audit_params,
                        okx_check=okx_snap,
                        orderbook=ob_snap,
                        trade_flow=tr_snap,
                    ))
                    # Fall through to continue normal trade evaluation
                    # (do not set traded, do not continue/sleep â€” let the
                    # outer flow proceed to EV gate / tier / fire path).
                else:
                    log.warning(
                        f"CONSENSUS VETO â€” both externals oppose {rec} trade. "
                        f"Futures move={db_move:+.4f}%, OKX move={okx_move:+.4f}% "
                        f"(both |move| â‰¥ {consensus_min_move_pct:.3f}% directional threshold). "
                        f"Sub-veto-bps but consistent cross-source bearishness."
                    )
                    audit.write(build_poll_record(
                        window_id=window_id, ticker=ticker, close_dt=close_dt,
                        decision="NO_TRADE", rec=rec,
                        reasons=[
                            f"consensus_veto: db={db_move:+.4f}% okx={okx_move:+.4f}% "
                            f"both oppose {rec} (threshold {consensus_min_move_pct:.3f}%)"
                        ],
                        signal=sig, market=mkt, recent_5m_pct=recent_5m,
                        tp_cached=window_tp_cache.get(window_id),
                        db_check=db_check, ev_calc=None,
                        params=audit_params,
                        okx_check=okx_snap,
                        orderbook=ob_snap,
                        trade_flow=tr_snap,
                    ))
                    # 2026-05-29 BUG FIX: consensus veto is TRANSIENT, not terminal.
                    # Externals can flip in seconds; the model said "yes trade",
                    # only externals said "wait". Previously, veto added the window
                    # to `session.traded` and slept until close â€” abandoning a
                    # strong-signal window after a single 6-second snapshot of
                    # external opposition. Now treat veto like other transient
                    # blockers: sleep briefly, re-evaluate on the next poll.
                    # If mins<3.5 the window's near-close and we wrap it up.
                    if mins_left > 3.5:
                        await asyncio.sleep(10)
                    else:
                        session.traded.add(window_id)
                        await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
                    continue
                # else: bypass_active=True â†’ fall through to normal trade flow
            elif db_opp or ok_opp:
                # Single-source opposing â€” log but don't veto (historical data
                # shows wins in this category).
                log.info(
                    f"   Consensus check: one source opposes "
                    f"(db={db_move}, okx={okx_move}) â€” single-source opposing "
                    f"not a veto trigger, proceeding."
                )

        # â”€â”€ 2026-06-09: ROLLING-WR DEFENSIVE MODE GATE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # When recent rolling win rate has dropped below threshold, block
        # weaker tiers (default: standard, strong). HC / late-sure / late-dir
        # still fire since they require extreme signals. This adapts to
        # regime shifts that show up via observed outcomes (clustered losses)
        # rather than via per-trade signal features.
        _eff_tier = locals().get("active_tier") or "standard"
        if session.is_tier_blocked_defensive(_eff_tier):
            log.warning(
                f"[REGIME] DEFENSIVE MODE \u2014 skipping {rec} {_eff_tier} trade. "
                f"Rolling WR: {session._rolling_win_rate()*100:.1f}% "
                f"({sum(session.recent_outcomes)}/{len(session.recent_outcomes)}). "
                f"Tier '{_eff_tier}' is blocked when defensive."
            )
            audit.write(build_poll_record(
                window_id=window_id, ticker=ticker, close_dt=close_dt,
                decision="NO_TRADE", rec=rec,
                reasons=[f"defensive_mode: rolling_WR={session._rolling_win_rate()*100:.1f}% "
                         f"\u2264 {session.rolling_wr_threshold*100:.0f}%, "
                         f"tier={_eff_tier} is blocked"],
                signal=sig, market=mkt, recent_5m_pct=recent_5m,
                tp_cached=window_tp_cache.get(window_id),
                db_check=None, ev_calc=None, params=audit_params,
                orderbook=ob_snap, trade_flow=tr_snap,
            ))
            session.traded.add(window_id)
            await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
            continue

        # â”€â”€ 2026-06-15: MIN-ENTRY-SIZE GATE (standard tier) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # A marginal-EV standard entry can be Kelly-sized down to a token 1-2
        # contracts. Filling it opens the position and marks the window traded,
        # which LOCKS OUT any larger/better re-entry (incl. a high-conv setup)
        # for the rest of the window (see session.traded skip). So if the
        # standard size is below the threshold, skip WITHOUT marking traded and
        # keep re-evaluating: the window stays open for a better-priced standard
        # entry or a high-conv qualification later. Standard only â€” the sure
        # tiers already floor their size up via *_min_stake_pct.
        _ms_tier = locals().get("active_tier") or "standard"
        if (_ms_tier == "standard"
                and standard_min_entry_contracts > 1
                and contracts < standard_min_entry_contracts):
            log.info(
                f"[MIN-SIZE] standard {rec} {contracts}c < min "
                f"{standard_min_entry_contracts}c @ {limit_price}\u00a2 â€” skipping; "
                f"keeping window open for a better/high-conv entry")
            audit.write(build_poll_record(
                window_id=window_id, ticker=ticker, close_dt=close_dt,
                decision="NO_TRADE", rec=rec,
                reasons=[f"min_entry_size: standard {contracts}c < "
                         f"{standard_min_entry_contracts}c"],
                signal=sig, market=mkt, recent_5m_pct=recent_5m,
                tp_cached=window_tp_cache.get(window_id),
                db_check=None, ev_calc=None, params=audit_params,
                orderbook=ob_snap, trade_flow=tr_snap,
            ))
            if mins_left > 3.5:
                await asyncio.sleep(10)   # re-poll like a transient NO_TRADE
            else:
                session.traded.add(window_id)
                await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
            continue

        # â”€â”€ TRADE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # â”€â”€ 2026-06-17: MAX PER-TRADE CAPITAL CAP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Hard ceiling on a single ticket's capital (contracts x entry). Bounds
        # single-ticket dollar risk regardless of tier/Kelly â€” clips only the
        # oversized monster tickets (the âˆ’$835/âˆ’$1070 near-strike reversals)
        # while leaving the small/medium book full size. 10-day study: capping
        # at ~$500 flipped the all-tier +10c book from breakeven to +$285,
        # because the few catastrophic tickets are bounded without cutting the
        # profitable bulk. Applied at the final contracts value so it covers
        # both the base sizing and all EV-gate tiers; the walk-up/top-up target
        # `contracts`, so the cap propagates. Off when max_trade_usd <= 0.
        if max_trade_usd > 0 and limit_price > 0:
            _cap_ct = int(max_trade_usd * 100.0 / limit_price)
            if contracts > _cap_ct:
                log.info(
                    f"[MAX-TRADE-USD] capping {rec} {contracts}c -> {_cap_ct}c "
                    f"@ {limit_price}\u00a2 (ticket ${contracts*limit_price/100.0:.0f} "
                    f"> ${max_trade_usd:.0f} cap)")
                contracts = max(0, _cap_ct)
                if contracts <= 0:
                    audit.write(build_poll_record(
                        window_id=window_id, ticker=ticker, close_dt=close_dt,
                        decision="NO_TRADE", rec=rec,
                        reasons=[f"max_trade_usd: cap ${max_trade_usd:.0f} < 1 "
                                 f"contract at {limit_price}\u00a2"],
                        signal=sig, market=mkt, recent_5m_pct=recent_5m,
                        tp_cached=window_tp_cache.get(window_id),
                        db_check=None, ev_calc=None, params=audit_params,
                        orderbook=ob_snap, trade_flow=tr_snap,
                    ))
                    if mins_left > 3.5:
                        await asyncio.sleep(10)
                    else:
                        session.traded.add(window_id)
                        await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
                    continue

        p_d      = limit_price / 100
        fee_c    = MAKER_FEE_RATE * p_d * (1 - p_d)
        cost_per = p_d + fee_c
        net_win  = (1 - p_d) - fee_c

        log.info(
            f"{'[DRY RUN] ' if dry_run else ''}TRADE  BUY {rec} {contracts}c @ {limit_price}Â¢  "
            f"max_loss=${signal['max_loss_usd']:.2f}  EV=${signal['expected_value']:+.2f}  "
            f"ticker={ticker}"
        )
        # 2026-06-12: remember the largest size we WANTED in this window
        # (patient top-up tries to complete it at EV-valid prices later).
        window_max_intent[window_id] = max(window_max_intent.get(window_id, 0),
                                           contracts)

        # â”€â”€ LOG-ONLY new-signal snapshot (ChainVector live signals) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Observational: records candidate veto signals at fire time so we can
        # validate them against outcomes before enforcing anything. Does NOT
        # gate, veto, or modify the trade in any way.
        _ns = None
        try:
            _ns = snapshot_new_signals(rec)
            log.info(f"   [NEW-SIGNALS log-only] {_ns['summary']}")
            _cgc = _build_cv_composite(rec, mkt, mins_left, _ns, recent_5m)
            audit.write({
                "type": "NEW_SIGNALS",
                "window_id": window_id,
                "ticker": ticker,
                "side": rec,
                "limit_price_cents": limit_price,
                **{k: _ns[k] for k in (
                    "perp_m6s", "perp_m30s", "perp_m60s", "perp_m5m",
                    "perp_imb", "perp_spread",
                    "liq_adverse_5m_k", "liq_support_5m_k",
                    "taker_1m", "oi_d5m_pct",
                    "funding", "whale", "ls_ratio", "book_skew",
                    "perp_feed_ok", "cv_feed_ok",
                    "perp_veto", "liq_veto",
                    "liq_heatmap", "cv_momentum", "cascade",
                )},
                "cb_mom_30s": coinbase_momentum(30.0),
                "cb_mom_60s": coinbase_momentum(60.0),
                "cv_composite": _cgc,
                # Recorded ChainVector context: model-vs-market edge for THIS
                # ticker plus regime/volatility/risk-index. Log-only for now â€”
                # promotable to gates after outcome validation.
                "cv_edge": _cv_client.edge(ticker) if _cv_client.enabled else None,
                "cv_context": context_signals(mins_left),
                "tier": locals().get("active_tier"),
                "gap": signal.get("gap"),
                "persist": signal.get("persist"),
                "dist_pct": signal.get("dist_pct"),
                "p_yes": signal.get("p_yes"),
                "combined_p": signal.get("combined_p"),
            })
        except Exception as _e:
            log.debug(f"   [NEW-SIGNALS] snapshot failed (non-fatal): {_e!r}")

        # 2026-06-29: TAKER-FLOW ENTRY VETO (near-strike reversal trap).
        # Block when recent Polymarket taker aggression is strongly AGAINST the
        # bet AND price is close to strike. Backtest (20 audit files):
        # agg_against>=0.90 & |dist|<0.25 blocked 33 (10 losers/23 tiny
        # winners) -> saved ~$1829 vs $371 cut = +$1458. Catches the 00:48
        # -$291 reversal (99.6% YES taker aggression) and the -$1029 blow-up.
        # Fail-open when trade-flow is thin/unavailable.
        _tf_agg_against = None
        if (taker_flow_veto_enabled and tr_snap is not None
                and tr_snap.get("n_trades", 0) >= taker_flow_veto_min_trades
                and tr_snap.get("yes_aggression") is not None):
            _ya_frac = tr_snap["yes_aggression"]
            _tf_agg_against = _ya_frac if rec == "NO" else (1.0 - _ya_frac)
        _tf_dist = mkt.get("dist_pct")
        if (taker_flow_veto_enabled and _tf_agg_against is not None
                and _tf_dist is not None
                and _tf_agg_against >= taker_flow_veto_agg_min
                and abs(_tf_dist) < taker_flow_veto_dist_max):
            log.warning(
                f"TAKER-FLOW VETO - {_tf_agg_against*100:.0f}% of recent taker "
                f"volume (n={tr_snap.get('n_trades')}) aggressing AGAINST "
                f"{rec} while price only {abs(_tf_dist):.3f}% from strike "
                f"(< {taker_flow_veto_dist_max:.2f}); near-strike reversal trap; skipping."
            )
            audit.write(build_poll_record(
                window_id=window_id, ticker=ticker, close_dt=close_dt,
                decision="NO_TRADE", rec=rec,
                reasons=[f"taker_flow_veto: agg_against={_tf_agg_against:.3f} >= "
                         f"{taker_flow_veto_agg_min:.2f} & |dist|={abs(_tf_dist):.3f} "
                         f"< {taker_flow_veto_dist_max:.2f}"],
                signal=sig, market=mkt, recent_5m_pct=recent_5m,
                tp_cached=window_tp_cache.get(window_id),
                db_check=None, ev_calc=None, params=audit_params,
                orderbook=ob_snap, trade_flow=tr_snap,
            ))
            if mins_left > 3.5:
                await asyncio.sleep(10)
            else:
                session.traded.add(window_id)
                await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
            continue

        # â”€â”€ 2026-07-02: POLYMARKET BID-STABILITY ENTRY VETO â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Our side's executable bid (100 - opposite ask) must be stable or
        # rising over the recent lookback â€” a falling / off-peak bid means the
        # market is actively repricing AGAINST this position (reversal risk).
        # BTC 30d backtest (net>=0 & fade<=2c over 60s): vetoed 117 entries,
        # saved $3,626 in losses vs $2,493 winners clipped (+$1,133 net,
        # positive in BOTH halves). Fail-open below min samples. Transient:
        # re-polls while time remains, so a dip that recovers can re-enter.
        if bid_stab_veto_enabled:
            _bsv = _bidstab_check(window_id, rec, bid_stab_lookback_s,
                                  bid_stab_max_fade_cents, bid_stab_min_samples)
            # 2026-07-02 v2: history clean -> confirm REAL-TIME stability over
            # the last few seconds (any downtick = wobble = wait).
            if _bsv is None and bid_stab_burst_samples > 0:
                _bsv = await _bidstab_burst(rec, bid_stab_burst_samples,
                                            bid_stab_burst_interval_s,
                                            window_id)
            if _bsv:
                log.warning(f"BID-STABILITY VETO â€” {_bsv}; skipping entry.")
                audit.write(build_poll_record(
                    window_id=window_id, ticker=ticker, close_dt=close_dt,
                    decision="NO_TRADE", rec=rec,
                    reasons=[f"bid_stab_veto: {_bsv}"],
                    signal=sig, market=mkt, recent_5m_pct=recent_5m,
                    tp_cached=window_tp_cache.get(window_id),
                    db_check=None, ev_calc=None, params=audit_params,
                    orderbook=ob_snap, trade_flow=tr_snap,
                ))
                # 2026-07-02 v3: stability-watch fast re-poll. A stability
                # veto is a WAIT, not a rejection: recheck every ~3s so we
                # enter the moment the bid holds (waiting a full poll cycle
                # let the ask run away while the move confirmed). Keep
                # watching down to 2.5 min left instead of abandoning the
                # window at 3.5 min; timing/EV/price gates still apply on
                # every recheck with fresh quotes.
                if mins_left > 2.5:
                    await asyncio.sleep(3)
                else:
                    session.traded.add(window_id)
                    await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
                continue

        # â”€â”€ ChainVector PREDICTION QUOTE-STABILITY VETO â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Cross-checks the local bid-stability read with ChainVector's
        # /predictions/stability for THIS ticker: momentum_score is the signed
        # YES-mid repricing speed (âˆ’100..+100, positive = YES getting more
        # expensive). If the market is repricing hard AGAINST our side, the
        # crowd is exiting our direction at this exact moment â€” wait it out.
        # Complements the local check (which only sees our own top-of-book
        # samples) with tick-derived quote history. Fail-open when no data;
        # transient fast re-poll like the bid-stab veto.
        if cv_stab_veto_enabled and _cv_client.enabled:
            _stab = None
            try:
                _stab = _cv_client.stability(ticker)
            except Exception:
                _stab = None
            _w = None
            if isinstance(_stab, dict):
                _w = _stab.get("1m") or _stab.get("30s") or _stab.get("5m")
            _stab_mom = None if not isinstance(_w, dict) else _w.get("momentum_score")
            if _stab_mom is not None:
                # momentum_score is YES-signed; flip for NO positions.
                _stab_signed = float(_stab_mom) if rec == "YES" else -float(_stab_mom)
                if _stab_signed <= -cv_stab_mom_against:
                    log.warning(
                        f"CV-STABILITY VETO â€” Polymarket quote repricing against {rec}: "
                        f"momentum_score={float(_stab_mom):+.1f} (signed "
                        f"{_stab_signed:+.1f} \u2264 -{cv_stab_mom_against:.0f}), "
                        f"stability={_w.get('stability_score')}. Waiting for the "
                        f"quote to settle."
                    )
                    audit.write(build_poll_record(
                        window_id=window_id, ticker=ticker, close_dt=close_dt,
                        decision="NO_TRADE", rec=rec,
                        reasons=[f"cv_stability_veto: mom={float(_stab_mom):+.1f} "
                                 f"signed={_stab_signed:+.1f}"],
                        signal=sig, market=mkt, recent_5m_pct=recent_5m,
                        tp_cached=window_tp_cache.get(window_id),
                        db_check=None, ev_calc=None, params=audit_params,
                        orderbook=ob_snap, trade_flow=tr_snap,
                        extra={"cv_stability": _stab},
                    ))
                    # Same wait-not-reject cadence as the bid-stab veto.
                    if mins_left > 2.5:
                        await asyncio.sleep(3)
                    else:
                        session.traded.add(window_id)
                        await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
                    continue

        # â”€â”€ ChainVector LIQUIDATION CASCADE-RISK VETO â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # /liquidations/cascade-risk scores the odds that current price action
        # ignites a chain of forced liquidations (0-100) and names the side
        # that would be liquidated. A high score with the cascade side AGAINST
        # us (longs being liquidated while we hold YES = forced selling into
        # our position) is a fat-tail setup the Markov stack can't see.
        # Fail-open when the feed is down; transient re-poll.
        if cv_cascade_veto_enabled and _ns is not None:
            _casc = _ns.get("cascade")
            if isinstance(_casc, dict) and _casc.get("risk_score") is not None:
                try:
                    _crisk = float(_casc["risk_score"])
                except (TypeError, ValueError):
                    _crisk = None
                _cside = str(_casc.get("cascade_side") or "").lower()
                _against = ((rec == "YES" and _cside.startswith("long"))
                            or (rec == "NO" and _cside.startswith("short")))
                if _crisk is not None and _crisk >= cv_cascade_veto_score and _against:
                    log.warning(
                        f"CV-CASCADE VETO â€” liquidation cascade risk {_crisk:.0f} "
                        f"(\u2265{cv_cascade_veto_score:.0f}) with {_cside} side at "
                        f"risk = forced flow AGAINST {rec}. Skipping."
                    )
                    audit.write(build_poll_record(
                        window_id=window_id, ticker=ticker, close_dt=close_dt,
                        decision="NO_TRADE", rec=rec,
                        reasons=[f"cv_cascade_veto: risk={_crisk:.0f} "
                                 f"side={_cside} against {rec}"],
                        signal=sig, market=mkt, recent_5m_pct=recent_5m,
                        tp_cached=window_tp_cache.get(window_id),
                        db_check=None, ev_calc=None, params=audit_params,
                        orderbook=ob_snap, trade_flow=tr_snap,
                        extra={"cv_cascade": _casc},
                    ))
                    if mins_left > 3.5:
                        await asyncio.sleep(10)
                    else:
                        session.traded.add(window_id)
                        await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
                    continue

        # â”€â”€ 2026-06-12: PERP-MOMENTUM ENTRY VETO (ENFORCED) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Skip the trade when the Polymarket perp's 30s tape is actively moving
        # against the position at fire time. Validated on 14 logged fires:
        # blocked winners were tiny (~$60 total) while blocked losses included
        # a single âˆ’$446 catastrophe (2026-06-12 13:56). Fail-open: if the
        # perp feed is down (m30s is None), no veto. Transient like the
        # consensus veto: re-polls while time remains rather than abandoning.
        if (perp_veto_enabled and _ns is not None
                and _ns.get("perp_m30s") is not None
                and _ns["perp_m30s"] <= perp_veto_m30s_threshold):
            log.warning(
                f"PERP VETO â€” perp 30s momentum {_ns['perp_m30s']:+.2f}bp "
                f"(signed toward {rec}) \u2264 {perp_veto_m30s_threshold:+.1f}bp "
                f"threshold. Venue perp tape is moving against this entry; "
                f"skipping. ({_ns['summary']})"
            )
            audit.write(build_poll_record(
                window_id=window_id, ticker=ticker, close_dt=close_dt,
                decision="NO_TRADE", rec=rec,
                reasons=[f"perp_veto: m30s={_ns['perp_m30s']:+.2f}bp <= "
                         f"{perp_veto_m30s_threshold:+.1f}bp"],
                signal=sig, market=mkt, recent_5m_pct=recent_5m,
                tp_cached=window_tp_cache.get(window_id),
                db_check=None, ev_calc=None, params=audit_params,
                orderbook=ob_snap, trade_flow=tr_snap,
            ))
            # Transient veto: the 30s blip may pass â€” re-poll if time permits.
            if mins_left > 3.5:
                await asyncio.sleep(10)
            else:
                session.traded.add(window_id)
                await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
            continue

        # â”€â”€ 2026-06-14: BOOK-SKEW ENTRY VETO (golden-zone only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Binance futures resting-depth imbalance (signed toward the trade
        # side). Validated golden-only: a -0.15 gate on 65-73c entries cut 4
        # losers (incl. the -$446 & -$417) vs 7 small winners, +$720 net.
        # Backtest showed it HURTS non-golden tiers (cuts 25W/9L), so it is
        # scoped to the golden price band by default. book_skew is the
        # ChainVector orderbook-imbalance read â€” fail-open if the feed is
        # down. Transient re-poll
        # like the perp veto (book imbalance can shift within the window).
        _is_golden_entry = (65 <= limit_price <= 73)
        _bs_applies = _is_golden_entry or not book_skew_golden_only
        if (book_skew_veto_enabled and _bs_applies and _ns is not None
                and _ns.get("book_skew") is not None
                and _ns["book_skew"] <= book_skew_threshold):
            log.warning(
                f"BOOK-SKEW VETO â€” Binance futures depth skew "
                f"{_ns['book_skew']:+.3f} (signed toward {rec}) \u2264 "
                f"{book_skew_threshold:+.2f} threshold on golden entry "
                f"{limit_price}\u00a2. Resting depth positioned against this "
                f"trade; skipping. ({_ns['summary']})"
            )
            audit.write(build_poll_record(
                window_id=window_id, ticker=ticker, close_dt=close_dt,
                decision="NO_TRADE", rec=rec,
                reasons=[f"book_skew_veto: skew={_ns['book_skew']:+.3f} <= "
                         f"{book_skew_threshold:+.2f} (golden {limit_price}\u00a2)"],
                signal=sig, market=mkt, recent_5m_pct=recent_5m,
                tp_cached=window_tp_cache.get(window_id),
                db_check=None, ev_calc=None, params=audit_params,
                orderbook=ob_snap, trade_flow=tr_snap,
            ))
            if mins_left > 3.5:
                await asyncio.sleep(10)
            else:
                session.traded.add(window_id)
                await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
            continue

        # â”€â”€ 2026-06-15: PERP-IMB DEEP VETO (all tiers) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Polymarket perp book depth imbalance (signed toward trade side). When the
        # perp book is STRONGLY stacked against the position (<= deep negative
        # threshold), it flags a reversal-prone subset (e.g. the âˆ’$248 NO at
        # perp_imb âˆ’0.73). Backtest: deep threshold (â‰¤âˆ’0.50) net-positive on
        # all tiers, cuts only SMALL winners (no big winners â‰¤âˆ’0.50). Catches a
        # MINORITY of big losses (the visibly book-stacked ones), not all.
        # Transient re-poll: delays and re-enters rather than hard-blocking.
        if (perp_imb_veto_enabled and _ns is not None
                and _ns.get("perp_imb") is not None
                and _ns["perp_imb"] <= perp_imb_veto_threshold):
            log.warning(
                f"PERP-IMB VETO â€” Polymarket perp book imbalance "
                f"{_ns['perp_imb']:+.3f} (signed toward {rec}) \u2264 "
                f"{perp_imb_veto_threshold:+.2f} threshold. Perp book strongly "
                f"stacked against this {rec} entry (reversal risk); skipping. "
                f"({_ns['summary']})"
            )
            audit.write(build_poll_record(
                window_id=window_id, ticker=ticker, close_dt=close_dt,
                decision="NO_TRADE", rec=rec,
                reasons=[f"perp_imb_veto: imb={_ns['perp_imb']:+.3f} <= "
                         f"{perp_imb_veto_threshold:+.2f}"],
                signal=sig, market=mkt, recent_5m_pct=recent_5m,
                tp_cached=window_tp_cache.get(window_id),
                db_check=None, ev_calc=None, params=audit_params,
                orderbook=ob_snap, trade_flow=tr_snap,
            ))
            if mins_left > 3.5:
                await asyncio.sleep(10)
            else:
                session.traded.add(window_id)
                await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
            continue

        # â”€â”€ 2026-06-17: GOLDEN NEAR + HIGH-VOL VETO â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Skip golden-band entries that are BOTH close to the strike AND in
        # elevated short-term vol. 10-day study: the near+high-vol bucket
        # (|dist%|<0.08, GK_vol>=~0.0020) lost ~36% vs ~20% for far entries â€”
        # these are the knife-edge whips that flip back through the strike late
        # (e.g. 2026-06-17 NO losers at dist 0.078-0.111). Distance alone is
        # confounded with vol (corr +0.49), so this gates on BOTH. Golden-only
        # (the band where this was validated). Fail-open if GK_vol/dist missing.
        # Transient re-poll like the other entry vetoes â€” keeps the window open
        # in case BTC moves clearly away from the strike before close.
        # 2026-06-22: minimum entry-cushion floor. Near-money entries are
        # net-losing on BTC 15m (recent regime: <0.20% cushion = -$5.5k;
        # >=0.20% = +$0.5k). Skip them; re-poll so the window stays open if
        # BTC moves clear of the strike before close.
        if MIN_ENTRY_CUSHION_PCT > 0:
            _mec_dist = mkt.get("dist_pct") if isinstance(mkt, dict) else None
            if _mec_dist is not None and abs(_mec_dist) < MIN_ENTRY_CUSHION_PCT:
                log.warning(
                    f"MIN-CUSHION VETO â€” {rec} @ {limit_price}Â¢ only "
                    f"|dist|={abs(_mec_dist):.3f}% from strike "
                    f"(< {MIN_ENTRY_CUSHION_PCT:.3f}%); near-money reversal "
                    f"risk â€” skipping."
                )
                audit.write(build_poll_record(
                    window_id=window_id, ticker=ticker, close_dt=close_dt,
                    decision="NO_TRADE", rec=rec,
                    reasons=[f"min_entry_cushion: |dist|={abs(_mec_dist):.3f}%"
                             f"<{MIN_ENTRY_CUSHION_PCT:.3f}%"],
                    signal=sig, market=mkt, recent_5m_pct=recent_5m,
                    tp_cached=window_tp_cache.get(window_id),
                    db_check=None, ev_calc=None, params=audit_params,
                    orderbook=ob_snap, trade_flow=tr_snap,
                ))
                if mins_left > 3.5:
                    await asyncio.sleep(10)
                else:
                    session.traded.add(window_id)
                    await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
                continue

        if golden_near_vol_veto_enabled:
            _gnv_gk = sig.get("gk_vol") if isinstance(sig, dict) else None
            _gnv_dist = mkt.get("dist_pct") if isinstance(mkt, dict) else None
            _gnv_golden = golden_price_lo <= limit_price <= golden_price_hi
            if (_gnv_golden and _gnv_gk is not None and _gnv_dist is not None
                    and abs(_gnv_dist) < golden_near_vol_dist_max
                    and _gnv_gk >= golden_near_vol_gk_min):
                log.warning(
                    f"GOLDEN NEAR+VOL VETO â€” {rec} @ {limit_price}\u00a2 is "
                    f"near-strike (|dist|={abs(_gnv_dist):.3f}% < "
                    f"{golden_near_vol_dist_max:.3f}%) AND high-vol "
                    f"(GK={_gnv_gk:.5f} \u2265 {golden_near_vol_gk_min:.5f}); "
                    f"knife-edge whip risk â€” skipping."
                )
                audit.write(build_poll_record(
                    window_id=window_id, ticker=ticker, close_dt=close_dt,
                    decision="NO_TRADE", rec=rec,
                    reasons=[f"golden_near_vol_veto: |dist|={abs(_gnv_dist):.3f}%"
                             f"<{golden_near_vol_dist_max:.3f}% & GK={_gnv_gk:.5f}"
                             f">={golden_near_vol_gk_min:.5f}"],
                    signal=sig, market=mkt, recent_5m_pct=recent_5m,
                    tp_cached=window_tp_cache.get(window_id),
                    db_check=None, ev_calc=None, params=audit_params,
                    orderbook=ob_snap, trade_flow=tr_snap,
                ))
                if mins_left > 3.5:
                    await asyncio.sleep(10)
                else:
                    session.traded.add(window_id)
                    await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
                continue

        # â”€â”€ 2026-06-22: HARD ENTRY-PRICE FLOOR (all tiers, final pre-order) â”€â”€
        if min_entry_price > 0 and limit_price < min_entry_price:
            log.warning(
                f"ENTRY-PRICE FLOOR â€” {rec} @ {limit_price}Â¢ < floor "
                f"{min_entry_price}Â¢; skipping (cheap-tier reversal risk). "
                f"Re-polling in case price rises into the floor."
            )
            audit.write(build_poll_record(
                window_id=window_id, ticker=ticker, close_dt=close_dt,
                decision="NO_TRADE", rec=rec,
                reasons=[f"min_entry_price: {limit_price}c < {min_entry_price}c"],
                signal=sig, market=mkt, recent_5m_pct=recent_5m,
                tp_cached=window_tp_cache.get(window_id),
                db_check=None, ev_calc=None, params=audit_params,
                orderbook=ob_snap, trade_flow=tr_snap,
            ))
            if mins_left > 3.5:
                await asyncio.sleep(10)
            else:
                session.traded.add(window_id)
                await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
            continue

        # Initialize hedge_state to a default so both dry_run and live paths
        # have a consistent variable to reference. Live path overrides it
        # inside the chunked-loop scope below.
        hedge_state = {"eligible": False, "skip_reason": "not initialized (dry-run)"}
        # 2026-05-31 bugfix: v2_price_yes_leg must be initialized BEFORE the
        # dry_run/else split because downstream audit code (build_trade_record)
        # uses it as a fallback when avg_fill_price is None (always None in
        # dry-run). Previously only set in the live `else:` branch, causing
        # UnboundLocalError under --dry-run.
        v2_price_yes_leg = (limit_price / 100) if rec == "YES" else (1 - limit_price / 100)

        if dry_run:
            order_id       = "dry-run"
            filled_count   = contracts
            avg_fill_price = None  # let downstream fall back to limit price
            log.info(f"[DRY RUN] Order simulated â€” no real order sent")
            # Dry-run hedge logging â€” show what hedge WOULD have done
            if hedge_enabled:
                pre_active_tier = locals().get("active_tier") or "standard"
                opposite_ask_at_init = (mkt.get("no_ask") if rec == "YES"
                                         else mkt.get("yes_ask"))
                elig, reason = hedge_eligible(
                    tier=pre_active_tier, side=rec, entry_cents=limit_price,
                    opposite_ask_cents=opposite_ask_at_init,
                    hedge_enabled=hedge_enabled, hedge_tiers=hedge_tiers,
                    hedge_min_yes_entry=hedge_min_yes_entry,
                    hedge_max_yes_entry=hedge_max_yes_entry,
                    hedge_max_no_cost=hedge_max_no_cost,
                )
                if elig and opposite_ask_at_init:
                    base_sl = (sl_loss_cents_high_conv
                               if pre_active_tier in ("high_conv", "late_sure")
                               else sl_loss_cents)
                    effective_sl = max(base_sl, hedge_widened_sl_cents)
                    M_dry = compute_hedge_target(
                        yes_contracts=filled_count,
                        yes_entry_cents=limit_price,
                        opposite_ask_cents=opposite_ask_at_init,
                        sl_loss_cents=effective_sl,
                        opposite_settle_assumed=hedge_no_settle_assumed,
                    )
                    log.info(
                        f"[DRY RUN] Safety HEDGE would BUY "
                        f"{'NO' if rec=='YES' else 'YES'} {M_dry}c "
                        f"@ ~{opposite_ask_at_init}\u00a2 (effective_sl={effective_sl}\u00a2)"
                    )
                else:
                    log.info(f"[DRY RUN] Safety HEDGE skipped: {reason}")
            # â”€â”€ 2026-06-01: Dry-run Smart FLIP simulation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Even though no real SL can fire in dry-run, simulate the flip
            # decision "what if SL fires here at the expected trigger price?"
            if smart_flip_enabled:
                pre_active_tier = locals().get("pre_active_tier") or locals().get("active_tier") or "standard"
                base_sl_for_sim = (sl_loss_cents_high_conv
                                    if pre_active_tier in ("high_conv", "late_sure")
                                    else sl_loss_cents)
                est_sl_exit_cents = max(1, limit_price - base_sl_for_sim - 3)
                est_opp_bid = min(99, 100 - est_sl_exit_cents + 2)
                est_opp_ask = min(99, est_opp_bid + 1)
                est_primary_loss = (limit_price - est_sl_exit_cents) * filled_count / 100.0

                flip_elig_sim, flip_reason_sim = smart_flip_eligibility(
                    tier=pre_active_tier, primary_side=rec,
                    mins_remaining=sig.get('minutes_left', 15),
                    opp_bid_cents=est_opp_bid,
                    hedge_active=hedge_enabled,
                    enabled=smart_flip_enabled,
                    eligible_tiers=smart_flip_tiers,
                    min_opp=smart_flip_min_opp_entry,
                    max_opp=smart_flip_max_opp_entry,
                    min_mins_remaining=smart_flip_min_mins_remaining,
                )
                if not flip_elig_sim:
                    log.info(f"[DRY RUN] Smart FLIP would skip: {flip_reason_sim} "
                             f"(simulated SL at {est_sl_exit_cents}\u00a2, opp_bid~{est_opp_bid}\u00a2)")
                    audit.write({
                        "type":                "FLIP_SKIP_SIM",
                        "window_id":           window_id,
                        "ticker":              ticker,
                        "primary_side":        rec,
                        "primary_entry":       limit_price,
                        "primary_tier":        pre_active_tier,
                        "est_sl_exit":         est_sl_exit_cents,
                        "est_opp_bid":         est_opp_bid,
                        "est_primary_loss":    round(est_primary_loss, 2),
                        "skip_reason":         flip_reason_sim,
                        "simulated":           True,
                        "ts":                  datetime.now(timezone.utc).isoformat(),
                        "ts_ms":               int(time.time() * 1000),
                    })
                else:
                    M_flip_sim, M_reason_sim = compute_smart_flip_size(
                        primary_loss_usd=est_primary_loss,
                        opp_entry_cents=est_opp_ask,
                        sell_target_cents=smart_flip_sell_target,
                        recovery_ratio=smart_flip_recovery_ratio,
                        max_capital_usd=smart_flip_max_capital_usd,
                    )
                    flip_opp_side_sim = "NO" if rec == "YES" else "YES"
                    if M_flip_sim <= 0:
                        log.info(f"[DRY RUN] Smart FLIP would skip: {M_reason_sim} "
                                 f"(simulated SL at {est_sl_exit_cents}\u00a2)")
                        audit.write({
                            "type":                "FLIP_SKIP_SIM",
                            "window_id":           window_id,
                            "ticker":              ticker,
                            "primary_side":        rec,
                            "primary_entry":       limit_price,
                            "primary_tier":        pre_active_tier,
                            "est_sl_exit":         est_sl_exit_cents,
                            "est_opp_bid":         est_opp_bid,
                            "est_primary_loss":    round(est_primary_loss, 2),
                            "skip_reason":         M_reason_sim,
                            "simulated":           True,
                            "ts":                  datetime.now(timezone.utc).isoformat(),
                            "ts_ms":               int(time.time() * 1000),
                        })
                    else:
                        flip_capital_sim = M_flip_sim * est_opp_ask / 100.0
                        target_recovery = est_primary_loss * smart_flip_recovery_ratio
                        best_case_gain = M_flip_sim * (smart_flip_sell_target - 1 - est_opp_ask) / 100.0
                        worst_case_loss = -(M_flip_sim * (smart_flip_sl_cents + 5) / 100.0)
                        log.info(
                            f"[DRY RUN] Smart FLIP would FIRE: BUY {flip_opp_side_sim} "
                            f"{M_flip_sim}c @ ~{est_opp_ask}\u00a2 "
                            f"(if primary SL'd at {est_sl_exit_cents}\u00a2, est_loss=${est_primary_loss:.2f})  "
                            f"target_recovery=${target_recovery:.2f}  capital=${flip_capital_sim:.2f}  "
                            f"best_gain=${best_case_gain:+.2f}  worst_loss=${worst_case_loss:+.2f}"
                        )
                        audit.write({
                            "type":                "FLIP_ATTACH_SIM",
                            "window_id":           window_id,
                            "ticker":              ticker,
                            "primary_side":        rec,
                            "primary_entry":       limit_price,
                            "primary_tier":        pre_active_tier,
                            "est_sl_exit":         est_sl_exit_cents,
                            "est_primary_loss":    round(est_primary_loss, 2),
                            "flip_side":           flip_opp_side_sim,
                            "flip_target":         M_flip_sim,
                            "flip_filled":         M_flip_sim,
                            "flip_entry_cents":    est_opp_ask,
                            "flip_capital_usd":    round(flip_capital_sim, 2),
                            "flip_sl_cents":       smart_flip_sl_cents,
                            "flip_sell_target":    smart_flip_sell_target,
                            "flip_trail":          smart_flip_trail_cents,
                            "best_case_gain":      round(best_case_gain, 2),
                            "worst_case_loss":     round(worst_case_loss, 2),
                            "simulated":           True,
                            "ts":                  datetime.now(timezone.utc).isoformat(),
                            "ts_ms":               int(time.time() * 1000),
                        })
        else:
            # â”€â”€ Chunked IOC submission â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Rather than one large IOC that fails completely if depth at our
            # limit < target, we submit small chunks at the SAME limit price.
            # Each chunk probes remaining depth: if a chunk fills, we ride; if
            # a chunk returns 0 (or partial), the orderbook is exhausted at
            # this price and we stop. Crucially we NEVER walk the price up
            # past the cap â€” that would defeat the strategy's EV protection.
            #
            # The cumulative fill_count drives our P&L bookkeeping. The
            # weighted average fill price across chunks becomes the entry
            # price for settlement math.
            #
            # Lead-offset: when --order-lead-cents N > 0, the initial order
            # submits at limit_price + N to "lead" a fast-moving orderbook.
            # Tested with an EV recheck against the conservative strong-floor
            # (-0.06 default): if the leading price still passes, we use it;
            # otherwise we fall back to the observed price. This captures
            # trades where the orderbook is racing up faster than 1Â¢/poll
            # walks can chase.
            effective_lead = 0
            if order_lead_cents > 0:
                # Recompute combined_p / p_win at the LEADING price
                lead_tp = None
                tp_now = window_tp_cache.get(window_id)
                if tp_now and not tp_now.get("error"):
                    lead_tp = tp_now.get("bs_p_above")
                if lead_tp is not None:
                    lead_combined = ev_tp_weight * lead_tp + (1 - ev_tp_weight) * sig["p_yes"]
                else:
                    lead_combined = sig["p_yes"]
                lead_p_win = lead_combined if rec == "YES" else (1 - lead_combined)
                lead_price = limit_price + order_lead_cents
                lead_p_d   = lead_price / 100.0
                lead_fee   = MAKER_FEE_RATE * lead_p_d * (1 - lead_p_d)
                lead_cost  = lead_p_d + lead_fee
                lead_net   = (1 - lead_p_d) - lead_fee
                lead_ev    = lead_p_win * lead_net - (1 - lead_p_win) * lead_cost
                # 2026-05-29 BUG FIX: use the same floor that APPROVED the trade,
                # not the hardcoded strong-floor. Previously, LATE-SURE trades
                # (with -$0.10 floor) had their lead-walks blocked using the
                # tighter strong-floor (-$0.06). Now lead-walking is consistent
                # with the tier that approved the entry.
                lead_floor = approved_floor
                if lead_ev >= lead_floor:
                    effective_lead = order_lead_cents
                    limit_price    = lead_price
                    log.info(
                        f"   ORDER LEAD +{effective_lead}Â¢ â€” submitting at {limit_price}Â¢ "
                        f"(EV ${lead_ev:+.4f}/c â‰¥ floor ${lead_floor:+.3f}/c, "
                        f"p_win={lead_p_win:.3f})"
                    )
                else:
                    log.info(
                        f"   ORDER LEAD +{order_lead_cents}Â¢ BLOCKED â€” EV at "
                        f"{lead_price}Â¢ would be ${lead_ev:+.4f}/c, below "
                        f"floor ${lead_floor:+.3f}/c. Submitting at observed "
                        f"price {limit_price}Â¢."
                    )
            v2_price_yes_leg = (limit_price / 100) if rec == "YES" else (1 - limit_price / 100)
            CHUNK_SIZE       = 5
            MAX_CHUNKS       = max(10, (contracts // CHUNK_SIZE) + 4)

            filled_count        = 0
            sum_filled_x_price  = 0.0          # for weighted-avg yes-leg price
            order_ids: list[str] = []
            fill_log:  list[str] = []
            api_error = False

            # 2026-05-31 Phase 2: Initialize interleaved hedge state.
            # The helper _hedge_topup_inline is called between YES chunks
            # to fill NO progressively. This captures NO at progressively
            # lower asks as YES walks up (the orderbook moves) and reduces
            # the unhedged window.
            #
            # `pre_active_tier` is the tier locally computed BEFORE the
            # `active_tier` variable is set later in the flow. We need it
            # here for the eligibility check.
            pre_active_tier = locals().get("active_tier") or "standard"
            opposite_ask_at_init = (mkt.get("no_ask") if rec == "YES"
                                     else mkt.get("yes_ask"))
            _hedge_elig, _hedge_reason = hedge_eligible(
                tier=pre_active_tier, side=rec, entry_cents=limit_price,
                opposite_ask_cents=opposite_ask_at_init,
                hedge_enabled=hedge_enabled, hedge_tiers=hedge_tiers,
                hedge_min_yes_entry=hedge_min_yes_entry,
                hedge_max_yes_entry=hedge_max_yes_entry,
                hedge_max_no_cost=hedge_max_no_cost,
            )
            # Compute effective SL for hedge sizing (widened if hedge active)
            _base_sl = (sl_loss_cents_high_conv
                        if pre_active_tier in ("high_conv", "late_sure")
                        else sl_loss_cents)
            _effective_sl_for_hedge = max(_base_sl, hedge_widened_sl_cents)
            hedge_state = {
                "eligible":             _hedge_elig,
                "skip_reason":          _hedge_reason,
                "cumulative_filled":    0,
                "sum_filled_x_price":   0.0,
                "cumulative_cost_usd":  0.0,
                "order_ids":            [],
                "fill_log":             [],
                "warnings":             [],
                "last_no_ask":          opposite_ask_at_init,
                "last_target":          0,
                "last_topup_ts":        None,
                "primary_side":         rec,
                "opposite_side":        "NO" if rec == "YES" else "YES",
                "effective_sl_cents":   _effective_sl_for_hedge,
                "tier":                 pre_active_tier,
            }
            if hedge_enabled and not _hedge_elig:
                log.info(f"  Safety HEDGE skipped (pre-fill): {_hedge_reason}")

            for chunk_idx in range(MAX_CHUNKS):
                remaining = contracts - filled_count
                if remaining <= 0:
                    break
                this_chunk = min(CHUNK_SIZE, remaining)
                body: dict = {
                    "ticker":                     ticker,
                    "client_order_id":            uuid.uuid4().hex,
                    "side":                       "bid" if rec == "YES" else "ask",
                    "count":                      str(this_chunk),
                    "price":                      f"{v2_price_yes_leg:.4f}",
                    "time_in_force":              "immediate_or_cancel",
                    "self_trade_prevention_type": "taker_at_cross",
                }
                try:
                    result = await _kpost("/portfolio/events/orders", body)
                except httpx.HTTPStatusError as e:
                    log.error(f"Polymarket API error on chunk {chunk_idx+1}: "
                              f"{e.response.status_code} {e.response.text}")
                    api_error = True
                    break
                except Exception as e:
                    log.error(f"Chunk {chunk_idx+1} failed: {e}")
                    api_error = True
                    break

                try:
                    chunk_filled = int(float(result.get("fill_count") or 0))
                except (TypeError, ValueError):
                    chunk_filled = 0
                order_ids.append(result.get("order_id") or "unknown")
                chunk_avg_s = result.get("average_fill_price")
                try:
                    chunk_avg = float(chunk_avg_s) if chunk_avg_s else v2_price_yes_leg
                except (TypeError, ValueError):
                    chunk_avg = v2_price_yes_leg

                if chunk_filled == 0:
                    fill_log.append(
                        f"chunk#{chunk_idx+1}: 0/{this_chunk} (depth exhausted at {limit_price}Â¢)"
                    )
                    break

                filled_count       += chunk_filled
                sum_filled_x_price += chunk_filled * chunk_avg
                fill_log.append(
                    f"chunk#{chunk_idx+1}: {chunk_filled}/{this_chunk} @ ${chunk_avg:.4f}"
                )

                # â”€â”€ Phase 2: interleaved hedge top-up â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # Between YES chunks, top up the NO hedge to match the
                # current cumulative YES fill. Catches lower NO ask prices
                # as YES walks up (correlated orderbook), and shrinks the
                # unhedged window. Bounded to 2 NO chunks per call to keep
                # latency manageable.
                if hedge_state.get("eligible"):
                    primary_capital_now = filled_count * limit_price / 100.0
                    await _hedge_topup_inline(
                        hedge_state=hedge_state,
                        current_yes_filled=filled_count,
                        ticker=ticker,
                        primary_side=rec,
                        yes_entry_cents=limit_price,
                        effective_sl_cents=hedge_state["effective_sl_cents"],
                        hedge_no_settle_assumed=hedge_no_settle_assumed,
                        hedge_max_no_cost=hedge_max_no_cost,
                        hedge_max_capital_mult=hedge_max_capital_mult,
                        primary_capital_usd=primary_capital_now,
                        max_chunks_this_call=2,
                )

                # A partial chunk fill means orderbook depth thinning at our price,
                # but we can keep going â€” next chunk may catch newly-arrived bids/asks.
                # Loop will stop naturally on the FIRST 0-fill chunk.

            # â”€â”€ Walking +NÂ¢ retry on UNFILLED contracts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # 2026-05-29 BUG FIX: previously walked ONLY on zero fills. Partial
            # fills (e.g., chunk#1: 3/5, chunk#2: 0/5 â€” total 3/32) skipped the
            # walk entirely. Now we walk whenever any contracts remain unfilled.
            # Each walk step submits ONLY the remaining quantity at the new price
            # and ADDS to the cumulative fill rather than overwriting.
            #
            # When chunked attempt at `limit_price` returns less than full fill,
            # the orderbook moved between observation and submission. Walk the
            # price up one cent at a time, re-checking EV at each step against
            # the applicable floor, until one of:
            #   â€¢ full quantity filled (success)
            #   â€¢ EV at next price drops below floor (math says stop)
            #   â€¢ price would exceed the safety ceiling (hard stop)
            #   â€¢ we've walked the configured maximum (retry_walk_cents)
            walked = 0
            retry_stop_reason: Optional[str] = None
            while (filled_count < contracts and not api_error
                   and retry_walk_cents > 0 and walked < retry_walk_cents):
                walked         += 1
                new_limit_price = limit_price + walked

                # Recompute EV at new price (combined_p if EV gate is on, else Markov-only)
                tp_cached_now = window_tp_cache.get(window_id)
                if ev_gate and tp_cached_now and not tp_cached_now.get("error"):
                    retry_p_yes = ev_tp_weight * tp_cached_now["bs_p_above"] + \
                                  (1 - ev_tp_weight) * sig["p_yes"]
                else:
                    retry_p_yes = sig["p_yes"]
                retry_p_d   = new_limit_price / 100
                retry_fee   = MAKER_FEE_RATE * retry_p_d * (1 - retry_p_d)
                retry_cost  = retry_p_d + retry_fee
                retry_net   = (1 - retry_p_d) - retry_fee
                retry_pwin  = retry_p_yes if rec == "YES" else (1 - retry_p_yes)

                # Determine applicable tier & ceiling for THIS price
                retry_floor    = ev_floor if ev_gate else 0.0
                retry_ceiling  = ev_ceiling
                retry_tier_tag = ""
                # 2026-05-28 PM: re-evaluate orderbook lock-in at each walk
                # step so HC/LATE-SURE bypasses stay live during the retry
                # ladder. Uses the most recent ob_snap from this poll iteration.
                retry_ob_lockin = False
                if ev_gate and orderbook_lockin_enabled:
                    retry_ob_lockin, _ = _is_orderbook_lockin(
                        orderbook=ob_snap,
                        new_rec=rec,
                        spread_max=orderbook_lockin_spread_max,
                        price_min=orderbook_lockin_price_min,
                        gap=sig.get("gap", 0.0),
                        gap_min=orderbook_lockin_gap_min,
                    )
                retry_effective_ls_cap = (
                    late_window_price_max_lockin if retry_ob_lockin
                    else late_window_price_max
                )
                if ev_gate:
                    retry_is_late = False
                    retry_is_hc   = False
                    retry_is_ld   = False
                    # 2026-05-28 PM: both checks accept orderbook_lockin_bypass.
                    # _is_late_window_sure now allows TP-extreme bypass under
                    # lock-in (mirrors HC). Allows LATE-SURE to fire even when
                    # TP_bs is below 0.75 if orderbook confirms.
                    retry_is_late, _ = _is_late_window_sure(
                        mins_left=sig["minutes_left"],
                        limit_price=new_limit_price,
                        tp_cached=tp_cached_now,
                        new_rec=rec,
                        markov_p_yes=sig["p_yes"],
                        recent_5m=recent_5m,
                        late_window_mins=late_window_mins,
                        late_window_price_max=retry_effective_ls_cap,
                        late_window_min_tp=late_window_min_tp,
                        orderbook_lockin_bypass=retry_ob_lockin,
                    )
                    retry_is_hc, _ = _is_high_conviction(
                        mins_left=sig["minutes_left"],
                        limit_price=new_limit_price,
                        tp_cached=tp_cached_now,
                        new_rec=rec,
                        sig=sig,
                        recent_5m=recent_5m,
                        high_conv_gap_min=high_conv_gap_min,
                        high_conv_persist_min=high_conv_persist_min,
                        high_conv_tp_strong=high_conv_tp_strong,
                        high_conv_price_max=high_conv_price_max,
                        high_conv_max_mins=high_conv_max_mins,
                        orderbook_lockin_bypass=retry_ob_lockin,
                        hc_low_hurst_veto_enabled=hc_low_hurst_veto_enabled,
                        hc_low_hurst_threshold=hc_low_hurst_threshold,
                        hc_low_hurst_markov_extremity=hc_low_hurst_markov_extremity,
                        dist_pct=mkt.get("dist_pct"),
                        hc_dist_min=hc_dist_min,
                    )
                    if late_dir_enabled and tp_cached_now is not None and not tp_cached_now.get("error"):
                        retry_is_ld, _ = _is_late_window_directional(
                            mins_left=sig["minutes_left"],
                            gap=sig.get("gap", 0.0),
                            persist=sig.get("persist", 0.0),
                            tp_p_yes=tp_cached_now.get("bs_p_above"),
                            new_rec=rec,
                            btc_price=mkt["btc_price"],
                            strike=mkt["strike"],
                            recent_5m=recent_5m,
                            late_dir_mins=late_dir_mins,
                            late_dir_gap_min=late_dir_gap_min,
                            late_dir_persist_min=late_dir_persist_min,
                            late_dir_distance_min=late_dir_distance_min,
                            late_dir_momentum_min=late_dir_momentum_min,
                            late_dir_strong_distance=late_dir_strong_distance,
                        )
                    # Mirror the meaningful-TP requirement + timing cap + net
                    # momentum check from the main is_strong logic.
                    retry_yes_lean = sig["p_yes"] > 0.5
                    retry_tp_p = (tp_cached_now or {}).get("bs_p_above")
                    retry_tp_meaningful = (
                        retry_tp_p is not None
                        and (
                            (retry_yes_lean and retry_tp_p >= ev_strong_tp_min)
                            or (not retry_yes_lean and retry_tp_p <= (1 - ev_strong_tp_min))
                        )
                    )
                    retry_timing_ok = sig["minutes_left"] <= ev_strong_max_mins
                    retry_recent_sum = sum(recent_5m or [])
                    if retry_yes_lean:
                        retry_momentum_ok = retry_recent_sum >= -ev_strong_max_adverse_momentum
                    else:
                        retry_momentum_ok = retry_recent_sum <= ev_strong_max_adverse_momentum
                    retry_is_strong = (
                        sig["gap"] >= ev_strong_gap_min
                        and new_limit_price <= ev_strong_price_max
                        and tp_cached_now is not None
                        and not tp_cached_now.get("error")
                        and retry_tp_meaningful
                        and retry_timing_ok
                        and retry_momentum_ok
                    )
                    # Markov-only p_win for use when lock-in bypass is active
                    # (mirrors main EV gate logic).
                    retry_markov_dir = (sig["p_yes"] if rec == "YES"
                                        else 1.0 - sig["p_yes"])
                    if retry_is_late:
                        if retry_ob_lockin:
                            # Lock-in path: TP is bypassed, use Markov-only p_win
                            retry_pwin = max(retry_pwin, retry_markov_dir)
                        elif tp_cached_now and not tp_cached_now.get("error"):
                            # Standard LATE-SURE path: boost with TP
                            tp_p_dir = (tp_cached_now["bs_p_above"]
                                        if rec == "YES"
                                        else 1.0 - tp_cached_now["bs_p_above"])
                            retry_pwin = max(retry_pwin, tp_p_dir)
                        retry_floor   = late_window_ev_floor
                        retry_ceiling = retry_effective_ls_cap
                        retry_tier_tag = (" [LATE-SURE bypass:ob-lockin]"
                                          if retry_ob_lockin else " [LATE-SURE]")
                    elif retry_is_hc:
                        if retry_ob_lockin:
                            # Lock-in HC path: Markov-only p_win, looser floor
                            retry_pwin   = max(retry_pwin, retry_markov_dir)
                            retry_floor  = hc_lockin_ev_floor
                            retry_tier_tag = " [HIGH-CONV bypass:ob-lockin]"
                        else:
                            if tp_cached_now and not tp_cached_now.get("error"):
                                tp_p_dir = (tp_cached_now["bs_p_above"]
                                            if rec == "YES"
                                            else 1.0 - tp_cached_now["bs_p_above"])
                                retry_pwin = max(retry_pwin, tp_p_dir)
                            retry_floor = high_conv_ev_floor
                            retry_tier_tag = " [HIGH-CONV]"
                        retry_ceiling = high_conv_price_max
                    elif retry_is_ld:
                        # Late-dir: no p_win boost (uses combined directly)
                        retry_floor   = late_dir_ev_floor
                        retry_ceiling = late_dir_price_max
                        retry_tier_tag = " [LATE-DIR]"
                    elif retry_is_strong:
                        retry_floor   = ev_strong_floor
                        retry_tier_tag = " [STRONG]"

                retry_ev = retry_pwin * retry_net - (1 - retry_pwin) * retry_cost

                # Hard safety: never walk above the applicable ceiling
                if new_limit_price > retry_ceiling:
                    retry_stop_reason = f"price {new_limit_price}Â¢ > ceiling {retry_ceiling}Â¢"
                    log.info(f"   +{walked}Â¢ walk STOPPED: {retry_stop_reason}")
                    break

                # â”€â”€ 2026-06-15: SURE-TRADE EV WALK-UP OVERRIDE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # On high-conviction trades (p_win high) where the Binance
                # futures book skew AGREES (book_skew>0), relax the EV floor a
                # few cents to win the fill. Backtest: strong + book_skew>0 +
                # 78-86c was +EV (actual WR exceeded price), while the SAME
                # band with book_skew<=0 was strongly -EV â€” so book_skew is the
                # gate that separates "sure" from "overpaying". Tight scope +
                # flag; never relaxes below ev_override_floor.
                eff_floor = retry_floor
                _ov_active = False
                if (ev_walkup_override_enabled
                        and retry_pwin >= ev_override_pwin_min
                        and new_limit_price <= ev_override_price_max
                        and _ns is not None
                        and _ns.get("book_skew") is not None
                        and _ns["book_skew"] >= ev_override_book_skew_min):
                    eff_floor = min(retry_floor, ev_override_floor)
                    _ov_active = True

                if retry_ev < eff_floor:
                    tier_note = retry_tier_tag if retry_tier_tag else " (no tier qualifies at this price â†’ standard floor)"
                    retry_stop_reason = (
                        f"EV at {new_limit_price}Â¢ is ${retry_ev:+.4f}/c, "
                        f"below floor ${eff_floor:+.3f}/c{tier_note}"
                        + (" [override-exhausted]" if _ov_active else "")
                    )
                    log.info(f"   +{walked}Â¢ walk STOPPED: {retry_stop_reason}")
                    break
                if _ov_active and retry_ev < retry_floor:
                    # Override is carrying this fill past the normal floor.
                    log.info(
                        f"   +{walked}Â¢ EV-OVERRIDE [sure+book]: walking to "
                        f"{new_limit_price}\u00a2 (p_win={retry_pwin:.3f}, "
                        f"book_skew={_ns['book_skew']:+.3f}, EV ${retry_ev:+.4f}/c "
                        f"\u2265 override floor ${ev_override_floor:+.3f}, "
                        f"normal floor was ${retry_floor:+.3f})")

                # Submit IOC at the new price for the REMAINING unfilled quantity
                remaining_to_fill = contracts - filled_count
                new_v2_price = (new_limit_price / 100) if rec == "YES" else (1 - new_limit_price / 100)
                log.info(
                    f"   +{walked}Â¢ WALK{retry_tier_tag} â€” trying {new_limit_price}Â¢ "
                    f"for {remaining_to_fill}c (EV ${retry_ev:+.4f}/c, p_win={retry_pwin:.3f})"
                )
                retry_body: dict = {
                    "ticker":                     ticker,
                    "client_order_id":            uuid.uuid4().hex,
                    "side":                       "bid" if rec == "YES" else "ask",
                    "count":                      str(remaining_to_fill),
                    "price":                      f"{new_v2_price:.4f}",
                    "time_in_force":              "immediate_or_cancel",
                    "self_trade_prevention_type": "taker_at_cross",
                }
                try:
                    retry_result = await _kpost("/portfolio/events/orders", retry_body)
                    try:
                        retry_filled = int(float(retry_result.get("fill_count") or 0))
                    except (TypeError, ValueError):
                        retry_filled = 0
                    retry_avg_s = retry_result.get("average_fill_price")
                    try:
                        retry_avg = float(retry_avg_s) if retry_avg_s else new_v2_price
                    except (TypeError, ValueError):
                        retry_avg = new_v2_price
                    if retry_filled > 0:
                        order_ids.append(retry_result.get("order_id") or "unknown")
                        # 2026-05-29 BUG FIX: ADD to filled_count + weighted price
                        # rather than overwriting. Walking can now top-up partial
                        # fills from earlier chunks/walks.
                        filled_count       += retry_filled
                        sum_filled_x_price += retry_filled * retry_avg
                        # Update bookkeeping with most-recent walk price (used
                        # by downstream logging as the highest paid price)
                        limit_price        = new_limit_price
                        v2_price_yes_leg   = new_v2_price
                        fill_log.append(
                            f"+{walked}Â¢ walk: {retry_filled}/{remaining_to_fill} @ ${retry_avg:.4f}"
                        )
                        log.info(f"   +{walked}Â¢ walk FILLED â€” {retry_filled}/{remaining_to_fill}c at ${retry_avg:.4f} (cumulative {filled_count}/{contracts})")
                        # Stop only if we got the FULL remaining quantity. Otherwise
                        # keep walking up to fill the rest.
                        if filled_count >= contracts:
                            break
                    else:
                        fill_log.append(
                            f"+{walked}Â¢ walk: 0/{remaining_to_fill} (orderbook empty at {new_limit_price}Â¢)"
                        )
                except Exception as e:
                    log.warning(f"+{walked}Â¢ walk submit failed: {e}")
                    retry_stop_reason = f"API error: {e}"
                    break

            if filled_count <= 0:
                walk_msg = ""
                if retry_walk_cents > 0:
                    if walked > 0:
                        walk_msg = (f", walked {walked} cent(s)"
                                    + (f" â€” stopped: {retry_stop_reason}" if retry_stop_reason else ""))
                    else:
                        walk_msg = " (walk disabled or skipped pre-attempt)"

                # Bump the per-window failed-fill counter
                window_no_fills[window_id] = window_no_fills.get(window_id, 0) + 1
                attempt_n = window_no_fills[window_id]

                if attempt_n >= max_window_fill_attempts:
                    log.warning(
                        f"NO FILLS â€” requested {contracts}c @ {limit_price}\u00a2{walk_msg}. "
                        f"Attempt {attempt_n}/{max_window_fill_attempts} hit cap; abandoning window."
                    )
                    session.traded.add(window_id)
                    await asyncio.sleep(max(5, (mins_left + 0.5) * 60))
                    continue
                else:
                    # Re-evaluate next poll. Don't mark window as traded â€” if
                    # signal still approves AND orderbook is back in our range
                    # at the next poll, the daemon will fire again.
                    log.warning(
                        f"NO FILLS â€” requested {contracts}c @ {limit_price}Â¢{walk_msg}. "
                        f"Attempt {attempt_n}/{max_window_fill_attempts}; "
                        f"re-evaluating signal in {refill_retry_sleep_s:.0f}s."
                    )
                    await asyncio.sleep(refill_retry_sleep_s)
                    continue

            # Build a synthetic avg_fill_price (yes-leg dollars) for downstream
            # P&L math â€” matches the V2 API's string format.
            avg_fill_yes_leg = sum_filled_x_price / filled_count
            avg_fill_price   = f"{avg_fill_yes_leg:.4f}"
            order_id         = order_ids[0] if order_ids else "unknown"

            log.info(
                f"ORDER PLACED â€” {len(order_ids)} chunk(s), filled={filled_count}/{contracts}c "
                f"avg_yes_leg=${avg_fill_yes_leg:.4f}  "
                f"({'partial' if filled_count < contracts else 'full'} fill"
                f"{', API error stopped further chunks' if api_error else ''})"
            )
            for line in fill_log:
                log.info(f"   {line}")

        session.traded.add(window_id)

        # Queue for settlement check using actual filled count (and actual avg
        # fill price if Polymarket returned one â€” slightly tighter P&L accounting).
        if not dry_run:
            if avg_fill_price:
                try:
                    avg_yes_leg = float(avg_fill_price)
                    avg_p_d     = avg_yes_leg if rec == "YES" else (1 - avg_yes_leg)
                except (TypeError, ValueError):
                    avg_p_d = p_d
            else:
                avg_p_d = p_d
            avg_fee_c = MAKER_FEE_RATE * avg_p_d * (1 - avg_p_d)
            avg_cost  = avg_p_d   + avg_fee_c
            avg_net   = (1 - avg_p_d) - avg_fee_c

            # Track which tier this trade was placed under. Used by the SL
            # monitor to apply tier-specific stop-loss thresholds (tighter
            # for HIGH-CONV positions where downside is biggest).
            # When EV-gate didn't fire (price was within standard caps),
            # active_tier stays "" â†’ default to "standard" for the saved
            # field.
            saved_tier = locals().get("active_tier") or "standard"
            session.pending[window_id] = {
                "ticker":      ticker,
                "side":        rec,
                "contracts":   filled_count,
                "limit_price": limit_price,
                "cost":        round(avg_cost * filled_count, 2),
                "net_win":     round(avg_net  * filled_count, 2),
                "order_id":    order_id,
                "entry_yes_leg": (float(avg_fill_price) if avg_fill_price
                                  else v2_price_yes_leg),
                "entry_ts":    datetime.now(timezone.utc).isoformat(),
                "tier":        saved_tier,
                # 2026-05-31: safety hedge state (populated by hedge-attach below)
                "hedge":       None,
                # 2026-06-03: cached entry signals for FADE-BOUNCE dual-entry
                # decisions inside the SL monitor. These don't refresh during
                # monitoring (signals are expensive to recompute), so we use
                # the entry-time values for the fade-bounce qualifier check.
                "entry_markov_p_yes": sig.get("p_yes", 0.5) if isinstance(sig, dict) else 0.5,
                "entry_hurst":        sig.get("hurst", 0.5) if isinstance(sig, dict) else 0.5,
                "entry_dist_pct":     mkt.get("dist_pct", 0) if isinstance(mkt, dict) else 0,
                "entry_gap":          (sig.get("gap", 0.0) if isinstance(sig, dict) else 0.0),
                # Populated by FADE-BOUNCE attach (if it fires)
                "fade_bounce":        None,
                # 2026-06-12: PATIENT TOP-UP state. intended = the largest
                # size requested across fire attempts in this window; the
                # top-up watcher (in the SL monitor) tries to complete it.
                "intended_contracts": max(window_max_intent.get(window_id, 0),
                                          filled_count),
                "topup_last_try_ts":  0.0,
                "topup_best_ask":     None,
                "topup_filled_total": 0,
                # 2026-06-15: entry perp momentum, for the perp-confirmed
                # take-profit gate (TP only earns its keep on m30s>0 trades).
                "entry_perp_m30s":    (_ns.get("perp_m30s") if _ns else None),
                # 2026-06-15: once the TP target is hit, commit to selling the
                # whole position out in profit (partial-fill remainders keep
                # selling at any profitable price, not just at +TP).
                "tp_committed":       False,
            }

            # â”€â”€ 2026-05-31 Phase 2: SAFETY HEDGE FINAL TOP-UP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # The hedge was already incrementally filled during the YES
            # chunked loop (interleaved). Now that YES is FULLY filled
            # (including walking-retry phase), do a final top-up call to
            # catch any remaining hedge needs that earlier polls missed
            # (e.g., NO ask was too high earlier but dropped now that
            # YES has finished walking up).
            opposite_side = hedge_state.get("opposite_side", "NO" if rec == "YES" else "YES")
            if hedge_state.get("eligible") and not dry_run:
                # Final top-up â€” allow up to 5 more chunks to catch up
                await _hedge_topup_inline(
                    hedge_state=hedge_state,
                    current_yes_filled=filled_count,
                    ticker=ticker,
                    primary_side=rec,
                    yes_entry_cents=limit_price,
                    effective_sl_cents=hedge_state["effective_sl_cents"],
                    hedge_no_settle_assumed=hedge_no_settle_assumed,
                    hedge_max_no_cost=hedge_max_no_cost,
                    hedge_max_capital_mult=hedge_max_capital_mult,
                    primary_capital_usd=filled_count * limit_price / 100.0,
                    max_chunks_this_call=5,
                )
            if not hedge_state.get("eligible"):
                # Already logged the skip reason pre-fill.
                pass
            else:
                # Persist the aggregated hedge state to pos["hedge"].
                hedge_filled = hedge_state.get("cumulative_filled", 0)
                if hedge_filled > 0:
                    hedge_avg_yes_leg = (hedge_state["sum_filled_x_price"]
                                          / hedge_filled)
                    hedge_avg_cents = (round((1 - hedge_avg_yes_leg) * 100)
                                        if opposite_side == "NO"
                                        else round(hedge_avg_yes_leg * 100))
                    actual_hedge_capital = round(
                        hedge_state.get("cumulative_cost_usd", 0), 2
                    )
                    session.pending[window_id]["hedge"] = {
                        "active":            True,
                        "side":              opposite_side,
                        "contracts_target":  hedge_state.get("last_target", hedge_filled),
                        "contracts_filled":  hedge_filled,
                        "entry_yes_leg":     hedge_avg_yes_leg,
                        "entry_cents":       hedge_avg_cents,
                        "entry_ts":          datetime.now(timezone.utc).isoformat(),
                        "order_ids":         list(hedge_state.get("order_ids", [])),
                        "status":            "active",
                        "capital_usd":       actual_hedge_capital,
                        "widened_sl_cents":  hedge_state["effective_sl_cents"],
                        "interleaved":       True,  # Phase 2 flag
                        "n_topup_chunks":    len(hedge_state.get("fill_log", [])),
                    }
                    log.info(
                        f"  Safety HEDGE ATTACHED (interleaved): BUY {opposite_side} "
                        f"{hedge_filled}/{hedge_state.get('last_target', hedge_filled)}c "
                        f"avg={hedge_avg_cents}\u00a2 capital=${actual_hedge_capital:.2f}  "
                        f"SL widened to {hedge_state['effective_sl_cents']}\u00a2  "
                        f"({len(hedge_state.get('fill_log', []))} top-up chunk(s))"
                    )
                    for ln in hedge_state.get("fill_log", []):
                        log.info(f"   {ln}")
                    if hedge_state.get("warnings"):
                        for w in hedge_state["warnings"]:
                            log.info(f"   hedge warning: {w}")
                    # Audit the hedge attach as a separate record
                    audit.write({
                        "type":              "HEDGE_ATTACH",
                        "window_id":         window_id,
                        "ticker":            ticker,
                        "primary_side":      rec,
                        "primary_entry":     limit_price,
                        "primary_contracts": filled_count,
                        "primary_tier":      saved_tier,
                        "hedge_side":        opposite_side,
                        "hedge_target":      hedge_state.get("last_target", hedge_filled),
                        "hedge_filled":      hedge_filled,
                        "hedge_entry_cents": hedge_avg_cents,
                        "hedge_capital_usd": actual_hedge_capital,
                        "effective_sl_cents": hedge_state["effective_sl_cents"],
                        "no_settle_assumed": hedge_no_settle_assumed,
                        "interleaved":       True,
                        "n_topup_chunks":    len(hedge_state.get("fill_log", [])),
                        "warnings":          list(hedge_state.get("warnings", [])),
                        "ts":                datetime.now(timezone.utc).isoformat(),
                        "ts_ms":             int(time.time() * 1000),
                    })
                else:
                    log.warning(
                        f"  Safety HEDGE: ZERO fills across interleaved + final top-up. "
                        f"Primary position remains UNHEDGED."
                    )
                    if hedge_state.get("warnings"):
                        for w in hedge_state["warnings"]:
                            log.info(f"   hedge warning: {w}")

        # Audit: record the trade firing + full execution detail
        # Snapshot OKX at trade-decision moment too
        okx_snap_trade = None
        if okx_feed is not None:
            try:
                okx_snap_trade = okx_feed.get_recent_move(
                    lookback_s=futures_lead_lookback_s
                )
            except Exception:
                okx_snap_trade = None
        audit.write(build_poll_record(
            window_id=window_id, ticker=ticker, close_dt=close_dt,
            decision="TRADE", rec=rec, reasons=[],
            signal=sig, market=mkt, recent_5m_pct=recent_5m,
            tp_cached=window_tp_cache.get(window_id),
            db_check=db_check, ev_calc=None,
            params=audit_params,
            okx_check=okx_snap_trade,
            orderbook=ob_snap,
            trade_flow=tr_snap,
        ))
        audit.write(build_trade_record(
            window_id=window_id, ticker=ticker, side=rec,
            limit_price_cents=limit_price,
            contracts_requested=contracts,
            contracts_filled=filled_count,
            avg_fill_yes_leg=(float(avg_fill_price) if avg_fill_price else v2_price_yes_leg),
            order_ids=order_ids if not dry_run else ["dry-run"],
            chunks=(fill_log if not dry_run else []),
            retry_fired=(not dry_run and len(fill_log) > 0
                          and any("walk" in line for line in fill_log)),
            retry_filled=(not dry_run and len(fill_log) > 0
                          and any("walk:" in line and "0/" not in line for line in fill_log)),
            max_loss=signal["max_loss_usd"],
            expected_value=signal["expected_value"],
            params=audit_params,
        ))

        # â”€â”€ Stop-loss monitor (15m-specific) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # 15m positions are held only 3-10 min from entry to settlement, so
        # we don't bother with a take-profit (which the 1hr daemon has).
        # We DO have a stop-loss to cap downside on the rare loss trades
        # â€” typically once-per-day-ish at our 92%+ WR. Without SL, a HIGH-
        # CONV trade at 90Â¢ that goes wrong loses the full $0.90/c Ã— 25c
        # = ~$22 (seen on 2026-05-27). A 45Â¢ SL caps that loss to ~$11.
        #
        # Implementation: poll every sl_poll_interval_s, verify position
        # with Polymarket before SELL (defensive â€” prevents over-sell that
        # could create unintended short positions), and use mid-price by
        # default for the trigger to react to orderbook moves faster than
        # bid alone.
        pos = session.pending.get(window_id)
        if (sl_enabled and pos and filled_count > 0 and not dry_run):
            entry_yes_leg = pos["entry_yes_leg"]
            entry_cents   = pos["limit_price"]
            entry_ts      = datetime.fromisoformat(
                pos["entry_ts"].replace("Z", "+00:00"))
            # Tier-specific SL threshold. HIGH-CONV and LATE-SURE trades get
            # a tighter stop (default 30Â¢) because their downside is
            # asymmetric: upside is capped at 100Â¢ - entry (e.g., 7Â¢ for a
            # 93Â¢ trade) while downside can be most of entry cost. Tighter
            # stop limits the rare losses without blocking winners (these
            # tiers rarely dip 30Â¢+ from entry and recover).
            #
            # LATE-SURE added 2026-05-28 after raising its cap 89â†’96Â¢: at
            # 96Â¢ entry, the wider 45Â¢ SL would mean a 45Â¢/c loss on the
            # rare bad trade. Tightening to 30Â¢ matches HIGH-CONV's
            # protection level for high-price tiers.
            position_tier = pos.get("tier", "standard")
            uses_tight_sl = position_tier in ("high_conv", "late_sure")
            effective_sl_cents = (sl_loss_cents_high_conv if uses_tight_sl
                                  else sl_loss_cents)
            # 2026-05-31: when a safety hedge is attached, the position can
            # tolerate a deeper SL because the NO leg will offset most of the
            # loss. Widening the SL reduces false-stops on whip-saws while the
            # hedge still caps the worst case. The hedge sizing math used
            # this same widened number when computing hedge contracts.
            hedge_state = pos.get("hedge")
            if hedge_state and hedge_state.get("active"):
                widened = hedge_state.get("widened_sl_cents", hedge_widened_sl_cents)
                if widened > effective_sl_cents:
                    effective_sl_cents = widened
            # 2026-05-30: HC + LATE-SURE use faster poll cadence. Audit of all
            # observed HC SL exits showed the bid blew past trigger BETWEEN
            # 5-second polls â€” the daemon never saw an intermediate price.
            # 2s polling gives 2.5x more chances to catch the bid mid-drop.
            effective_poll_interval = (sl_poll_interval_hc_s if uses_tight_sl
                                       else sl_poll_interval_s)
            log.info(
                f"  â”€â”€ STOP-LOSS MONITOR active: entry={entry_cents}Â¢ {rec} "
                f"(tier={position_tier}), stop=-{effective_sl_cents}Â¢ trigger({sl_trigger_mode}), "
                f"grace={sl_grace_mins:.1f}min, late-disable<{sl_disable_late_mins:.1f}min, "
                f"poll every {effective_poll_interval:.0f}s"
            )
            # 2026-05-31: track bid trajectory during the hold so we can later
            # answer "would a tighter SL have caught winners?" From the audit
            # alone we only see entry/exit; this captures min/max bid + when
            # the worst dip happened. Written to audit on exit (SL or via
            # settled record reading from pos dict).
            pos["min_bid_cents"]      = None  # worst observed bid for our side
            pos["max_bid_cents"]      = None
            pos["max_drawdown_cents"] = 0     # entry - min_bid
            pos["min_bid_secs_from_entry"] = 0
            pos["n_sl_polls"]         = 0
            pos["_pcross_polls"]      = 0       # predict-cross-exit confirm streak
            pos["_pcross_fired"]      = False   # one fire per position
            pos["_holdwin_active"]    = False   # hold-to-win: TP currently cancelled
            pos["_holdwin_peak"]      = None    # peak favorable bid while holding

            # â”€â”€ 2026-06-11: REVERSAL-RISK MONITOR (RRM) â€” LOG-ONLY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Multi-signal confirmation score for genuine adverse moves:
            # mandatory strike-breach gate + velocity + liq cascade + taker
            # surge + OI build + time pressure. Fires a WOULD-EXIT log/audit
            # record; NEVER sells. Validation: join RRM_WOULD_EXIT records
            # with settlements to measure saved-on-losers vs cost-on-winners.
            rrm_state = {
                "anchor_btc":  mkt.get("btc_price"),
                "anchor_perp": latest_perp_mid(),
                "strike":      mkt.get("strike"),
                "breach_logged": False,
                "fired": False,
                "max_score": 0,
            }
            if not (rrm_state["anchor_btc"] and rrm_state["anchor_perp"]
                    and rrm_state["strike"]):
                log.info("  RRM: inactive for this position (missing anchor/feed)")

            # â”€â”€ 2026-06-15: place RESTING TAKE-PROFIT order â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # On a perp-confirmed position, rest a GTC limit sell at entry+TP.
            # Polymarket fills it the instant the bid touches the level (captures
            # spikes the poll misses). We track fills + cancel it before any
            # other exit (below) and near settlement. pos["resting_tp_id"]
            # holds the live order; None when not placed/already cancelled.
            pos["resting_tp_id"] = None
            pos["resting_tp_qty"] = 0
            pos["resting_tp_filled"] = 0
            pos["resting_tp_price"] = None

            # â”€â”€ 2026-06-27: HIGH-RISK TIGHT-TP state. Rather than manage a
            # second order (oversell-prone), this REUSES the normal resting-TP
            # machinery: when the high-risk condition trips we cancel & re-place
            # the resting TP at a tight entry+K target (`highrisk_target`), then
            # mark `highrisk_armed` so the late-disable keeps it alive to the
            # close (it's a profit price, never a bad late fill). All existing
            # lifecycle (fill polling, topup-resize re-place, exit cancels) then
            # applies unchanged because it's the same resting_tp_id.
            pos["highrisk_target"] = None      # tight TP price once armed
            pos["highrisk_armed"]  = False      # fired once per position

            async def _arm_highrisk_tp():
                """Arm the tight high-risk TP if a high_conv position has drawn
                down past the threshold while close to strike. Fires once."""
                if (not highrisk_tp_enabled or not resting_tp_enabled
                        or pos.get("highrisk_armed")):
                    return
                if pos.get("tier", "standard") != "high_conv":
                    return
                if pos.get("max_drawdown_cents", 0) < highrisk_tp_dd_cents:
                    return
                _ed = pos.get("entry_dist_pct")
                if _ed is None or abs(_ed) > highrisk_tp_dist_max:
                    return
                _basis = await get_venue_position_avg_cents(pos["ticker"])
                if _basis is None:
                    _basis = entry_cents
                tp_price = int(_basis) + highrisk_tp_cents
                pos["highrisk_armed"] = True     # one shot regardless of outcome
                if tp_price > 99 or tp_price <= _basis or pos["contracts"] <= 0:
                    log.info(f"  [HIGHRISK-TP] no room to lock "
                             f"+{highrisk_tp_cents}\u00a2 (basis {_basis}\u00a2) â€” skip.")
                    return
                pos["highrisk_target"] = tp_price
                # replace any existing resting TP with the tight target
                if pos.get("resting_tp_id"):
                    await _cancel_resting_tp("highrisk-tp-replace")
                placed = await _place_resting_tp()
                audit.write({
                    "type": "HIGHRISK_TP_ARMED", "window_id": window_id,
                    "ticker": pos["ticker"], "side": pos["side"],
                    "contracts": int(pos["contracts"]), "entry_cents": _basis,
                    "tp_price": tp_price, "placed": bool(placed),
                    "max_drawdown_cents": pos.get("max_drawdown_cents", 0),
                    "entry_dist_pct": round(_ed, 4),
                    "mins_remaining": round(mins_remaining, 2)})
                log.warning(
                    f"  [HIGHRISK-TP] ARMED tight sell @ {tp_price}\u00a2 "
                    f"(entry {_basis}\u00a2, drawdown "
                    f"{pos.get('max_drawdown_cents',0)}\u00a2, dist {abs(_ed):.3f}%, "
                    f"{mins_remaining:.1f}min left) â€” kept alive to close")

            async def _place_resting_tp():
                """Place/replace the resting TP for the current position size.
                Returns True if an order is now live. Never raises.

                Two target modes:
                  * HIGH-PRICE CEILING: entry >= high_price_tp_min_cents -> rest
                    at high_price_tp_target_cents (e.g. 98). Price-based, all
                    tiers, not perp-gated. Handles "sure" 89Â¢+ trades whose
                    +10Â¢ target would exceed 99Â¢.
                  * STANDARD +NÂ¢: entry + take_profit_cents, gated by the
                    perp-confirmed rules.
                """
                if not resting_tp_enabled:
                    return False
                # 2026-06-15: base the target on Polymarket's ACTUAL average cost,
                # not the daemon's blended limit_price (which drifts above the
                # true average via top-up rounding and pushes the TP too high).
                # Fall back to limit_price if the query fails.
                _basis = await get_venue_position_avg_cents(pos["ticker"])
                if _basis is None:
                    _basis = entry_cents
                if pos.get("highrisk_target"):
                    # 2026-06-27: high-risk tight TP overrides ceiling/standard.
                    tp_price = pos["highrisk_target"]
                elif (high_price_tp_enabled
                        and _basis >= high_price_tp_min_cents):
                    tp_price = high_price_tp_target_cents
                elif take_profit_enabled:
                    _ep = pos.get("entry_perp_m30s")
                    if take_profit_perp_confirmed_only and not (
                            _ep is not None and _ep > take_profit_perp_min):
                        return False
                    tp_price = _basis + take_profit_cents
                else:
                    return False
                if tp_price > 99 or tp_price <= _basis:
                    return False   # unreachable target or no profit
                qty = pos["contracts"]
                if qty <= 0:
                    return False
                _pside = pos["side"]; _ptkr = pos["ticker"]
                try:
                    # Documented CreateOrder (POST /portfolio/orders): a SELL on
                    # our outcome side, resting at the TP price.
                    #   action="sell", side="yes"|"no", yes_price/no_price in
                    #   integer cents, time_in_force="good_till_canceled" (rests
                    #   until filled/cancelled).
                    # NOTE: reduce_only is NOT used â€” verified 2026-06-15 the API
                    # rejects it on GTC orders ("reduce_only can only be used
                    # with IoC orders", HTTP 400). So net-short protection relies
                    # on the software guards instead: startup orphan cleanup,
                    # confirmed cancel-before-exit, late-disable cancel, and the
                    # SL sell re-querying Polymarket's live position (sells min).
                    # 2026-06-18: V2 migration (legacy POST /portfolio/orders â†’ 410).
                    # V2 YES-leg: SELL = ask (YES pos) / bid (NO pos) at YES-leg price.
                    _yes_leg = (tp_price / 100.0) if _pside == "YES" else (1.0 - tp_price / 100.0)
                    body = {
                        "ticker": _ptkr,
                        "client_order_id": uuid.uuid4().hex,
                        "side": "ask" if _pside == "YES" else "bid",
                        "count": str(int(qty)),
                        "price": f"{_yes_leg:.4f}",
                        "time_in_force": "good_till_canceled",
                        "self_trade_prevention_type": "taker_at_cross",
                    }
                    res = await _kpost("/portfolio/events/orders", body)
                    oid = (res.get("order") or res).get("order_id") if isinstance(res, dict) else None
                    if oid:
                        pos["resting_tp_id"] = oid
                        pos["resting_tp_qty"] = qty
                        pos["resting_tp_price"] = tp_price
                        pos["resting_tp_entry"] = _basis   # cost basis for PnL
                        _mode = ("ceiling" if (high_price_tp_enabled
                                 and _basis >= high_price_tp_min_cents)
                                 else f"+{take_profit_cents}\u00a2")
                        _drift = (f" [venue avg {_basis}\u00a2 vs tracked "
                                  f"{entry_cents}\u00a2]" if _basis != entry_cents else "")
                        log.warning(
                            f"  [RESTING-TP] placed sell {qty}c {_pside} @ "
                            f"{tp_price}\u00a2 ({_mode}, entry {_basis}\u00a2){_drift} "
                            f"id={oid[:8]}")
                        return True
                except Exception as e:
                    log.warning(f"  [RESTING-TP] place failed (non-fatal): {e!r}")
                return False

            async def _cancel_resting_tp(reason="exit"):
                """Cancel the live resting TP, CONFIRM it's gone, and reconcile
                any fills into the position. Returns (newly_filled, confirmed).
                'confirmed' is True only when the order is verified no longer
                resting (cancelled/filled/not-found). Callers about to market-
                sell MUST NOT sell unless confirmed, else a still-live resting
                order could fill later and put us net short (double-sell).
                Never raises."""
                oid = pos.get("resting_tp_id")
                if not oid:
                    return 0, True
                newly_filled = 0
                confirmed = False
                # Snapshot fills before cancelling (canonical fill_count_fp).
                try:
                    st = await _korder_status(oid)
                    if st:
                        fc = _order_fill_count(st)
                        newly_filled = max(0, fc - pos.get("resting_tp_filled", 0))
                except Exception:
                    pass
                # Cancel + verify, up to 3 attempts. We only set confirmed when
                # POSITIVELY observed gone (DELETE 404, status canceled/executed,
                # or remaining==0). An empty/unknown/"resting" status is NEVER
                # treated as confirmed â€” that would risk a double-sell.
                for _attempt in range(3):
                    try:
                        res = await _kdelete(f"/portfolio/events/orders/{oid}")
                        if isinstance(res, dict) and res.get("_not_found"):
                            confirmed = True   # already gone (filled/cancelled)
                    except Exception as e:
                        log.warning(f"  [RESTING-TP] cancel attempt {_attempt+1} "
                                    f"failed ({reason}): {e!r}")
                    # Verify it is no longer resting.
                    try:
                        st2 = await _korder_status(oid)
                        if st2 is not None:
                            stt = (st2.get("status") or "").lower()
                            fc2 = _order_fill_count(st2)
                            newly_filled = max(newly_filled,
                                               fc2 - pos.get("resting_tp_filled", 0))
                            if (stt in _ORDER_STATUS_DONE
                                    or _order_remaining_count(st2) == 0):
                                confirmed = True
                    except Exception:
                        pass
                    if confirmed:
                        break
                    await asyncio.sleep(0.5)
                if newly_filled > 0:
                    _reconcile_resting_fill(newly_filled)
                if confirmed:
                    pos["resting_tp_id"] = None
                    log.info(f"  [RESTING-TP] cancelled id={oid[:8]} ({reason}); "
                             f"reconciled {newly_filled}c fill")
                else:
                    log.error(f"  [RESTING-TP] COULD NOT CONFIRM cancel id="
                              f"{oid[:8]} ({reason}) â€” holding market sell to "
                              f"avoid double-sell; will retry next poll")
                return newly_filled, confirmed

            def _reconcile_resting_fill(filled_c):
                """Account a resting-TP fill: record PnL on the filled portion,
                reduce the position. filled_c is NEWLY filled contracts."""
                if filled_c <= 0:
                    return
                tp_price = pos.get("resting_tp_price", entry_cents + take_profit_cents)
                _cost_basis = pos.get("resting_tp_entry", entry_cents)
                p_d = tp_price / 100.0
                fee = MAKER_FEE_RATE * p_d * (1 - p_d)
                gain_per = (tp_price - _cost_basis) / 100.0 - fee
                realized = gain_per * filled_c
                session.record(realized)
                pos["resting_tp_filled"] = pos.get("resting_tp_filled", 0) + filled_c
                old_n = pos["contracts"]
                ratio = max(0, (old_n - filled_c)) / old_n if old_n else 0
                pos["contracts"] = max(0, old_n - filled_c)
                pos["cost"]    = round(pos["cost"] * ratio, 2)
                pos["net_win"] = round(pos["net_win"] * ratio, 2)
                log.warning(
                    f"  [RESTING-TP] FILLED {filled_c}c @ {tp_price}\u00a2 "
                    f"(+{take_profit_cents}\u00a2/c) realized ${realized:+.2f}; "
                    f"{pos['contracts']}c remaining")
                audit.write({
                    "type": "RESTING_TP_FILL", "window_id": window_id,
                    "ticker": pos["ticker"], "side": pos["side"],
                    "filled_contracts": filled_c, "tp_price_cents": tp_price,
                    "realized_pnl": round(realized, 4),
                    "remaining_contracts": pos["contracts"],
                })

            # 2026-06-25: HOLD-TO-WIN entry-arm [HOLDWIN_ENTRY_ARM]. When the
            # entry Markov gap AND live cushion clear the bar on a golden/
            # standard trade, skip the resting TP and ride to settlement for the
            # full (100-entry)c. Backtest: gap>=0.30 reversed 1/21 (+$754 vs TP).
            # The trail-rearm (below) + $-stop/RRM are the reversal backstop.
            _hw_entry_arm_enabled = True
            _hw_armed_at_entry = False
            if (_hw_entry_arm_enabled and holdwin_enabled and resting_tp_enabled
                    and position_tier in holdwin_tiers):
                _hw_eg = pos.get("entry_gap")
                _hw_eed = pos.get("entry_dist_pct")
                if _hw_eg is not None and _hw_eed is not None:
                    _hw_ecush = _hw_eed if pos.get("side") == "YES" else -_hw_eed
                    if abs(_hw_eg) >= holdwin_min_gap and _hw_ecush >= holdwin_min_dist_pct:
                        _hw_armed_at_entry = True
                        pos["_holdwin_active"] = True
                        pos["_holdwin_peak"] = None
                        log.warning(
                            f"  [HOLD-WIN] {pos['ticker']} {pos['side']} "
                            f"{pos['contracts']}c: armed AT ENTRY (gap="
                            f"{abs(_hw_eg):.2f}>={holdwin_min_gap}, cushion="
                            f"{_hw_ecush:+.3f}%>={holdwin_min_dist_pct}%) â€” skip "
                            f"resting TP, ride to settle w/ trail backstop.")
                        audit.write({
                            "type": "HOLDWIN_ARM_ENTRY", "window_id": window_id,
                            "ticker": pos["ticker"], "side": pos["side"],
                            "contracts": pos["contracts"], "entry_cents": entry_cents,
                            "entry_gap": round(abs(_hw_eg), 4),
                            "cushion_pct": round(_hw_ecush, 4)})
            if not _hw_armed_at_entry:
                await _place_resting_tp()

            sl_exited = False
            while True:
                now = datetime.now(timezone.utc)
                mins_remaining = (close_dt - now).total_seconds() / 60.0
                mins_since_entry = (now - entry_ts).total_seconds() / 60.0

                # In the final stretch, give up monitoring and let it settle.
                # Cancel any resting TP first so it can't fill at a bad late
                # price or be orphaned past settlement.
                # 2026-06-18: predict-cross-exit keep-alive. Normally the SL
                # monitor disarms at sl_disable_late_mins. When predict-cross-
                # exit is enabled we KEEP this proven sell path alive down to
                # pcross_keep_alive_mins so a confirmed drift reversal can still
                # be sold in the final 2 min â€” but we still cancel the resting
                # TP at the original late threshold, and the cents-based SL
                # stays gated to its original disarm (below), so nothing else
                # about late behavior changes.
                _disarm_floor = (pcross_keep_alive_mins
                                 if predict_cross_exit_enabled
                                 else sl_disable_late_mins)
                if (predict_cross_exit_enabled
                        and pos.get("resting_tp_id")
                        and mins_remaining < sl_disable_late_mins):
                    await _cancel_resting_tp("late-disable (pcross keep-alive)")
                if mins_remaining < _disarm_floor:
                    # 2026-06-27: a high-risk tight TP is a PROFIT price (entry+K),
                    # never a bad late fill â€” keep it resting through the close so
                    # it can still fill on a recovery bounce. Otherwise cancel the
                    # normal resting TP as before.
                    if pos.get("highrisk_armed") and pos.get("resting_tp_id"):
                        log.info(
                            f"  SL-monitor: {mins_remaining:.1f}min left â€” "
                            f"[HIGHRISK-TP] leaving tight sell @ "
                            f"{pos.get('highrisk_target')}\u00a2 resting to close.")
                    elif pos.get("resting_tp_id"):
                        await _cancel_resting_tp("late-disable")
                    # 2026-06-30: RRM SHADOW TAIL (log-only telemetry). Instead of
                    # going dark in the final stretch, keep sampling RRM in shadow
                    # so late-window reversals are finally measurable. This NEVER
                    # sells: the shadow `continue` after the RRM block skips every
                    # live-exit path (pcross / cents-SL / $-stop / TP / holdwin).
                    # Break out only once essentially settled (< 0.1min).
                    if (rrm_state.get("anchor_btc") and rrm_state.get("anchor_perp")
                            and rrm_state.get("strike") and mins_remaining > 0.1):
                        if not pos.get("_rrm_shadow"):
                            log.info(
                                f"  SL-monitor: {mins_remaining:.1f}min left "
                                f"(< {_disarm_floor:.1f}min) â€” entering RRM SHADOW "
                                f"tail (log-only, no sells) until settle.")
                            pos["_rrm_shadow"] = True
                    else:
                        log.info(
                            f"  SL-monitor: {mins_remaining:.1f}min left "
                            f"(< {_disarm_floor:.1f}min disarm threshold) â€” "
                            f"letting position settle naturally"
                        )
                        break

                # Re-check position still in pending (could be removed by
                # settlement check between polls)
                if window_id not in session.pending:
                    if pos.get("resting_tp_id"):
                        await _cancel_resting_tp("no-longer-pending")
                    log.info(f"  SL-monitor: position no longer in pending â€” done")
                    break

                pos_ticker = pos["ticker"]
                pos_side   = pos["side"]
                pos_count  = pos["contracts"]

                # â”€â”€ 2026-06-15: monitor the RESTING TP order's fills â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # Poll its status; reconcile any new fills into the position.
                # If it fully filled, the position is closed â€” done.
                if pos.get("resting_tp_id"):
                    try:
                        _st = await _korder_status(pos["resting_tp_id"])
                        if _st:
                            _fc = _order_fill_count(_st)
                            _new = max(0, _fc - pos.get("resting_tp_filled", 0))
                            if _new > 0:
                                _reconcile_resting_fill(_new)
                            _ostatus = (_st.get("status") or "").lower()
                            _rem = _order_remaining_count(_st)
                            # Full close iff: position drained, OR order
                            # status==executed, OR remaining_count_fp == 0.
                            if (pos["contracts"] <= 0
                                    or _ostatus == "executed"
                                    or _rem == 0):
                                log.warning(f"  [RESTING-TP] fully filled â€” position closed.")
                                pos["resting_tp_id"] = None
                                if window_id in session.pending:
                                    del session.pending[window_id]
                                sl_exited = True
                                # â”€â”€ 2026-06-16: RESTING-TP RE-ENTRY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                                # The whole position just sold out in profit via
                                # the resting TP. If re-entry is enabled AND the
                                # window still has trading time left, re-open it
                                # for a fresh signal evaluation: discard it from
                                # session.traded so the main loop re-runs the
                                # FULL gate stack and may re-enter (re-buy the
                                # same side, or flip) ONLY IF signals still
                                # align. This branch is the sole re-entry path â€”
                                # it fires exclusively on a complete sell, never
                                # on partial fills, SL/RRM exits, flips, or
                                # settlement. Re-entry is naturally repeatable:
                                # each fresh entry re-arms its own resting TP,
                                # whose full sell re-opens the window again.
                                if tp_reentry_enabled:
                                    _now = datetime.now(timezone.utc)
                                    _mins_left_reentry = (
                                        close_dt - _now).total_seconds() / 60.0
                                    if _mins_left_reentry >= 2.5:
                                        session.traded.discard(window_id)
                                        # Reset the per-window no-fill counter so
                                        # the re-entry gets a clean fill-attempt
                                        # budget rather than inheriting the prior
                                        # leg's count.
                                        window_no_fills.pop(window_id, None)
                                        log.warning(
                                            f"  [TP-REENTRY] full sell complete "
                                            f"with {_mins_left_reentry:.1f}min "
                                            f"left â€” re-opening window {window_id} "
                                            f"for fresh signal evaluation "
                                            f"(re-entry IF signals align).")
                                        audit.write({
                                            "type": "TP_REENTRY_REOPEN",
                                            "window_id": window_id,
                                            "ticker": pos_ticker,
                                            "prior_side": pos_side,
                                            "mins_left": round(
                                                _mins_left_reentry, 2),
                                            "ts": _now.isoformat(),
                                            "ts_ms": int(time.time() * 1000),
                                        })
                                    else:
                                        log.info(
                                            f"  [TP-REENTRY] full sell complete "
                                            f"but only {_mins_left_reentry:.1f}min "
                                            f"left (< 2.5min) â€” not re-opening; "
                                            f"letting window close.")
                                break
                    except Exception as _re:
                        log.debug(f"  [RESTING-TP] status poll failed: {_re!r}")
                    pos_count = pos["contracts"]   # refresh after possible fill

                # Fetch current quote for this ticker
                try:
                    mkt_data = await _kget(f"/markets/{pos_ticker}")
                    fresh_mkt = mkt_data.get("market", mkt_data)
                    _normalize_market(fresh_mkt)
                except Exception as e:
                    log.debug(f"  SL-monitor: fetch {pos_ticker} failed: {e}")
                    await asyncio.sleep(effective_poll_interval)
                    continue

                yes_bid = int(fresh_mkt.get("yes_bid") or 0)
                no_bid  = int(fresh_mkt.get("no_bid")  or 0)
                yes_ask = int(fresh_mkt.get("yes_ask") or 0)
                no_ask  = int(fresh_mkt.get("no_ask")  or 0)
                last_price = int(fresh_mkt.get("last_price") or 0)
                current_sell_price = yes_bid if pos_side == "YES" else no_bid
                if current_sell_price <= 0:
                    await asyncio.sleep(effective_poll_interval)
                    continue

                # 2026-05-31: track bid trajectory for retrospective SL tuning.
                # Updates each poll; written to exit audit record.
                pos["n_sl_polls"] = pos.get("n_sl_polls", 0) + 1
                if (pos.get("min_bid_cents") is None
                        or current_sell_price < pos["min_bid_cents"]):
                    pos["min_bid_cents"] = current_sell_price
                    pos["max_drawdown_cents"] = entry_cents - current_sell_price
                    pos["min_bid_secs_from_entry"] = int(mins_since_entry * 60)
                if (pos.get("max_bid_cents") is None
                        or current_sell_price > pos["max_bid_cents"]):
                    pos["max_bid_cents"] = current_sell_price

                # â”€â”€ 2026-06-27: arm the conditional high-risk tight TP once the
                # drawdown crosses the threshold on a close-in high_conv trade.
                # 2026-06-30: never arm/place orders during the log-only shadow tail.
                if not pos.get("_rrm_shadow"):
                    await _arm_highrisk_tp()

                # â”€â”€ 2026-06-11: RRM evaluation (log-only OR live exit) â”€â”€â”€â”€â”€â”€â”€
                # 2026-06-12: live-exit mode added (--rrm-exit-enabled). When
                # the confluence score fires, set rrm_exit_triggered and
                # piggyback on the proven SL sell path (same as fast-exit).
                rrm_exit_triggered = False
                _shadow = bool(pos.get("_rrm_shadow"))
                if ((not rrm_state["fired"] or _shadow) and rrm_state["anchor_btc"]
                        and rrm_state["anchor_perp"] and rrm_state["strike"]
                        and mins_since_entry >= 1.0):
                    try:
                        rrm = rrm_evaluate(
                            side=pos_side,
                            strike=rrm_state["strike"],
                            btc_entry=rrm_state["anchor_btc"],
                            perp_mid_entry=rrm_state["anchor_perp"],
                            mins_remaining=mins_remaining,
                        )
                        if rrm["ok"]:
                            rrm_state["max_score"] = max(rrm_state["max_score"],
                                                         rrm["score"])
                            if rrm["breach"] and not rrm_state["breach_logged"]:
                                rrm_state["breach_logged"] = True
                                log.warning(
                                    f"  [RRM] STRIKE BREACH [{pos_ticker}] "
                                    f"{pos_side}: {rrm['summary']} "
                                    f"(bid={current_sell_price}\u00a2, "
                                    f"{mins_remaining:.1f}min left)")
                            if rrm["would_exit"]:
                                if not _shadow:
                                    rrm_state["fired"] = True
                                would_pnl = ((current_sell_price - entry_cents)
                                             / 100.0 * pos_count)
                                # 2026-06-22: cushion-gate the LIVE exit. RRM
                                # is net-additive only on thin-cushion entries;
                                # on wide-cushion winners it clips upside.
                                _rrm_cush_ok = (RRM_EXIT_CUSHION_MAX <= 0 or
                                                abs(pos.get("entry_dist_pct") or 0)
                                                < RRM_EXIT_CUSHION_MAX)
                                # Always write the validation record (the
                                # dataset keeps growing whether live or not).
                                # 2026-06-30: telemetry enrichment â€” log the
                                # features needed to calibrate RRM gating
                                # (tier / cushion / SL / loss / $-stop) and flag
                                # shadow-tail samples. _shadow is ALWAYS log-only
                                # (the shadow tail never reaches a sell path), so
                                # force live_exit False whenever _shadow is set.
                                _rrm_live = bool(
                                    rrm_exit_enabled
                                    and not _shadow
                                    and rrm["score"] >= rrm_exit_min_score
                                    and pos_count >= rrm_exit_min_contracts
                                    and _rrm_cush_ok)
                                audit.write({
                                    "type": "RRM_WOULD_EXIT",
                                    "window_id": window_id,
                                    "ticker": pos_ticker,
                                    "side": pos_side,
                                    "contracts": pos_count,
                                    "entry_cents": entry_cents,
                                    "would_exit_bid_cents": current_sell_price,
                                    "would_realize_pnl": round(would_pnl, 2),
                                    "score": rrm["score"],
                                    "components": rrm["components"],
                                    "est_spot": rrm["est_spot"],
                                    "strike": rrm_state["strike"],
                                    "mins_remaining": round(mins_remaining, 2),
                                    "mins_since_entry": round(mins_since_entry, 2),
                                    "tier": pos.get("tier"),
                                    "entry_dist_pct": pos.get("entry_dist_pct"),
                                    "effective_sl_cents": effective_sl_cents,
                                    "loss_cents": entry_cents - current_sell_price,
                                    "dollar_stop_eligible": bool(
                                        (position_tier in ("golden", "standard")
                                         and max_loss_per_trade > 0)
                                        or (position_tier == "high_conv"
                                            and max_loss_per_trade_high_conv > 0)),
                                    "shadow_tail": _shadow,
                                    "live_exit": _rrm_live,
                                })
                                if _rrm_live:
                                    rrm_exit_triggered = True
                                    log.warning(
                                        f"  [RRM EXIT] [{pos_ticker}] {pos_side} "
                                        f"{pos_count}c: {rrm['summary']} â€” "
                                        f"reversal confirmed, selling at bid "
                                        f"{current_sell_price}\u00a2 (entry "
                                        f"{entry_cents}\u00a2, est. realize "
                                        f"${would_pnl:+.2f} vs ride-to-zero risk).")
                                else:
                                    log.warning(
                                        f"  [RRM log-only] WOULD-EXIT [{pos_ticker}] "
                                        f"{pos_side} {pos_count}c: {rrm['summary']} â€” "
                                        f"would sell at {current_sell_price}\u00a2 "
                                        f"(entry {entry_cents}\u00a2, would-realize "
                                        f"${would_pnl:+.2f}). NOT SELLING "
                                        f"({'below min size' if rrm_exit_enabled else 'log-only mode'}).")
                    except Exception as _rrm_e:
                        log.debug(f"  RRM eval failed (non-fatal): {_rrm_e!r}")

                # 2026-06-30: shadow tail is strictly log-only â€” skip EVERY live
                # exit path (pcross / cents-SL / $-stop / TP / holdwin) and just
                # re-poll for the next RRM telemetry sample until settle.
                if pos.get("_rrm_shadow"):
                    await asyncio.sleep(effective_poll_interval)
                    continue

                # â”€â”€ 2026-06-18: PREDICT-CROSS-EXIT (drift-aware) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # In the final pcross_max_mins, estimate P(contract ends OTM)
                # from the perp oracle: distance to strike + projected 30s
                # adverse drift over the remaining time, scaled by realized
                # perp vol. When P(OTM) >= pcross_prob for pcross_confirm_polls
                # consecutive polls, sell pre-emptively via the proven SL path
                # (rides hit_stop below). Backtest (bid<60c proxy, 24d): a
                # 2-min gate caught ~33 late reversals (+$1.8-2.1k) while
                # touching only ~3 winners. One fire per position; min-contracts
                # guard keeps small positions log-only.
                pcross_triggered = False
                if (predict_cross_exit_enabled
                        and not pos.get("_pcross_fired")
                        and mins_since_entry >= sl_grace_mins
                        and mins_remaining <= pcross_max_mins
                        and rrm_state["anchor_btc"] and rrm_state["anchor_perp"]
                        and rrm_state["strike"]):
                    try:
                        pcm = rrm_evaluate(
                            side=pos_side,
                            strike=rrm_state["strike"],
                            btc_entry=rrm_state["anchor_btc"],
                            perp_mid_entry=rrm_state["anchor_perp"],
                            mins_remaining=mins_remaining,
                        )
                        _potm = pcm.get("p_otm") if (pcm and pcm.get("ok")) else None
                        if _potm is not None and _potm >= pcross_prob:
                            pos["_pcross_polls"] = pos.get("_pcross_polls", 0) + 1
                            if pos["_pcross_polls"] >= pcross_confirm_polls:
                                pos["_pcross_fired"] = True
                                _big = pos_count >= pcross_min_contracts
                                would_pnl = ((current_sell_price - entry_cents)
                                             / 100.0 * pos_count)
                                _cgc_exit = _build_cv_composite_exit(
                                    pos_side, pcm.get("est_spot"),
                                    rrm_state["strike"], mins_remaining)
                                audit.write({
                                    "type": "PCROSS_EXIT",
                                    "cv_composite_exit": _cgc_exit,
                                    "window_id": window_id,
                                    "ticker": pos_ticker,
                                    "side": pos_side,
                                    "contracts": pos_count,
                                    "entry_cents": entry_cents,
                                    "exit_bid_cents": current_sell_price,
                                    "would_realize_pnl": round(would_pnl, 2),
                                    "p_otm": _potm,
                                    "dist_pct": pcm.get("dist_pct"),
                                    "mean_dist_pct": pcm.get("mean_dist_pct"),
                                    "sigma_rem": pcm.get("sigma_rem"),
                                    "est_spot": pcm.get("est_spot"),
                                    "strike": rrm_state["strike"],
                                    "mins_remaining": round(mins_remaining, 2),
                                    "mins_since_entry": round(mins_since_entry, 2),
                                    "confirm_polls": pos["_pcross_polls"],
                                    "live_exit": bool(_big),
                                })
                                if _big:
                                    pcross_triggered = True
                                    log.warning(
                                        f"  [PCROSS EXIT] [{pos_ticker}] {pos_side} "
                                        f"{pos_count}c: P(OTM)={_potm:.2f} \u2265"
                                        f"{pcross_prob:.2f} ({pos['_pcross_polls']} "
                                        f"polls, {mins_remaining:.1f}min left) \u2014 "
                                        f"selling at bid {current_sell_price}\u00a2 "
                                        f"(entry {entry_cents}\u00a2, est. realize "
                                        f"${would_pnl:+.2f}).")
                                else:
                                    log.warning(
                                        f"  [PCROSS log-only] WOULD-EXIT "
                                        f"[{pos_ticker}] {pos_side} {pos_count}c: "
                                        f"P(OTM)={_potm:.2f} (bid {current_sell_price}"
                                        f"\u00a2) \u2014 below min size "
                                        f"{pcross_min_contracts}c, not selling.")
                        elif pos.get("_pcross_polls"):
                            pos["_pcross_polls"] = 0   # reset the confirm streak
                    except Exception as _pce:
                        log.debug(f"  PCROSS eval failed (non-fatal): {_pce!r}")

                # â”€â”€ 2026-06-24: HOLD-TO-WIN (cancel resting TP on conviction) â”€
                # On golden/standard winners that are clearly working â€” already
                # in profit, wide live cushion, low modeled P(end-OTM) â€” cancel
                # the +Nc resting TP and ride to settlement for the full
                # (100-entry)c. Keep MONITORING: re-arm the resting TP (which
                # fills instantly while the bid is still above entry+Nc, locking
                # the gain) the moment conviction deteriorates â€” cushion
                # collapse, P(OTM) spike, or a trailing retrace from the peak
                # bid. The $-stop / pcross / RRM exits below stay fully active as
                # the reversal backstop, so a held position that truly reverses
                # is still flattened (capped by --max-loss-per-trade).
                if (holdwin_enabled and resting_tp_enabled
                        and position_tier in holdwin_tiers
                        and pos_count > 0
                        and mins_since_entry >= min(0.25, sl_grace_mins)):
                    _hw_profit = current_sell_price - entry_cents
                    _hw_dist = None
                    _hw_potm = None
                    if (rrm_state.get("anchor_btc") and rrm_state.get("anchor_perp")
                            and rrm_state.get("strike")):
                        try:
                            _hw = rrm_evaluate(
                                side=pos_side, strike=rrm_state["strike"],
                                btc_entry=rrm_state["anchor_btc"],
                                perp_mid_entry=rrm_state["anchor_perp"],
                                mins_remaining=mins_remaining)
                            if _hw and _hw.get("ok"):
                                _hw_dist = abs(_hw.get("dist_pct") or 0) * 100  # 2026-06-29 FIX: rrm dist_pct is FRACTION; holdwin thresholds are PERCENT -> *100 (else TP re-armed every poll, defeating hold-to-win).
                                _hw_potm = _hw.get("p_otm")
                        except Exception:
                            pass
                    if not pos.get("_holdwin_active"):
                        _sure = (pos.get("resting_tp_id")
                                 and abs(pos.get("entry_gap") or 0) >= holdwin_min_gap
                                 and _hw_profit >= holdwin_min_profit_cents
                                 and (_hw_dist is None or _hw_dist >= holdwin_min_dist_pct)
                                 and (_hw_potm is None or _hw_potm <= holdwin_max_potm))
                        if _sure:
                            _rf, _conf = await _cancel_resting_tp("holdwin-cancel")
                            if _conf:
                                pos["_holdwin_active"] = True
                                pos["_holdwin_peak"] = current_sell_price
                                pos_count = pos["contracts"]
                                log.warning(
                                    f"  [HOLD-WIN] {pos_ticker} {pos_side} "
                                    f"{pos_count}c: sure-win (profit "
                                    f"+{_hw_profit}Â¢, dist={_hw_dist}, "
                                    f"P(OTM)={_hw_potm}) â€” TP cancelled, riding "
                                    f"to settle for full (100-{entry_cents})Â¢.")
                                audit.write({
                                    "type": "HOLDWIN_CANCEL_TP",
                                    "window_id": window_id, "ticker": pos_ticker,
                                    "side": pos_side, "contracts": pos_count,
                                    "entry_cents": entry_cents,
                                    "bid_cents": current_sell_price,
                                    "profit_cents": _hw_profit,
                                    "dist_pct": _hw_dist, "p_otm": _hw_potm,
                                    "mins_remaining": round(mins_remaining, 2)})
                    else:
                        if current_sell_price > (pos.get("_holdwin_peak") or 0):
                            pos["_holdwin_peak"] = current_sell_price
                        _retrace = ((pos.get("_holdwin_peak") or current_sell_price)
                                    - current_sell_price)
                        _deteriorate = (
                            (_hw_dist is not None and _hw_dist < holdwin_rearm_dist_pct)
                            or (_hw_potm is not None and _hw_potm >= holdwin_rearm_potm)
                            or (_retrace >= holdwin_trail_cents))
                        if _deteriorate and not pos.get("resting_tp_id"):
                            pos["_holdwin_active"] = False
                            await _place_resting_tp()
                            log.warning(
                                f"  [HOLD-WIN] {pos_ticker} re-arm TP: "
                                f"dist={_hw_dist} P(OTM)={_hw_potm} "
                                f"retrace={_retrace}Â¢ (peak "
                                f"{pos.get('_holdwin_peak')}Â¢ bid "
                                f"{current_sell_price}Â¢).")
                            audit.write({
                                "type": "HOLDWIN_REARM_TP",
                                "window_id": window_id, "ticker": pos_ticker,
                                "side": pos_side, "contracts": pos_count,
                                "bid_cents": current_sell_price,
                                "peak_cents": pos.get("_holdwin_peak"),
                                "retrace_cents": _retrace, "dist_pct": _hw_dist,
                                "p_otm": _hw_potm,
                                "mins_remaining": round(mins_remaining, 2)})

                # Choose trigger price by mode
                if sl_trigger_mode == "mid":
                    if pos_side == "YES" and yes_ask > 0:
                        trigger_price = (yes_bid + yes_ask) // 2
                    elif pos_side == "NO" and no_ask > 0:
                        trigger_price = (no_bid + no_ask) // 2
                    else:
                        trigger_price = current_sell_price
                elif sl_trigger_mode == "last":
                    if last_price > 0:
                        trigger_price = (last_price if pos_side == "YES"
                                         else 100 - last_price)
                    else:
                        trigger_price = current_sell_price
                else:  # "bid"
                    trigger_price = current_sell_price

                loss_cents = entry_cents - trigger_price
                # 2026-06-18: the cents-based SL keeps its ORIGINAL late disarm
                # even when the predict-cross-exit keep-alive holds this loop
                # open past it (no-op when pcross is off, since the loop already
                # breaks at sl_disable_late_mins then).
                hit_stop = (mins_since_entry >= sl_grace_mins
                            and loss_cents >= effective_sl_cents
                            and mins_remaining >= sl_disable_late_mins)
                # 2026-06-12: RRM live exit rides the SL sell path
                if rrm_exit_triggered:
                    hit_stop = True
                # 2026-06-18: predict-cross-exit rides the SL sell path
                if pcross_triggered:
                    hit_stop = True
                # â”€â”€ 2026-06-22: PER-TRADE DOLLAR STOP (golden/standard tail cap) â”€â”€
                # Hard $ loss cap on the cheap tiers. Backtest (06-22): a ~$100
                # per-trade stop flips golden+standard from net-negative to
                # strongly positive across regimes by truncating the fat left
                # tail (bankroll-scaled reversals). Scoped to golden/standard
                # only â€” high_conv/late_sure keep their tight per-contract stop
                # and are untouched. Bypasses grace/late-disarm: a $100 drawdown
                # this size is a real adverse move, not entry noise, so we
                # flatten immediately via the existing SL sell path.
                # 2026-07-01: per-tier dollar cap. golden/standard use
                # max_loss_per_trade; high_conv uses its own (wider) cap so the
                # tighter golden/standard number never clips high_conv's deep
                # winner dips. Same fire path; runs every poll until the
                # disarm floor (pcross_keep_alive_mins), so it catches all but
                # the last-~24s cliff drops.
                if position_tier == "high_conv":
                    _ds_cap = max_loss_per_trade_high_conv
                elif position_tier in ("golden", "standard"):
                    _ds_cap = max_loss_per_trade
                else:
                    _ds_cap = 0.0
                dollar_stop_hit = False
                if (_ds_cap > 0 and pos_count > 0
                        and loss_cents > 0
                        and (loss_cents * pos_count / 100.0) >= _ds_cap):
                    hit_stop = True
                    dollar_stop_hit = True
                    log.warning(
                        f"  $-STOP [{pos_ticker}] tier={position_tier}: down "
                        f"-${loss_cents * pos_count / 100.0:.2f} "
                        f"(-{loss_cents}Â¢/c x {pos_count}c) â‰¥ cap "
                        f"${_ds_cap:.0f} â€” flattening position."
                    )

                # â”€â”€ 2026-06-15: PERP-CONFIRMED TAKE-PROFIT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # Resting TP at entry+take_profit_cents. Backtest: on perp-
                # momentum-confirmed trades (entry m30s>0) a +10c TP was net
                # +$221 (+$520 on strong perp) â€” it captures the early profit
                # perp momentum produces before late-window reversals. On
                # NON-confirmed trades a TP is strongly -EV, so it is GATED to
                # entry_perp_m30s > threshold. Exits via the SL sell path.
                tp_triggered = False
                if (not hit_stop and take_profit_enabled
                        and not resting_tp_enabled   # resting order handles TP
                        and mins_since_entry >= sl_grace_mins):
                    _ep = pos.get("entry_perp_m30s")
                    _tp_ok = (not take_profit_perp_confirmed_only
                              or (_ep is not None and _ep > take_profit_perp_min))
                    # 2026-06-15: TP COMMIT. Once the +TP target is hit, we
                    # commit to exiting the WHOLE position in profit. On a
                    # partial fill, subsequent polls keep selling the remainder
                    # aggressively at ANY profitable price (bid > entry), even
                    # if the bid has receded below +TP â€” so the remainder can't
                    # ride back into a loss. We stop only if the bid falls to
                    # entry or below (no longer profitable); then the remainder
                    # holds for SL/settlement rather than selling at a loss.
                    if pos.get("tp_committed"):
                        if current_sell_price > entry_cents:
                            tp_triggered = True
                            hit_stop = True
                            log.warning(
                                f"  TAKE-PROFIT (commit) [{pos_ticker}] {pos_side}: "
                                f"selling remainder at bid {current_sell_price}\u00a2 "
                                f"(+{current_sell_price - entry_cents}\u00a2/c, still "
                                f"profitable) â€” locking whole position out of profit.")
                        # else: bid <= entry, no longer profitable -> stop TP-
                        # selling the remainder; it holds for SL/settlement.
                    elif _tp_ok and current_sell_price >= entry_cents + take_profit_cents:
                        tp_triggered = True
                        hit_stop = True
                        pos["tp_committed"] = True
                        log.warning(
                            f"  TAKE-PROFIT [{pos_ticker}] {pos_side}: bid "
                            f"{current_sell_price}\u00a2 \u2265 entry {entry_cents}\u00a2 "
                            f"+{take_profit_cents}\u00a2 (entry perp m30s="
                            f"{_ep if _ep is not None else 'n/a'}). Locking "
                            f"+{current_sell_price - entry_cents}\u00a2/c before reversal.")

                # â”€â”€ 2026-05-30 NEW: FUTURES FAST-EXIT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # Lead the SL: if the lead-venue futures has moved sharply
                # against the position over the configured window, exit
                # NOW rather than waiting for Polymarket's bid to follow.
                # Skipped during grace period (avoid panic-exiting on
                # initial chop) and when SL is already triggered.
                fast_exit_triggered = False
                fast_exit_move_pct = 0.0
                if (not hit_stop and futures_fast_exit_enabled
                        and (futures_lead is not None or okx_feed is not None)
                        and mins_since_entry >= sl_grace_mins):
                    ft_move = None
                    ft_src = None
                    if futures_lead is not None:
                        try:
                            ft = futures_lead.get_recent_move(
                                lookback_s=futures_fast_exit_window_s)
                        except Exception:
                            ft = None
                        ft_move = (ft or {}).get("move_pct")
                        if ft_move is not None:
                            ft_src = "CME"
                    # 2026-06-27: OKX-as-primary fallback for fast-exit. When the primary venue
                    # is stale (Fri-Sun close) the held position would otherwise
                    # lose futures-led exit protection; use OKX BTC-USDT-SWAP
                    # (24/7) instead. Same window/threshold (a 30s % move is
                    # comparable across CME futures and OKX perp).
                    if ft_move is None and okx_feed is not None:
                        try:
                            ftx = okx_feed.get_recent_move(
                                lookback_s=futures_fast_exit_window_s)
                        except Exception:
                            ftx = None
                        ft_move = (ftx or {}).get("move_pct")
                        if ft_move is not None:
                            ft_src = "OKX(CME stale)"
                    if (ft_move is not None
                            and abs(ft_move) <= futures_fast_exit_sanity_max_pct):
                        # YES position adverse if futures drops (move < 0)
                        # NO  position adverse if futures rises (move > 0)
                        adverse_move = -ft_move if pos_side == "YES" else ft_move
                        if adverse_move >= futures_fast_exit_threshold_pct:
                            fast_exit_triggered = True
                            fast_exit_move_pct = ft_move
                            hit_stop = True  # piggyback on existing SL exit path
                            log.warning(
                                f"  FUTURES FAST-EXIT [{pos_ticker}] tier={position_tier}: "
                                f"{ft_src} futures moved {ft_move:+.4f}% over "
                                f"{futures_fast_exit_window_s:.0f}s (adverse "
                                f"{adverse_move:+.4f}% â‰¥ threshold "
                                f"{futures_fast_exit_threshold_pct:.3f}%). "
                                f"Exiting before SL triggers."
                            )

                # â”€â”€ 2026-06-03 NEW: FADE-BOUNCE dual-entry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # When the bid for our side dips into the 40-55Â¢ band mid-window,
                # 85% of windows still settle in the ORIGINAL direction (7-day
                # backtest). Buy more contracts at the discount price using
                # the cached entry signals as the qualifier.
                # Hard-capped by --fade-bounce-max-capital-usd (default $20).
                if (not hit_stop and fade_bounce_enabled
                        and window_id not in session.fade_bounce_traded
                        and mins_since_entry >= 1.0
                        and mins_remaining >= fade_bounce_min_mins
                        and mins_remaining <= fade_bounce_max_mins):
                    # Current ASK for our side (what we'd pay)
                    fb_entry_price = no_ask if pos_side == "NO" else yes_ask
                    in_band = (fade_bounce_no_ask_min <= fb_entry_price
                               <= fade_bounce_no_ask_max
                               and fb_entry_price > 0)
                    if in_band:
                        # Cached entry signals
                        _markov_p = pos.get("entry_markov_p_yes")
                        _hurst    = pos.get("entry_hurst")
                        _dist     = pos.get("entry_dist_pct")
                        fb_sig_ok = False
                        if (_markov_p is not None and _hurst is not None
                                and _dist is not None):
                            if pos_side == "NO":
                                fb_sig_ok = (_markov_p <= fade_bounce_markov_no_max
                                             and _hurst >= fade_bounce_hurst_min
                                             and _dist <= -fade_bounce_dist_min)
                            elif fade_bounce_yes_side_enabled:
                                fb_sig_ok = (_markov_p >= fade_bounce_markov_yes_min
                                             and _hurst >= fade_bounce_hurst_min
                                             and _dist >= fade_bounce_dist_min)
                        if fb_sig_ok:
                            # Size cap by capital budget + Kelly fraction
                            fb_entry_d = fb_entry_price / 100.0
                            cap_max    = int(fade_bounce_max_capital_usd / fb_entry_d)
                            kelly_max  = int(session.effective_bankroll()
                                             * fade_bounce_kelly_frac / fb_entry_d)
                            floor_size = int(session.effective_bankroll()
                                             * fade_bounce_min_stake_pct / fb_entry_d)
                            # 2026-06-15 BUG FIX: cap_max (--fade-bounce-max-
                            # capital-usd) must be a HARD ceiling. The old
                            # `max(floor_size, min(cap_max, kelly_max))` let the
                            # 1.5%-of-bankroll floor override the dollar cap â€”
                            # so the $20 cap never bound and fade-bounce sized to
                            # ~$173 at an $11.5k bankroll (caused a -$347 loss
                            # buying into a reversal). Cap is now the outer limit.
                            fb_contracts = min(cap_max, max(floor_size, kelly_max))
                            if fb_contracts >= 5:  # min meaningful size
                                log.info(
                                    f"  FADE-BOUNCE ATTACH [{pos_ticker}]: "
                                    f"buying {fb_contracts} more {pos_side} @ "
                                    f"{fb_entry_price}Â¢  "
                                    f"(primary entry {entry_cents}Â¢, "
                                    f"savings {entry_cents - fb_entry_price}Â¢/c, "
                                    f"capital ${fb_contracts * fb_entry_d:.2f})"
                                )
                                fb_order_id = None
                                fb_filled   = 0
                                fb_price_d  = (fb_entry_d if pos_side == "YES"
                                               else (1.0 - fb_entry_d))
                                if dry_run:
                                    fb_filled   = fb_contracts
                                    fb_order_id = "dry-run"
                                else:
                                    try:
                                        fb_body = {
                                            "ticker":                     pos_ticker,
                                            "client_order_id":            uuid.uuid4().hex,
                                            "side": "bid" if pos_side == "YES" else "ask",
                                            "count":                      str(fb_contracts),
                                            "price":                      f"{fb_price_d:.4f}",
                                            "time_in_force":              "immediate_or_cancel",
                                            "self_trade_prevention_type": "taker_at_cross",
                                        }
                                        fb_result = await _kpost(
                                            "/portfolio/events/orders", fb_body
                                        )
                                        try:
                                            fb_filled = int(float(
                                                fb_result.get("fill_count") or 0))
                                        except (TypeError, ValueError):
                                            fb_filled = 0
                                        fb_order_id = fb_result.get("order_id")
                                    except Exception as fb_e:
                                        log.warning(
                                            f"  FADE-BOUNCE order failed: {fb_e}"
                                        )
                                # Update position + audit on any fill
                                if fb_filled > 0:
                                    new_cost_cents = (
                                        pos["cost"] * 100 / pos["contracts"]
                                        * pos["contracts"]
                                        + fb_filled * fb_entry_price
                                    )
                                    pos["contracts"] += fb_filled
                                    pos["cost"]      = round(
                                        new_cost_cents / 100.0, 2
                                    )
                                    pos["fade_bounce"] = {
                                        "contracts":   fb_filled,
                                        "entry_cents": fb_entry_price,
                                        "ts":          now.isoformat(),
                                    }
                                    audit.write({
                                        "type": "FADE_BOUNCE_ATTACH" + (
                                            "_SIM" if dry_run else ""),
                                        "window_id": window_id,
                                        "ticker":    pos_ticker,
                                        "side":      pos_side,
                                        "primary_entry_cents":  entry_cents,
                                        "primary_contracts":    pos_count,
                                        "fade_bounce_entry_cents": fb_entry_price,
                                        "fade_bounce_contracts":   fb_filled,
                                        "fade_bounce_capital_usd": round(
                                            fb_filled * fb_entry_d, 2),
                                        "savings_per_c":      entry_cents - fb_entry_price,
                                        "mins_remaining":     round(mins_remaining, 2),
                                        "mins_since_primary": round(mins_since_entry, 2),
                                        "cached_markov_p_yes": _markov_p,
                                        "cached_hurst":        _hurst,
                                        "cached_dist_pct":     _dist,
                                        "order_id": fb_order_id,
                                        "ts":       now.isoformat(),
                                    })
                                session.fade_bounce_traded.add(window_id)

                # â”€â”€ 2026-06-12: PATIENT TOP-UP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # When the original intended size wasn't fully filled (walk
                # stopped at the EV floor, ask ran away), keep watching the
                # ask for the rest of the window. If it improves AND a fresh
                # full signal still approves this direction with positive EV
                # (the adverse-selection guard), buy up to the remaining size.
                if (patient_topup_enabled and not dry_run and not hit_stop
                        and pos.get("intended_contracts", 0) > pos["contracts"]
                        and mins_remaining > patient_topup_min_mins
                        and not (pos.get("hedge") or {}).get("active")
                        and pos.get("fade_bounce") is None):
                    _tu_ask = yes_ask if pos_side == "YES" else no_ask
                    _tu_ref = pos.get("topup_best_ask")
                    if _tu_ref is None:
                        _tu_ref = entry_cents   # arm: only try below blended entry
                        pos["topup_best_ask"] = _tu_ref
                    if (_tu_ask > 0 and _tu_ask < _tu_ref
                            and time.time() - pos.get("topup_last_try_ts", 0)
                                >= patient_topup_interval_s):
                        pos["topup_last_try_ts"] = time.time()
                        pos["topup_best_ask"] = _tu_ask
                        try:
                            tu_sig = await run_signal(
                                market, session.effective_bankroll(),
                                last_bar_adverse_threshold=last_bar_adverse_threshold,
                                tp_bs_p_above=(window_tp_cache.get(window_id) or {}).get("bs_p_above"),
                                last_bar_extreme_gap_min=0.30,
                                last_bar_extreme_tp=high_conv_tp_strong,
                                standard_price_cap_yes=standard_price_cap_yes,
                                standard_price_cap_no=standard_price_cap_no,
                                golden_price_lo=golden_price_lo,
                                golden_price_hi=golden_price_hi,
                                golden_no_dist=golden_no_dist,
                                golden_no_hurst=golden_no_hurst,
                            )
                            tu_ok = (tu_sig.get("approved")
                                     and not tu_sig.get("rejection_reasons")
                                     and tu_sig.get("recommendation") == pos_side
                                     and (tu_sig.get("expected_value") or 0) > 0)
                            # 2026-06-15: top-up must clear the SAME live entry
                            # vetoes (perp-momentum / book-skew / perp-imb) as a
                            # fresh entry â€” the top-up fires late in the window
                            # exactly when reversals form, so re-checking the
                            # live perp/book signals here is most valuable.
                            # Transient: a blocked poll just waits; if signals
                            # re-align on a later poll, the top-up proceeds.
                            if tu_ok:
                                _tns = snapshot_new_signals(pos_side)
                                _tu_block = None
                                if (perp_veto_enabled and _tns.get("perp_m30s") is not None
                                        and _tns["perp_m30s"] <= perp_veto_m30s_threshold):
                                    _tu_block = f"perp_m30s={_tns['perp_m30s']:+.2f}bp"
                                elif (perp_imb_veto_enabled and _tns.get("perp_imb") is not None
                                        and _tns["perp_imb"] <= perp_imb_veto_threshold):
                                    _tu_block = f"perp_imb={_tns['perp_imb']:+.3f}"
                                elif (book_skew_veto_enabled
                                        and (65 <= pos["limit_price"] <= 73 or not book_skew_golden_only)
                                        and _tns.get("book_skew") is not None
                                        and _tns["book_skew"] <= book_skew_threshold):
                                    _tu_block = f"book_skew={_tns['book_skew']:+.3f}"
                                if _tu_block is not None:
                                    log.info(
                                        f"  [TOP-UP] held off [{pos_ticker}] "
                                        f"{pos_side}: live veto ({_tu_block}) against "
                                        f"position â€” will retry if signals re-align.")
                                    tu_ok = False
                            if tu_ok:
                                # 2026-06-15: dynamic-Kelly target. The entry
                                # may have been Kelly-floored to a tiny size
                                # because EV was negative/marginal at the time.
                                # If the price has since improved and EV is now
                                # POSITIVE (tu_ok requires EV>0), recompute the
                                # fresh Kelly size and let the position grow
                                # toward it â€” not just toward the frozen
                                # entry-time intent. Only ever adds at +EV, so
                                # it can never size up a still-negative trade.
                                fresh_kelly = int(tu_sig.get("contracts") or 0)
                                if patient_topup_dynamic_kelly:
                                    target = max(pos["intended_contracts"], fresh_kelly)
                                else:
                                    target = pos["intended_contracts"]
                                remaining = target - pos["contracts"]
                                tu_n = min(remaining, fresh_kelly)
                                tu_price = int(tu_sig.get("limit_price") or _tu_ask)
                                if tu_n >= 5:
                                    tu_yes_leg = (tu_price / 100.0
                                                  if pos_side == "YES"
                                                  else 1.0 - tu_price / 100.0)
                                    log.info(
                                        f"  [TOP-UP] [{pos_ticker}] buying "
                                        f"{tu_n} more {pos_side} @ {tu_price}\u00a2 "
                                        f"(intended {pos['intended_contracts']}, "
                                        f"have {pos['contracts']}, fresh "
                                        f"EV=${tu_sig.get('expected_value'):+.2f})")
                                    tu_filled = 0
                                    tu_order_id = None
                                    try:
                                        tu_body = {
                                            "ticker":          pos_ticker,
                                            "client_order_id": uuid.uuid4().hex,
                                            "side": "bid" if pos_side == "YES" else "ask",
                                            "count":           str(tu_n),
                                            "price":           f"{tu_yes_leg:.4f}",
                                            "time_in_force":   "immediate_or_cancel",
                                            "self_trade_prevention_type": "taker_at_cross",
                                        }
                                        tu_res = await _kpost(
                                            "/portfolio/events/orders", tu_body)
                                        try:
                                            tu_filled = int(float(
                                                tu_res.get("fill_count") or 0))
                                        except (TypeError, ValueError):
                                            tu_filled = 0
                                        tu_order_id = tu_res.get("order_id")
                                    except Exception as tu_e:
                                        log.warning(f"  [TOP-UP] order failed: {tu_e}")
                                    if tu_filled > 0:
                                        p_d2  = tu_price / 100.0
                                        fee2  = MAKER_FEE_RATE * p_d2 * (1 - p_d2)
                                        cost2 = p_d2 + fee2
                                        net2  = (1 - p_d2) - fee2
                                        old_n = pos["contracts"]
                                        pos["cost"]    = round(pos["cost"] + cost2 * tu_filled, 2)
                                        pos["net_win"] = round(pos["net_win"] + net2 * tu_filled, 2)
                                        pos["entry_yes_leg"] = round(
                                            (pos["entry_yes_leg"] * old_n
                                             + tu_yes_leg * tu_filled)
                                            / (old_n + tu_filled), 4)
                                        pos["limit_price"] = round(
                                            (entry_cents * old_n + tu_price * tu_filled)
                                            / (old_n + tu_filled))
                                        pos["contracts"] = old_n + tu_filled
                                        pos["topup_filled_total"] = (
                                            pos.get("topup_filled_total", 0) + tu_filled)
                                        entry_cents = pos["limit_price"]   # blended for SL math
                                        log.info(
                                            f"  [TOP-UP] FILLED {tu_filled}/{tu_n} @ "
                                            f"{tu_price}\u00a2 â€” position now "
                                            f"{pos['contracts']}c, blended entry "
                                            f"{entry_cents}\u00a2")
                                        # 2026-06-15: position grew â€” cancel &
                                        # replace the resting TP at the new
                                        # blended entry+TP for the new total qty
                                        # (reconciles any TP fills first).
                                        if resting_tp_enabled and pos.get("resting_tp_id"):
                                            _rf2, _conf2 = await _cancel_resting_tp("topup-resize")
                                            # Only re-place once the old order is
                                            # confirmed gone, else we'd have two
                                            # resting sells (over-sell risk).
                                            if _conf2 and pos["contracts"] > 0:
                                                await _place_resting_tp()
                                        audit.write({
                                            "type": "TRADE_TOPUP",
                                            "window_id": window_id,
                                            "ticker": pos_ticker,
                                            "side": pos_side,
                                            "topup_contracts": tu_filled,
                                            "topup_price_cents": tu_price,
                                            "intended_contracts": pos["intended_contracts"],
                                            "position_contracts": pos["contracts"],
                                            "blended_entry_cents": entry_cents,
                                            "fresh_ev_usd": tu_sig.get("expected_value"),
                                            "mins_remaining": round(mins_remaining, 2),
                                            "order_id": tu_order_id,
                                        })
                        except Exception as _tu_e:
                            log.debug(f"  [TOP-UP] eval failed (non-fatal): {_tu_e!r}")

                if not hit_stop:
                    await asyncio.sleep(effective_poll_interval)
                    continue

                # â”€â”€ 2026-06-15: CANCEL RESTING TP BEFORE ANY MARKET SELL â”€â”€â”€â”€â”€
                # Critical double-sell guard: any other exit (SL/RRM/fast-exit)
                # must cancel the resting TP first and reconcile its fills, so
                # we never sell the same contracts twice. After cancel, pos
                # contracts reflect what's actually left to sell.
                if pos.get("resting_tp_id"):
                    _rf, _confirmed = await _cancel_resting_tp("superseded-by-exit")
                    pos_count = pos["contracts"]
                    if not _confirmed:
                        # Resting order may still be live â€” selling now risks a
                        # double-sell (net short). Skip this poll and retry.
                        await asyncio.sleep(effective_poll_interval)
                        continue
                    if pos_count <= 0:
                        # Resting TP had already filled the whole position.
                        if window_id in session.pending:
                            del session.pending[window_id]
                        sl_exited = True
                        break

                if not fast_exit_triggered and not rrm_exit_triggered and not tp_triggered:
                    log.warning(
                        f"  STOP-LOSS HIT [{pos_ticker}] tier={position_tier}: entry={entry_cents}Â¢, "
                        f"trigger({sl_trigger_mode})={trigger_price}Â¢, sell_bid={current_sell_price}Â¢, "
                        f"loss=-{loss_cents}Â¢/c (threshold={effective_sl_cents}Â¢) Ã— {pos_count}c "
                        f"= -${loss_cents * pos_count / 100:.2f}, "
                        f"mins_remaining={mins_remaining:.1f}, "
                        f"mins_since_entry={mins_since_entry:.1f}"
                    )

                # â”€â”€ Verify actual position with Polymarket BEFORE selling â”€â”€
                # Defensive: prevent over-sell that could create an
                # unintended short. If query fails, fall back to in-memory.
                actual_count = await get_venue_position_count(pos_ticker)
                if actual_count is None:
                    log.warning(
                        f"  SL [{pos_ticker}]: position query failed â€” "
                        f"falling back to in-memory count {pos_count}c"
                    )
                    sell_count = pos_count
                elif actual_count == 0:
                    log.warning(
                        f"  SL [{pos_ticker}]: Polymarket shows 0 contracts but pending "
                        f"shows {pos_count}c. Removing stale position; no SELL submitted."
                    )
                    if window_id in session.pending:
                        del session.pending[window_id]
                    break
                else:
                    sell_count = min(pos_count, actual_count)
                    if sell_count < pos_count:
                        log.warning(
                            f"  SL [{pos_ticker}]: in-memory says {pos_count}c but "
                            f"Polymarket shows {actual_count}c â€” selling smaller {sell_count}c"
                        )
                        pos["contracts"] = sell_count

                # â”€â”€ Submit SELL IOC â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # Aggressive sell mode (default ON for SL): instead of
                # asking for the observed bid (which is often gone by
                # the time our order arrives in fast-falling markets),
                # submit an IOC at the EXTREME limit ($0.01 ask for YES,
                # $0.99 bid for NO equivalent). Polymarket fills limit orders
                # at the BEST available price, so we get whatever bid
                # exists â€” guaranteed fill, no chasing the falling bid.
                #
                # Background: today's two SL events lost ~$22 of EXTRA
                # money to slippage. Bid fell 25-30Â¢ between trigger
                # observation and order fill while we re-tried at the
                # observed (moving) bid each time. Aggressive sell
                # eliminates this â€” single shot, fills at the best bid
                # available right now, doesn't retry through a crash.
                sell_side = "ask" if pos_side == "YES" else "bid"
                if sl_aggressive_sell:
                    # Aggressive: extreme limit guaranteeing fill
                    aggressive_yes_leg = 0.01 if pos_side == "YES" else 0.99
                    sell_yes_leg = aggressive_yes_leg
                    sell_mode_tag = "AGGRESSIVE"
                else:
                    # Conservative: observed bid (legacy behavior; can
                    # miss in fast markets)
                    sell_yes_leg = (current_sell_price / 100 if pos_side == "YES"
                                     else (1 - current_sell_price / 100))
                    sell_mode_tag = "observed-bid"
                try:
                    sell_body = {
                        "ticker":                     pos_ticker,
                        "client_order_id":            uuid.uuid4().hex,
                        "side":                       sell_side,
                        "count":                      str(sell_count),
                        "price":                      f"{sell_yes_leg:.4f}",
                        "time_in_force":              "immediate_or_cancel",
                        "self_trade_prevention_type": "taker_at_cross",
                    }
                    sell_result = await _kpost("/portfolio/events/orders", sell_body)
                    sell_filled = int(float(sell_result.get("fill_count") or 0))
                except Exception as e:
                    log.error(f"  SL-SELL [{pos_ticker}] error: {e}; continuing monitor")
                    await asyncio.sleep(effective_poll_interval)
                    continue

                if sell_filled <= 0:
                    log.warning(
                        f"  SL-SELL [{pos_ticker}]: 0 fills (mode={sell_mode_tag}, "
                        f"orderbook empty); retry next poll"
                    )
                    await asyncio.sleep(effective_poll_interval)
                    continue

                # Compute realized P&L using ACTUAL fill price returned by
                # Polymarket (not our limit). With aggressive mode the fill
                # will be at the best bid that existed, not at our 0.01/0.99
                # extreme.
                #
                # Safer fallback for aggressive mode: if Polymarket's response
                # doesn't include `average_fill_price` (rare but possible),
                # DON'T fall back to our extreme limit ($0.01/$0.99) â€” that
                # would dramatically mis-record P&L. Fall back to the
                # observed bid instead, which is closer to reality. The
                # actual cash on Polymarket reflects the true fill price either
                # way; this fallback only affects our internal P&L log.
                sell_avg_s = sell_result.get("average_fill_price")
                fallback_yes_leg = (current_sell_price / 100 if pos_side == "YES"
                                     else (1 - current_sell_price / 100))
                try:
                    sell_avg_yes_leg = float(sell_avg_s) if sell_avg_s else fallback_yes_leg
                except (TypeError, ValueError):
                    sell_avg_yes_leg = fallback_yes_leg
                if not sell_avg_s and sl_aggressive_sell:
                    log.warning(
                        f"  SL-SELL [{pos_ticker}]: Polymarket didn't return average_fill_price; "
                        f"using observed bid {current_sell_price}Â¢ for P&L (actual fill may differ)"
                    )
                # Reverse-derive the cents we sold at for logging clarity
                actual_sell_cents = (round(sell_avg_yes_leg * 100) if pos_side == "YES"
                                      else round((1 - sell_avg_yes_leg) * 100))
                entry_cost = (entry_yes_leg if pos_side == "YES"
                              else (1 - entry_yes_leg))
                sell_proceeds = (sell_avg_yes_leg if pos_side == "YES"
                                 else (1 - sell_avg_yes_leg))
                realized_pnl = (sell_proceeds - entry_cost) * sell_filled
                fee_per = 2 * MAKER_FEE_RATE * 0.5
                realized_pnl -= fee_per * sell_filled

                is_partial = sell_filled < sell_count
                partial_tag = f" (PARTIAL {sell_filled}/{sell_count})" if is_partial else ""
                exit_reason = ("take-profit" if tp_triggered
                               else "predict-cross-exit" if pcross_triggered
                               else "rrm-reversal-exit" if rrm_exit_triggered
                               else "futures-fast-exit" if fast_exit_triggered
                               else "dollar-stop" if dollar_stop_hit
                               else "stop-loss")
                exit_tag = ("TAKE-PROFIT" if tp_triggered
                            else "PCROSS-EXIT" if pcross_triggered
                            else "RRM-EXIT" if rrm_exit_triggered
                            else "FAST-EXIT" if fast_exit_triggered else "SL-EXIT")
                log.info(
                    f"  {exit_tag}-FILLED [{pos_ticker}]{partial_tag} mode={sell_mode_tag} "
                    f"{sell_filled}/{sell_count}c at {actual_sell_cents}Â¢ (trigger was "
                    f"{trigger_price}Â¢)  realized_pnl=${realized_pnl:+.2f}"
                )
                session.record(realized_pnl)
                _x_spot_exit = None
                try:
                    if (rrm_state.get("anchor_btc") and rrm_state.get("anchor_perp")
                            and rrm_state.get("strike")):
                        _xr_exit = rrm_evaluate(
                            side=pos_side, strike=rrm_state["strike"],
                            btc_entry=rrm_state["anchor_btc"],
                            perp_mid_entry=rrm_state["anchor_perp"],
                            mins_remaining=mins_remaining)
                        if _xr_exit and _xr_exit.get("ok"):
                            _x_spot_exit = _xr_exit.get("est_spot")
                except Exception:
                    _x_spot_exit = None
                _cgc_exit_e = _build_cv_composite_exit(
                    pos_side, _x_spot_exit, rrm_state.get("strike"), mins_remaining)
                audit.write({
                    "type":                "EARLY_EXIT",
                    "cv_composite_exit":   _cgc_exit_e,
                    "window_id":           window_id,
                    "ticker":              pos_ticker,
                    "side":                pos_side,
                    "tier":                position_tier,
                    "sl_threshold_cents":  effective_sl_cents,
                    "reason":              exit_reason,
                    "entry_cents":         entry_cents,
                    "exit_cents":          actual_sell_cents,
                    "observed_bid_at_trigger": current_sell_price,
                    "trigger_cents":       trigger_price,
                    "trigger_mode":        sl_trigger_mode,
                    "sell_mode":           sell_mode_tag,
                    "loss_cents":          loss_cents,
                    "actual_loss_cents":   entry_cents - actual_sell_cents,
                    "slippage_cents":      current_sell_price - actual_sell_cents,
                    "contracts_sold":      sell_filled,
                    "contracts_requested": sell_count,
                    "partial_fill":        is_partial,
                    "realized_pnl":        round(realized_pnl, 4),
                    "mins_remaining":      round(mins_remaining, 2),
                    "mins_since_entry":    round(mins_since_entry, 2),
                    "fast_exit_triggered": fast_exit_triggered,
                    "fast_exit_futures_move_pct": (round(fast_exit_move_pct, 4)
                                                    if fast_exit_triggered else None),
                    "rrm_exit_triggered":  rrm_exit_triggered,
                    "rrm_max_score":       rrm_state.get("max_score", 0),
                    # 2026-05-31: bid trajectory observed during SL monitor
                    "min_bid_cents":             pos.get("min_bid_cents"),
                    "max_bid_cents":             pos.get("max_bid_cents"),
                    "max_drawdown_cents":        pos.get("max_drawdown_cents", 0),
                    "min_bid_secs_from_entry":   pos.get("min_bid_secs_from_entry", 0),
                    "n_sl_polls":                pos.get("n_sl_polls", 0),
                })

                if is_partial:
                    # Keep remaining contracts in pending; will retry next poll
                    remaining = sell_count - sell_filled
                    ratio = remaining / pos_count if pos_count else 0
                    pos["contracts"] = remaining
                    pos["cost"]      = round(pos["cost"] * ratio, 2)
                    pos["net_win"]   = round(pos["net_win"] * ratio, 2)
                    log.warning(
                        f"  PARTIAL SL-EXIT: {remaining}c remaining on "
                        f"{pos_ticker}; will retry next poll"
                    )
                    await asyncio.sleep(effective_poll_interval)
                    continue

                # Full exit â€” but if a safety hedge is attached, KEEP the
                # window in pending so the settlement check can resolve the
                # hedge leg. The primary YES position has been sold; we hold
                # the NO hedge to settlement (this is the variance-reduction
                # whole point of the hedge).
                if hedge_state and hedge_state.get("active"):
                    # Mark primary as exited; preserve hedge for settlement
                    hedge_state["status"] = "yes_sl_fired_holding"
                    pos["primary_sold"]   = True
                    pos["primary_exit_cents"] = actual_sell_cents
                    pos["primary_realized_pnl"] = round(realized_pnl, 4)
                    log.info(
                        f"  Safety HEDGE held: primary {pos_side} closed, "
                        f"holding {hedge_state['contracts_filled']}c "
                        f"{hedge_state['side']} @ {hedge_state['entry_cents']}\u00a2 "
                        f"to settlement"
                    )
                    # â”€â”€ Phase 5: post-SL HEDGE BID MONITOR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    # Now that primary is exited, watch the hedge's bid.
                    # Sell IMMEDIATELY if:
                    #   (1) Bid >= hedge_no_sell_target (e.g., 97\u00a2): lock profit
                    #   (2) Bid has dropped >= hedge_no_sell_trail from peak: catch
                    #       reversal before hedge collapses
                    # Otherwise hold to settlement.
                    h_side = hedge_state.get("side", "NO")
                    h_contracts = hedge_state.get("contracts_filled", 0)
                    h_entry_yes_leg = hedge_state.get("entry_yes_leg", 0)
                    peak_bid = hedge_state.get("entry_cents", 0)
                    hedge_state["peak_bid_observed"] = peak_bid
                    log.info(
                        f"  Hedge BID-MONITOR started: target_sell={hedge_no_sell_target}\u00a2 "
                        f"trail={hedge_no_sell_trail}\u00a2 (poll every "
                        f"{hedge_post_sl_poll_s:.1f}s)"
                    )
                    while True:
                        now_h = datetime.now(timezone.utc)
                        h_mins_remaining = (close_dt - now_h).total_seconds() / 60.0
                        # Stop monitoring close to settlement â€” let it ride
                        if h_mins_remaining < sl_disable_late_mins:
                            log.info(
                                f"  Hedge BID-MONITOR: {h_mins_remaining:.1f}min left "
                                f"< {sl_disable_late_mins:.1f}min disable threshold "
                                f"\u2014 holding to settlement"
                            )
                            break
                        # Fetch fresh market for opposite-side bid
                        try:
                            h_mkt_data = await _kget(f"/markets/{pos_ticker}")
                            h_fresh = h_mkt_data.get("market", h_mkt_data)
                            _normalize_market(h_fresh)
                        except Exception as h_e:
                            log.debug(f"  Hedge BID-MONITOR fetch failed: {h_e}")
                            await asyncio.sleep(hedge_post_sl_poll_s)
                            continue
                        h_bid = int((h_fresh.get("no_bid") if h_side == "NO"
                                     else h_fresh.get("yes_bid")) or 0)
                        if h_bid <= 0:
                            await asyncio.sleep(hedge_post_sl_poll_s)
                            continue
                        if h_bid > peak_bid:
                            peak_bid = h_bid
                            hedge_state["peak_bid_observed"] = peak_bid
                        # Sell conditions
                        sell_reason = None
                        if h_bid >= hedge_no_sell_target:
                            sell_reason = "peak_target"
                        elif peak_bid - h_bid >= hedge_no_sell_trail and peak_bid > hedge_state.get("entry_cents", 0):
                            sell_reason = "trail_stop"
                        if not sell_reason:
                            await asyncio.sleep(hedge_post_sl_poll_s)
                            continue
                        # Sell the hedge at observed bid (NOT aggressive â€” we want
                        # the favorable bid we just observed).
                        h_sell_side = "ask" if h_side == "NO" else "bid"
                        # For SELLING NO: action is "ask" at yes_leg = 1 - bid/100
                        # For SELLING YES: action is "bid" at yes_leg = bid/100
                        h_sell_yes_leg = ((1 - h_bid/100) if h_side == "NO"
                                          else (h_bid/100))
                        h_sell_body = {
                            "ticker":                     pos_ticker,
                            "client_order_id":            uuid.uuid4().hex,
                            "side":                       h_sell_side,
                            "action":                     "sell",
                            "count":                      str(h_contracts),
                            "price":                      f"{h_sell_yes_leg:.4f}",
                            "time_in_force":              "immediate_or_cancel",
                            "self_trade_prevention_type": "taker_at_cross",
                        }
                        try:
                            h_sell_result = await _kpost("/portfolio/events/orders", h_sell_body)
                            h_sell_filled = int(float(h_sell_result.get("fill_count") or 0))
                        except Exception as h_e:
                            log.warning(f"  Hedge SELL failed: {h_e}; retry next poll")
                            await asyncio.sleep(hedge_post_sl_poll_s)
                            continue
                        if h_sell_filled <= 0:
                            log.warning(
                                f"  Hedge SELL: 0 fills at {h_bid}\u00a2 â€” bid moved; retry"
                            )
                            await asyncio.sleep(hedge_post_sl_poll_s)
                            continue
                        # Compute hedge PnL
                        h_sell_avg_s = h_sell_result.get("average_fill_price")
                        try:
                            h_sell_avg_yes_leg = (float(h_sell_avg_s) if h_sell_avg_s
                                                  else h_sell_yes_leg)
                        except (TypeError, ValueError):
                            h_sell_avg_yes_leg = h_sell_yes_leg
                        h_sell_cents = (round((1 - h_sell_avg_yes_leg) * 100)
                                        if h_side == "NO"
                                        else round(h_sell_avg_yes_leg * 100))
                        h_entry_cost_per = ((1 - h_entry_yes_leg) if h_side == "NO"
                                             else h_entry_yes_leg)
                        h_sell_proceeds_per = ((1 - h_sell_avg_yes_leg) if h_side == "NO"
                                                else h_sell_avg_yes_leg)
                        h_realized = (h_sell_proceeds_per - h_entry_cost_per) * h_sell_filled
                        h_realized -= 2 * MAKER_FEE_RATE * 0.5 * h_sell_filled
                        h_realized = round(h_realized, 4)
                        hedge_state["status"] = "sold"
                        hedge_state["exit_cents"] = h_sell_cents
                        hedge_state["exit_ts"] = datetime.now(timezone.utc).isoformat()
                        hedge_state["exit_reason"] = sell_reason
                        hedge_state["pnl_usd"] = h_realized
                        hedge_state["active"] = False  # no longer needs settlement
                        log.info(
                            f"  Hedge SOLD ({sell_reason}): {h_sell_filled}c "
                            f"@ {h_sell_cents}\u00a2 (peak was {peak_bid}\u00a2)  "
                            f"hedge_pnl=${h_realized:+.2f}"
                        )
                        session.record(h_realized)
                        # Combined PnL log line for visibility
                        combined_pnl = realized_pnl + h_realized
                        log.info(
                            f"  COMBINED (primary SL + hedge sold): "
                            f"primary=${realized_pnl:+.2f}  hedge=${h_realized:+.2f}  "
                            f"total=${combined_pnl:+.2f}"
                        )
                        # Audit HEDGE_EXIT (active sell, not settlement)
                        audit.write({
                            "type":                "HEDGE_EXIT",
                            "window_id":           window_id,
                            "ticker":              pos_ticker,
                            "primary_side":        pos_side,
                            "primary_pnl":         round(realized_pnl, 4),
                            "primary_sl_fired":    True,
                            "hedge_side":          h_side,
                            "hedge_contracts":     h_sell_filled,
                            "hedge_entry_cents":   hedge_state.get("entry_cents"),
                            "hedge_exit_cents":    h_sell_cents,
                            "hedge_peak_bid":      peak_bid,
                            "hedge_won":           h_realized > 0,
                            "hedge_pnl":           h_realized,
                            "combined_pnl":        round(combined_pnl, 4),
                            "exit_reason":         sell_reason,
                            "ts":                  datetime.now(timezone.utc).isoformat(),
                            "ts_ms":               int(time.time() * 1000),
                        })
                        # Clean up: remove window from pending now that both legs resolved
                        if window_id in session.pending:
                            del session.pending[window_id]
                        break
                else:
                    # â”€â”€ 2026-06-01: Smart Defensive FLIP (with retry loop) â”€â”€â”€
                    # No hedge is active. Now that primary SL has fired,
                    # check if signals support buying the OPPOSITE side as
                    # a defensive recovery position. Retry up to N times
                    # because eligibility can clear within 30-60s (e.g.,
                    # opp_bid climbs into the band, futures confirms).
                    flip_fired = False
                    if smart_flip_enabled:
                      for flip_attempt in range(max(1, smart_flip_retry_attempts)):
                        if flip_attempt > 0:
                            log.info(
                                f"  Smart FLIP retry {flip_attempt+1}/{smart_flip_retry_attempts} "
                                f"after {smart_flip_retry_sleep_s:.0f}s wait..."
                            )
                            await asyncio.sleep(smart_flip_retry_sleep_s)
                        # Check mins_remaining hasn't dropped below threshold
                        # during the wait (early-exit retry if so).
                        _retry_now = datetime.now(timezone.utc)
                        _retry_mins_remaining = (close_dt - _retry_now).total_seconds() / 60.0
                        if _retry_mins_remaining < smart_flip_min_mins_remaining:
                            log.info(
                                f"  Smart FLIP retry abandoned: only "
                                f"{_retry_mins_remaining:.1f}min remain "
                                f"(< {smart_flip_min_mins_remaining:.1f}min required)"
                            )
                            break
                        # Update mins_remaining used by eligibility check
                        mins_remaining = _retry_mins_remaining
                        # Fetch fresh market for opposite bid
                        try:
                            flip_mkt_data = await _kget(f"/markets/{pos_ticker}")
                            flip_fresh = flip_mkt_data.get("market", flip_mkt_data)
                            _normalize_market(flip_fresh)
                        except Exception as e:
                            log.warning(f"  Smart FLIP: fetch failed {e}; skipping retry")
                            flip_fresh = None
                        if flip_fresh:
                            flip_opp_side = "NO" if pos_side == "YES" else "YES"
                            flip_opp_bid = int((flip_fresh.get("no_bid") if flip_opp_side == "NO"
                                                else flip_fresh.get("yes_bid")) or 0)
                            flip_opp_ask = int((flip_fresh.get("no_ask") if flip_opp_side == "NO"
                                                else flip_fresh.get("yes_ask")) or 0)
                            # Eligibility gate
                            flip_elig, flip_reason = smart_flip_eligibility(
                                tier=position_tier, primary_side=pos_side,
                                mins_remaining=mins_remaining,
                                opp_bid_cents=flip_opp_bid,
                                hedge_active=False,
                                enabled=smart_flip_enabled,
                                eligible_tiers=smart_flip_tiers,
                                min_opp=smart_flip_min_opp_entry,
                                max_opp=smart_flip_max_opp_entry,
                                min_mins_remaining=smart_flip_min_mins_remaining,
                            )
                            if not flip_elig:
                                log.info(f"  Smart FLIP skipped: {flip_reason}")
                            else:
                                # Optional: futures continuation confirmation
                                # We want futures moving in the FLIP direction
                                # (= against the original primary direction)
                                futures_ok = True
                                futures_msg = "skip (confirm disabled)"
                                if smart_flip_require_futures_confirm and futures_lead is not None:
                                    try:
                                        flip_db = futures_lead.get_recent_move(
                                            lookback_s=smart_flip_futures_window_s
                                        )
                                    except Exception:
                                        flip_db = None
                                    flip_db_move = (flip_db or {}).get("move_pct")
                                    if flip_db_move is None:
                                        futures_ok = False
                                        futures_msg = "no futures-lead data"
                                    else:
                                        # For NO primary that SL'd â†’ flip side = YES â†’ we want futures UP
                                        # For YES primary that SL'd â†’ flip side = NO â†’ we want futures DOWN
                                        if flip_opp_side == "YES":
                                            # Need futures rising for flip-YES to win
                                            futures_ok = (flip_db_move >= smart_flip_futures_confirm_pct)
                                        else:
                                            # Need futures falling for flip-NO to win
                                            futures_ok = (flip_db_move <= -smart_flip_futures_confirm_pct)
                                        futures_msg = f"db_move={flip_db_move:+.4f}% vs threshold {smart_flip_futures_confirm_pct:.3f}%"
                                if not futures_ok:
                                    log.info(f"  Smart FLIP skipped: futures not confirming ({futures_msg})")
                                else:
                                    # Compute flip size
                                    primary_loss_dollars = abs(float(realized_pnl))
                                    # Use the ASK price (what we'd actually pay to buy)
                                    flip_buy_price = flip_opp_ask if flip_opp_ask > 0 else flip_opp_bid
                                    M_flip, M_reason = compute_smart_flip_size(
                                        primary_loss_usd=primary_loss_dollars,
                                        opp_entry_cents=flip_buy_price,
                                        sell_target_cents=smart_flip_sell_target,
                                        recovery_ratio=smart_flip_recovery_ratio,
                                        max_capital_usd=smart_flip_max_capital_usd,
                                    )
                                    if M_flip <= 0:
                                        log.info(f"  Smart FLIP skipped: {M_reason}")
                                    else:
                                        flip_capital_usd = M_flip * flip_buy_price / 100.0
                                        log.info(
                                            f"  Smart FLIP firing: BUY {flip_opp_side} {M_flip}c "
                                            f"@ {flip_buy_price}\u00a2 (capital=${flip_capital_usd:.2f}, "
                                            f"target_recovery=${primary_loss_dollars * smart_flip_recovery_ratio:.2f}, "
                                            f"futures: {futures_msg})"
                                        )
                                        # Submit chunked IOC for flip buy
                                        # For BUY NO on ticker: v2_side='ask', yes_leg = 1 - buy/100
                                        # For BUY YES on ticker: v2_side='bid', yes_leg = buy/100
                                        flip_v2_side = "ask" if flip_opp_side == "NO" else "bid"
                                        flip_v2_yes_leg = ((1 - flip_buy_price / 100)
                                                            if flip_opp_side == "NO"
                                                            else (flip_buy_price / 100))
                                        flip_filled = 0
                                        flip_sum_x_price = 0.0
                                        flip_order_ids = []
                                        FLIP_CHUNK = 5
                                        FLIP_MAX_CHUNKS = max(10, (M_flip // FLIP_CHUNK) + 4)
                                        for f_chunk_idx in range(FLIP_MAX_CHUNKS):
                                            f_remaining = M_flip - flip_filled
                                            if f_remaining <= 0:
                                                break
                                            f_this = min(FLIP_CHUNK, f_remaining)
                                            f_body = {
                                                "ticker":                     pos_ticker,
                                                "client_order_id":            uuid.uuid4().hex,
                                                "side":                       flip_v2_side,
                                                "count":                      str(f_this),
                                                "price":                      f"{flip_v2_yes_leg:.4f}",
                                                "time_in_force":              "immediate_or_cancel",
                                                "self_trade_prevention_type": "taker_at_cross",
                                            }
                                            try:
                                                f_result = await _kpost("/portfolio/events/orders", f_body)
                                            except Exception as f_e:
                                                log.warning(f"  Smart FLIP chunk {f_chunk_idx+1} failed: {f_e}")
                                                break
                                            f_filled_chunk = int(float(f_result.get("fill_count") or 0))
                                            f_avg_s = f_result.get("average_fill_price")
                                            try:
                                                f_avg_yes_leg = float(f_avg_s) if f_avg_s else flip_v2_yes_leg
                                            except (TypeError, ValueError):
                                                f_avg_yes_leg = flip_v2_yes_leg
                                            if f_filled_chunk <= 0:
                                                break  # orderbook exhausted at this ask
                                            flip_filled += f_filled_chunk
                                            flip_sum_x_price += f_filled_chunk * f_avg_yes_leg
                                            f_oid = f_result.get("order_id") or f_result.get("client_order_id")
                                            if f_oid:
                                                flip_order_ids.append(f_oid)
                                        if flip_filled <= 0:
                                            log.warning(
                                                f"  Smart FLIP: NO fills at {flip_buy_price}\u00a2 "
                                                f"(orderbook exhausted). Primary remains as-is."
                                            )
                                        else:
                                            flip_fired = True
                                            flip_avg_yes_leg = flip_sum_x_price / flip_filled
                                            flip_avg_cents = (round((1 - flip_avg_yes_leg) * 100)
                                                                if flip_opp_side == "NO"
                                                                else round(flip_avg_yes_leg * 100))
                                            flip_actual_capital = flip_filled * flip_avg_cents / 100.0
                                            pos["flip"] = {
                                                "active":            True,
                                                "side":              flip_opp_side,
                                                "contracts_target":  M_flip,
                                                "contracts_filled":  flip_filled,
                                                "entry_yes_leg":     flip_avg_yes_leg,
                                                "entry_cents":       flip_avg_cents,
                                                "entry_ts":          datetime.now(timezone.utc).isoformat(),
                                                "order_ids":         flip_order_ids,
                                                "status":            "active",
                                                "capital_usd":       round(flip_actual_capital, 2),
                                                "sl_cents":          smart_flip_sl_cents,
                                                "sell_target":       smart_flip_sell_target,
                                                "trail_cents":       smart_flip_trail_cents,
                                                "peak_bid":          flip_avg_cents,
                                            }
                                            # Mark primary as already-recorded
                                            pos["primary_sold"]   = True
                                            pos["primary_exit_cents"] = actual_sell_cents
                                            pos["primary_realized_pnl"] = round(realized_pnl, 4)
                                            log.info(
                                                f"  Smart FLIP ATTACHED: BUY {flip_opp_side} "
                                                f"{flip_filled}/{M_flip}c avg={flip_avg_cents}\u00a2 "
                                                f"capital=${flip_actual_capital:.2f}  "
                                                f"target_sell={smart_flip_sell_target}\u00a2  "
                                                f"flip_sl=-{smart_flip_sl_cents}\u00a2  "
                                                f"trail={smart_flip_trail_cents}\u00a2"
                                            )
                                            audit.write({
                                                "type":              "FLIP_ATTACH",
                                                "window_id":         window_id,
                                                "ticker":            pos_ticker,
                                                "primary_side":      pos_side,
                                                "primary_entry":     entry_cents,
                                                "primary_exit":      actual_sell_cents,
                                                "primary_pnl":       round(realized_pnl, 4),
                                                "primary_tier":      position_tier,
                                                "flip_side":         flip_opp_side,
                                                "flip_target":       M_flip,
                                                "flip_filled":       flip_filled,
                                                "flip_entry_cents":  flip_avg_cents,
                                                "flip_capital_usd":  round(flip_actual_capital, 2),
                                                "flip_sl_cents":     smart_flip_sl_cents,
                                                "flip_sell_target":  smart_flip_sell_target,
                                                "flip_trail":        smart_flip_trail_cents,
                                                "futures_confirm":   futures_msg,
                                                "ts":                datetime.now(timezone.utc).isoformat(),
                                                "ts_ms":             int(time.time() * 1000),
                                            })
                                            # Run flip monitor loop
                                            flip_state = pos["flip"]
                                            flip_peak = flip_avg_cents
                                            while True:
                                                now_f = datetime.now(timezone.utc)
                                                f_mins_remaining = (close_dt - now_f).total_seconds() / 60.0
                                                if f_mins_remaining < sl_disable_late_mins:
                                                    log.info(
                                                        f"  Smart FLIP monitor: {f_mins_remaining:.1f}min left "
                                                        f"\u2014 holding to settlement"
                                                    )
                                                    break
                                                try:
                                                    fm_data = await _kget(f"/markets/{pos_ticker}")
                                                    fm_fresh = fm_data.get("market", fm_data)
                                                    _normalize_market(fm_fresh)
                                                except Exception:
                                                    await asyncio.sleep(smart_flip_poll_s)
                                                    continue
                                                fm_bid = int((fm_fresh.get("no_bid") if flip_opp_side == "NO"
                                                              else fm_fresh.get("yes_bid")) or 0)
                                                if fm_bid <= 0:
                                                    await asyncio.sleep(smart_flip_poll_s)
                                                    continue
                                                if fm_bid > flip_peak:
                                                    flip_peak = fm_bid
                                                    flip_state["peak_bid"] = flip_peak
                                                # Sell conditions
                                                f_sell_reason = None
                                                if fm_bid >= smart_flip_sell_target:
                                                    f_sell_reason = "tp_target"
                                                elif (flip_avg_cents - fm_bid) >= smart_flip_sl_cents:
                                                    f_sell_reason = "flip_sl"
                                                elif (flip_peak - fm_bid) >= smart_flip_trail_cents and flip_peak > flip_avg_cents:
                                                    f_sell_reason = "trail_stop"
                                                if not f_sell_reason:
                                                    await asyncio.sleep(smart_flip_poll_s)
                                                    continue
                                                # Submit sell IOC
                                                # For SELLING NO: side='ask', yes_leg = 1 - bid/100
                                                # For SELLING YES: side='bid', yes_leg = bid/100
                                                # For SL exit we want aggressive (penny limit); for TP we use observed bid
                                                if f_sell_reason == "flip_sl":
                                                    f_sell_yes_leg = 0.01 if flip_opp_side == "YES" else 0.99
                                                else:
                                                    f_sell_yes_leg = ((1 - fm_bid/100) if flip_opp_side == "NO"
                                                                       else (fm_bid/100))
                                                f_sell_side = "ask" if flip_opp_side == "NO" else "bid"
                                                f_sell_body = {
                                                    "ticker":                     pos_ticker,
                                                    "client_order_id":            uuid.uuid4().hex,
                                                    "side":                       f_sell_side,
                                                    "action":                     "sell",
                                                    "count":                      str(flip_filled),
                                                    "price":                      f"{f_sell_yes_leg:.4f}",
                                                    "time_in_force":              "immediate_or_cancel",
                                                    "self_trade_prevention_type": "taker_at_cross",
                                                }
                                                try:
                                                    f_sell_result = await _kpost("/portfolio/events/orders", f_sell_body)
                                                    f_sell_filled = int(float(f_sell_result.get("fill_count") or 0))
                                                except Exception as f_e:
                                                    log.warning(f"  Smart FLIP sell failed: {f_e}; retry")
                                                    await asyncio.sleep(smart_flip_poll_s)
                                                    continue
                                                if f_sell_filled <= 0:
                                                    log.warning(f"  Smart FLIP sell: 0 fills at {fm_bid}\u00a2; retry")
                                                    await asyncio.sleep(smart_flip_poll_s)
                                                    continue
                                                # Compute realized PnL on flip
                                                f_sell_avg_s = f_sell_result.get("average_fill_price")
                                                try:
                                                    f_sell_avg = float(f_sell_avg_s) if f_sell_avg_s else f_sell_yes_leg
                                                except (TypeError, ValueError):
                                                    f_sell_avg = f_sell_yes_leg
                                                f_sell_cents = (round((1 - f_sell_avg) * 100)
                                                                  if flip_opp_side == "NO"
                                                                  else round(f_sell_avg * 100))
                                                f_entry_cost = ((1 - flip_avg_yes_leg) if flip_opp_side == "NO"
                                                                 else flip_avg_yes_leg)
                                                f_sell_proc = ((1 - f_sell_avg) if flip_opp_side == "NO"
                                                                else f_sell_avg)
                                                f_realized = (f_sell_proc - f_entry_cost) * f_sell_filled
                                                f_realized -= 2 * MAKER_FEE_RATE * 0.5 * f_sell_filled
                                                f_realized = round(f_realized, 4)
                                                flip_state["status"] = "sold"
                                                flip_state["exit_cents"] = f_sell_cents
                                                flip_state["exit_ts"] = datetime.now(timezone.utc).isoformat()
                                                flip_state["exit_reason"] = f_sell_reason
                                                flip_state["pnl_usd"] = f_realized
                                                flip_state["active"] = False
                                                log.info(
                                                    f"  Smart FLIP SOLD ({f_sell_reason}): "
                                                    f"{f_sell_filled}c @ {f_sell_cents}\u00a2 "
                                                    f"(peak {flip_peak}\u00a2)  flip_pnl=${f_realized:+.2f}"
                                                )
                                                session.record(f_realized)
                                                combined_pnl = realized_pnl + f_realized
                                                log.info(
                                                    f"  COMBINED (primary SL + flip): "
                                                    f"primary=${realized_pnl:+.2f}  "
                                                    f"flip=${f_realized:+.2f}  "
                                                    f"total=${combined_pnl:+.2f}"
                                                )
                                                audit.write({
                                                    "type":              "FLIP_EXIT",
                                                    "window_id":         window_id,
                                                    "ticker":            pos_ticker,
                                                    "primary_side":      pos_side,
                                                    "primary_pnl":       round(realized_pnl, 4),
                                                    "flip_side":         flip_opp_side,
                                                    "flip_contracts":    f_sell_filled,
                                                    "flip_entry_cents":  flip_avg_cents,
                                                    "flip_exit_cents":   f_sell_cents,
                                                    "flip_peak_bid":     flip_peak,
                                                    "flip_pnl":          f_realized,
                                                    "combined_pnl":      round(combined_pnl, 4),
                                                    "exit_reason":       f_sell_reason,
                                                    "ts":                datetime.now(timezone.utc).isoformat(),
                                                    "ts_ms":             int(time.time() * 1000),
                                                })
                                                if window_id in session.pending:
                                                    del session.pending[window_id]
                                                break
                            # End of flip elig/futures/size logic
                        # 2026-06-01: exit retry loop on successful fire
                        if flip_fired:
                            break
                    # If flip didn't fire across all retries (disabled,
                    # ineligible, or zero-fill), remove from pending.
                    if not flip_fired and window_id in session.pending:
                        del session.pending[window_id]
                # â”€â”€ 2026-06-16: STOP-LOSS RE-ENTRY (live test) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # The position was FULLY closed via an adverse exit (stop-
                # loss / RRM reversal / futures fast-exit). If SL re-entry is
                # enabled, the window still has trading time, and we did NOT
                # already open a defensive smart-flip position, re-open the
                # window for a fresh signal evaluation: discard it from
                # session.traded so the main loop re-runs the FULL gate stack
                # and may re-enter (re-buy the same side or flip) ONLY IF
                # signals still align. Independent of the resting-TP re-entry
                # path (which fires on a profitable full sell). Skipped on
                # partial fills (handled above via `continue`) and on the
                # hedge-held branch (which breaks earlier, holding to
                # settlement). WARNING: this re-evaluates immediately after a
                # losing exit â€” a fresh entry can catch a genuine flip OR
                # re-buy into a continuing reversal. Re-entry re-runs the full
                # gate, so it only fires when the signal qualifies again.
                if (sl_reentry_enabled
                        and exit_reason in ("stop-loss",
                                            "rrm-reversal-exit",
                                            "futures-fast-exit",
                                            "predict-cross-exit")
                        and not flip_fired):
                    _now = datetime.now(timezone.utc)
                    _mins_left_slre = (close_dt - _now).total_seconds() / 60.0
                    if _mins_left_slre >= 2.5:
                        session.traded.discard(window_id)
                        window_no_fills.pop(window_id, None)
                        log.warning(
                            f"  [SL-REENTRY] adverse exit ({exit_reason}) "
                            f"complete with {_mins_left_slre:.1f}min left â€” "
                            f"re-opening window {window_id} for fresh signal "
                            f"evaluation (re-entry IF signals align).")
                        audit.write({
                            "type": "SL_REENTRY_REOPEN",
                            "window_id": window_id,
                            "ticker": pos_ticker,
                            "prior_side": pos_side,
                            "exit_reason": exit_reason,
                            "mins_left": round(_mins_left_slre, 2),
                            "ts": _now.isoformat(),
                            "ts_ms": int(time.time() * 1000),
                        })
                    else:
                        log.info(
                            f"  [SL-REENTRY] adverse exit ({exit_reason}) "
                            f"complete but only {_mins_left_slre:.1f}min left "
                            f"(< 2.5min) â€” not re-opening; letting window close.")
                sl_exited = True
                break

            if sl_exited:
                # Skip the long sleep â€” position already closed, move on
                continue

            # â”€â”€ 2026-06-12: RRM LATE-WINDOW COVERAGE (log-only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # The SL monitor disarms in the final ~2.5 min, which left late
            # entries (e.g. 06-12 13:56, entered at 3.7 min, breached at
            # T+120s) completely unwatched. Keep evaluating the RRM (log +
            # audit only, no selling) until ~25s before close.
            if (window_id in session.pending and not rrm_state["fired"]
                    and rrm_state["anchor_btc"] and rrm_state["anchor_perp"]
                    and rrm_state["strike"]):
                while True:
                    now = datetime.now(timezone.utc)
                    late_mins_remaining = (close_dt - now).total_seconds() / 60.0
                    if late_mins_remaining < 0.4 or window_id not in session.pending:
                        break
                    try:
                        _lm = await _kget(f"/markets/{pos['ticker']}")
                        _lmkt = _lm.get("market", _lm)
                        _normalize_market(_lmkt)
                        _csp = int((_lmkt.get("yes_bid") if pos["side"] == "YES"
                                    else _lmkt.get("no_bid")) or 0)
                    except Exception:
                        _csp = 0
                    if _csp > 0:
                        try:
                            rrm = rrm_evaluate(
                                side=pos["side"],
                                strike=rrm_state["strike"],
                                btc_entry=rrm_state["anchor_btc"],
                                perp_mid_entry=rrm_state["anchor_perp"],
                                mins_remaining=late_mins_remaining,
                            )
                            if rrm["ok"]:
                                rrm_state["max_score"] = max(
                                    rrm_state["max_score"], rrm["score"])
                                if rrm["breach"] and not rrm_state["breach_logged"]:
                                    rrm_state["breach_logged"] = True
                                    log.warning(
                                        f"  [RRM log-only/late] STRIKE BREACH "
                                        f"[{pos['ticker']}] {pos['side']}: "
                                        f"{rrm['summary']} (bid={_csp}\u00a2, "
                                        f"{late_mins_remaining:.1f}min left)")
                                if rrm["would_exit"]:
                                    rrm_state["fired"] = True
                                    _wpnl = ((_csp - entry_cents) / 100.0
                                             * pos["contracts"])
                                    log.warning(
                                        f"  [RRM log-only/late] WOULD-EXIT "
                                        f"[{pos['ticker']}] {pos['side']} "
                                        f"{pos['contracts']}c: {rrm['summary']} â€” "
                                        f"would sell at {_csp}\u00a2 (entry "
                                        f"{entry_cents}\u00a2, would-realize "
                                        f"${_wpnl:+.2f}). NOT SELLING (log-only).")
                                    audit.write({
                                        "type": "RRM_WOULD_EXIT",
                                        "window_id": window_id,
                                        "ticker": pos["ticker"],
                                        "side": pos["side"],
                                        "contracts": pos["contracts"],
                                        "entry_cents": entry_cents,
                                        "would_exit_bid_cents": _csp,
                                        "would_realize_pnl": round(_wpnl, 2),
                                        "score": rrm["score"],
                                        "components": rrm["components"],
                                        "est_spot": rrm["est_spot"],
                                        "strike": rrm_state["strike"],
                                        "mins_remaining": round(late_mins_remaining, 2),
                                        "late_phase": True,
                                    })
                                    break   # one fire per position; stop polling
                        except Exception as _le:
                            log.debug(f"  RRM late eval failed: {_le!r}")
                    await asyncio.sleep(3)

            # Settlement wait recomputed from NOW (the monitor + late coverage
            # already consumed most of the window; the old formula re-slept
            # the full original duration).
            now = datetime.now(timezone.utc)
            wait_s = max(5, (close_dt - now).total_seconds() + 45)
            log.info(f"Waiting {fmt(wait_s)} for window to settle...")
            await asyncio.sleep(wait_s)
            continue

        # Sleep until window closes + 45s buffer for settlement
        wait_s = max(5, (mins_left + 0.75) * 60)
        log.info(f"Waiting {fmt(wait_s)} for window to settle...")
        await asyncio.sleep(wait_s)


# â”€â”€ Entry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Polymarket BTC 15m autonomous trading daemon (ChainVector-enhanced). "
                    "Defaults match the canonical production run â€” running with no "
                    "flags is equivalent to the full baseline command.")
    parser.add_argument("--dry-run",            action="store_true", help="Simulate trades, no real orders")
    parser.add_argument("--bankroll",           type=float, default=500.0, help="Starting bankroll in USD (default 500)")
    parser.add_argument("--strategy", type=str, default="baseline",
                        choices=["baseline", "conservative", "aggressive",
                                 "momentum", "probability", "stability"],
                        help="Strategy template: a named preset over the ChainVector "
                             "gate stack. baseline = canonical production defaults; "
                             "conservative = tighten every CV veto + smaller risk; "
                             "aggressive = relax CV vetoes for more entries; "
                             "momentum = momentum-scorecard-led (bigger /momentum EV "
                             "boost, stricter against-veto); probability = "
                             "probability-engine-led (higher --ev-tp-weight, higher "
                             "prob floor); stability = prediction-quote-microstructure-"
                             "led (strict /predictions/stability + bid-stability "
                             "gates). Every knob a preset sets can still be overridden "
                             "by passing its own flag explicitly.")
    parser.add_argument("--no-term-prob",       action="store_true",
                        help="Disable the ChainVector terminal-probability engine entirely")
    parser.add_argument("--term-prob-relax",    action="store_true",
                        help="Allow TermProb-confirmed signals to use relaxed gap floor (0.08)")
    parser.add_argument("--no-futures-lead",    action="store_true",
                        help="Disable the ChainVector futures lead-lag veto (binance_futures venue)")
    parser.add_argument("--futures-lead-lookback", type=float, default=6.0,
                        help="Seconds of lead-venue futures move to inspect (default 6)")
    parser.add_argument("--futures-lead-veto-bps", type=float, default=5.0,
                        help="Veto trade if futures moved >X bps against direction in lookback window (default 5)")
    # â”€â”€ ChainVector signal gates (this program's native signal provider) â”€â”€â”€â”€â”€
    parser.add_argument("--no-cv-mom-boost", dest="cv_mom_boost_enabled", action="store_false",
                        default=True,
                        help="Disable the ChainVector momentum-scorecard EV boost (ON by "
                             "default): combined_p is nudged toward the cross-venue "
                             "/momentum aggregate, capped at --cv-mom-boost-weight.")
    parser.add_argument("--cv-mom-boost-weight", type=float, default=0.05,
                        help="Max combined_p nudge from the momentum scorecard "
                             "(default 0.05 = \u00b15pp).")
    parser.add_argument("--cv-mom-boost-scale", type=float, default=50.0,
                        help="Aggregate score at which the boost ~saturates "
                             "(default 50 of the \u00b1100 scale).")
    parser.add_argument("--no-cv-mom-veto", dest="cv_mom_veto_enabled", action="store_false",
                        default=True,
                        help="Disable the ChainVector momentum strong-against veto (ON by "
                             "default): entries are blocked when the signed aggregate "
                             "score is \u2264 -â€“cv-mom-veto-score with breadth confirming.")
    parser.add_argument("--cv-mom-veto-score", type=float, default=65.0,
                        help="Signed aggregate momentum score at/below which entry is "
                             "vetoed (default 65 â€” 'strongly against').")
    parser.add_argument("--cv-mom-veto-breadth", type=float, default=0.60,
                        help="Breadth confirmation for the momentum veto: fraction of "
                             "venues moving the adverse way (default 0.60).")
    parser.add_argument("--no-cv-prob-veto", dest="cv_prob_veto_enabled", action="store_false",
                        default=True,
                        help="Disable the probability-engine floor veto (ON by default): "
                             "entries are blocked when the six-estimator ensemble gives "
                             "our side \u2264 --cv-prob-veto-max at the exact time-to-close.")
    parser.add_argument("--cv-prob-veto-max", type=float, default=0.22,
                        help="Ensemble P(our side) at/below which entry is vetoed "
                             "(default 0.22).")
    parser.add_argument("--no-cv-stab-veto", dest="cv_stab_veto_enabled", action="store_false",
                        default=True,
                        help="Disable the ChainVector prediction quote-stability veto (ON "
                             "by default): entries are blocked while /predictions/stability "
                             "shows the Polymarket quote repricing hard against our side.")
    parser.add_argument("--cv-stab-mom-against", type=float, default=45.0,
                        help="Signed stability momentum_score at/below which entry is "
                             "vetoed (default 45 on the \u00b1100 scale).")
    parser.add_argument("--no-cv-cascade-veto", dest="cv_cascade_veto_enabled",
                        action="store_false", default=True,
                        help="Disable the liquidation cascade-risk veto (ON by default): "
                             "entries are blocked when cascade risk_score \u2265 "
                             "--cv-cascade-veto-score and the at-risk side's forced flow "
                             "points against the position.")
    parser.add_argument("--cv-cascade-veto-score", type=float, default=75.0,
                        help="Cascade risk_score at/above which the veto can fire "
                             "(default 75).")
    parser.add_argument("--no-consensus-veto", action="store_true",
                        help="Disable the binance_futures+OKX consensus veto. By default, when BOTH "
                             "external feeds independently show a directional move â‰¥0.005%% against "
                             "the trade (even sub-5bps), the trade is vetoed. Overnight 5/27 data "
                             "showed this exact pattern in the only loss of 14 settled trades.")
    parser.add_argument("--consensus-min-move-pct", type=float, default=0.005,
                        help="Minimum |move%%| for an external feed to count as directional "
                             "in the consensus veto (default 0.005%% = 0.5 bps). Below this, "
                             "the feed is treated as NEUTRAL (no fresh info).")
    parser.add_argument("--consensus-okx-lookback-s", type=float, default=6.0,
                        help="OKX lookback window for the consensus veto in seconds "
                             "(default 6.0 â€” matches the futures-lead lookback).")
    # â”€â”€ 2026-06-01: Smart consensus-veto bypass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # 24h audit (17 vetoes): 65% had recent 5m bars strongly favoring the
    # trade direction â€” the 6s blip was clearly counter-trend noise. Three
    # independent bypasses prevent false-positive vetoes without weakening
    # the core veto when externals are genuinely confirming a real adverse
    # trend.
    parser.add_argument("--no-consensus-smart-bypass", action="store_true",
                        help="Disable smart bypasses on the consensus veto. With smart "
                             "bypass ENABLED (default), the veto is skipped when ANY of "
                             "three signals indicate the 6s blip is noise: "
                             "(1) longer-window futures favors the trade, "
                             "(2) recent 5m bars favor the trade, "
                             "(3) BTC is far from strike and the consensus moves are tiny.")
    parser.add_argument("--consensus-long-window-s", type=float, default=60.0,
                        help="Bypass-1: longer futures-lead window (seconds) checked against "
                             "the 6s blip. If this longer window shows â‰¥ "
                             "--consensus-long-min-pct in the FAVORABLE direction, the "
                             "6s blip is treated as noise and the veto is skipped. "
                             "Default 60s.")
    parser.add_argument("--consensus-long-min-pct", type=float, default=0.030,
                        help="Bypass-1: minimum |move%%| in the longer window to count "
                             "as a 'broader trend' for bypass. Default 0.030%% "
                             "(must be 6x stronger than the 6s consensus threshold).")
    parser.add_argument("--consensus-5m-favor-pct", type=float, default=0.10,
                        help="Bypass-2: minimum sum of recent_5m_pct bars (in the "
                             "FAVORABLE direction) to bypass the veto. Default 0.10%% "
                             "over the last ~30 min of 5m candles.")
    parser.add_argument("--consensus-far-dist-pct", type=float, default=0.25,
                        help="Bypass-3: minimum |dist_pct| (BTC vs strike) above which "
                             "tiny consensus moves are treated as noise. Default 0.25%%.")
    parser.add_argument("--consensus-far-max-move-pct", type=float, default=0.020,
                        help="Bypass-3: maximum of |db_move|, |okx_move| below which "
                             "the veto is bypassed if dist condition is also met. "
                             "Default 0.020%% (4x the consensus directional threshold).")
    # â”€â”€ Stop-loss (NEW 2026-05-27) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # 15m-specific stop-loss with WIDE 45Â¢ default trigger. Designed to cap
    # downside on the rare loss trades (we're at 92%+ WR but each loss is
    # roughly the full entry cost). Wide enough to weather normal bid-ask
    # volatility and brief reversals; tight enough to limit damage when
    # BTC actually moves against us.
    parser.add_argument("--no-sl", action="store_true",
                        help="Disable the stop-loss. Default: ENABLED with -45Â¢ trigger.")
    parser.add_argument("--sl-loss-cents", type=int, default=99,
                        help="Exit when trigger price drops this many cents below entry "
                             "(default 99 = per-contract stop effectively OFF; the DOLLAR "
                             "stop --max-loss-per-trade is the live protection).")
    parser.add_argument("--sl-loss-cents-high-conv", type=int, default=99,
                        help="Stop-loss threshold for HIGH-CONV AND LATE-SURE tiers (default 30). "
                             "Tighter than STANDARD's 45Â¢ because these high-price tiers have "
                             "asymmetric downside: entries at 86-96Â¢ mean upside is capped at "
                             "4-14Â¢ while downside can be 70-90Â¢+. A loss past 30Â¢ means BTC has "
                             "clearly moved against the thesis; better to exit early. The flag "
                             "is named '--sl-loss-cents-high-conv' for historical reasons; it "
                             "applies to BOTH high-price tiers.")
    parser.add_argument("--max-loss-per-trade", type=float, default=75.0,
                        help="Hard per-position DOLLAR stop-loss for golden/standard tiers "
                             "(default 75; 0 = disabled). When a golden/standard position is down "
                             "this many dollars (loss_cents x contracts / 100), flatten it via "
                             "the SL sell path, bypassing grace. Backtest 06-22: a $100 cap "
                             "turns golden+standard net-positive across regimes by truncating "
                             "the bankroll-scaled left tail. high_conv uses its own cap "
                             "(--max-loss-per-trade-high-conv); late_sure keeps its "
                             "per-contract stop (--sl-loss-cents-high-conv).")
    parser.add_argument("--max-loss-per-trade-high-conv", type=float, default=75.0,
                        help="Hard per-position DOLLAR stop-loss for the HIGH_CONV tier only "
                             "(default 75; 0 = disabled). high_conv enters deep ITM (86-99c) with no "
                             "effective per-contract stop, so a genuine reversal rides to "
                             "settlement (e.g. -$615/-$618). BID_TRAJECTORY backtest: the deepest "
                             "high_conv WINNER dipped $343 before recovering, while catchable "
                             "losers dipped $459/$606 â€” so a $400 cap clips 0 winners and truncates "
                             "the tail. Fires via the same SL sell path, live until the "
                             "pcross disarm floor (~0.4min), so it catches all but last-~24s cliffs.")
    parser.add_argument("--sl-grace-mins", type=float, default=1.5,
                        help="Minutes after entry before SL can fire (default 1.5). "
                             "15m daemon enters at 3-10 min left, so grace must be short.")
    parser.add_argument("--sl-disable-late-mins", type=float, default=1.5,
                        help="Disable SL when mins_left < this (default 1.5). In the final "
                             "minute or two, let natural settlement handle the position "
                             "rather than panic-selling on a transient bid dip.")
    parser.add_argument("--sl-poll-interval-s", type=float, default=2.0,
                        help="SL-monitor poll interval in seconds for STANDARD/strong/late-dir "
                             "tiers (default 2s).")
    parser.add_argument("--sl-poll-interval-hc-s", type=float, default=2.0,
                        help="SL-monitor poll interval in seconds for HIGH-CONV + LATE-SURE tiers "
                             "(default 2s, faster than standard 5s). 2026-05-30: faster polling "
                             "for high-price tiers where bid can drop 30-40\u00a2 between observations. "
                             "Lower = more API calls but more chances to catch the bid mid-drop "
                             "before it blows past the SL trigger.")
    parser.add_argument("--no-futures-fast-exit", action="store_true", default=True,
                        help="Disable the futures fast-exit signal (DISABLED by default per the "
                             "canonical baseline; use --futures-fast-exit to enable). When enabled, "
                             "during SL monitor the daemon checks the ChainVector lead-venue futures "
                             "and exits early if futures moved sharply against the position.")
    parser.add_argument("--futures-fast-exit", dest="no_futures_fast_exit", action="store_false",
                        help="Enable the futures fast-exit signal (off in the canonical baseline).")
    parser.add_argument("--futures-fast-exit-window-s", type=float, default=30.0,
                        help="Lookback window (seconds) for the futures fast-exit check. "
                             "Default 30 (tightened from 60 on 2026-05-30 PM after 0 fires "
                             "in first 10h showed initial setting was too conservative). "
                             "Shorter = more reactive but noisier; longer = smoother but slower.")
    parser.add_argument("--futures-fast-exit-threshold-pct", type=float, default=0.20,
                        help="Threshold (%%) for the futures fast-exit. Sustained adverse move "
                             ">= this magnitude triggers immediate exit. Default 0.20%% "
                             "(tightened from 0.30%% on 2026-05-30 PM). Lower = catches more "
                             "losses but more false-exits on winners.")
    parser.add_argument("--futures-fast-exit-sanity-max-pct", type=float, default=5.0,
                        help="Sanity guard (%%): any |futures move| larger than this is treated "
                             "as a bad tick and ignored. Default 5.0%% -- protects against "
                             "occasional garbage values; this filter prevents spurious "
                             "fast-exits on bad data.")
    # 2026-05-31 NEW: Safety Hedge (Phase 1 MVP)
    parser.add_argument("--hedge-enabled", action="store_true",
                        help="Enable the safety hedge feature. After a HIGH-CONV or STRONG-FLOOR "
                             "primary position fills, the daemon attempts to buy OPPOSITE-side "
                             "contracts on the same ticker as a downside hedge. If primary wins, "
                             "hedge settles at 0 (premium paid out of larger primary). If primary "
                             "SL fires, hedge is held to settlement and offsets the loss. Default OFF "
                             "(explicit opt-in for safety).")
    parser.add_argument("--hedge-tiers", type=str, default="high_conv,strong",
                        help="Comma-separated tier names eligible for hedging "
                             "(default: high_conv,strong). LATE-SURE is intentionally excluded "
                             "because at 96-99\u00a2 entries the hedge math breaks (no margin).")
    parser.add_argument("--hedge-min-yes-entry", type=int, default=70,
                        help="Min primary entry price (\u00a2) for hedge eligibility (default 70). "
                             "Below this the position has natural cushion and hedge is wasteful.")
    parser.add_argument("--hedge-max-yes-entry", type=int, default=88,
                        help="Max primary entry price (\u00a2) for hedge eligibility (default 88). "
                             "Above this the hedge ratio swallows all profit margin.")
    parser.add_argument("--hedge-max-no-cost", type=int, default=28,
                        help="Max opposite-side ask (\u00a2) to buy the hedge at (default 28). "
                             "If the opposite ask is wider than this, math fails -- skip.")
    parser.add_argument("--hedge-no-settle-assumed", type=float, default=0.95,
                        help="Conservative assumed settle value (dollars, 0-1) when computing hedge "
                             "size. Default 0.95 (95\u00a2). The actual settle value is $1.00 if hedge "
                             "wins, but using 0.95 builds in slippage on the hedge sell or "
                             "settle-timing risk.")
    parser.add_argument("--hedge-max-capital-mult", type=float, default=2.5,
                        help="Cap on total (primary + hedge) capital as a multiple of primary alone. "
                             "Default 2.5\u00d7. Limits how aggressively the hedge sizes up.")
    parser.add_argument("--hedge-widened-sl-cents", type=int, default=50,
                        help="When hedge is attached, widen the SL trigger to this many cents below "
                             "entry (default 50). Wider SL = fewer false-stops on whip-saws (hedge "
                             "covers the deeper loss). The hedge sizing math uses this same value.")
    parser.add_argument("--hedge-no-sell-target", type=int, default=97,
                        help="After primary SL fires, sell the hedge IMMEDIATELY if its bid hits "
                             "this threshold (default 97\u00a2). Locks in profit and avoids the "
                             "cross-strike-twice tail risk (BTC reverses again, hedge collapses).")
    parser.add_argument("--hedge-no-sell-trail", type=int, default=10,
                        help="After primary SL fires, sell the hedge if its bid drops this many "
                             "cents from peak (default 10\u00a2). Catches reversals before they "
                             "wipe out hedge profit.")
    parser.add_argument("--hedge-post-sl-poll-s", type=float, default=2.0,
                        help="Poll interval (seconds) for the hedge bid monitor that runs after "
                             "primary SL fires (default 2.0). Faster = catches peaks better but "
                             "more API calls.")
    # 2026-06-01 NEW: Smart Defensive Flip
    parser.add_argument("--smart-flip-enabled", action="store_true",
                        help="Enable the Smart Defensive Flip feature. After a HC/STRONG primary "
                             "position has its SL fired, the daemon evaluates buying the OPPOSITE "
                             "side as a defensive recovery position. Eligibility gates: opp_bid in "
                             "[50,75]\u00a2 sweet-spot band, >=5min remaining, signals support "
                             "continuation. Default OFF (explicit opt-in for safety).")
    # NOTE: --smart-flip-tiers is defined later (default high_conv,strong,late_sure)
    parser.add_argument("--smart-flip-min-opp-entry", type=int, default=50,
                        help="Min opposite-side bid (\u00a2) at SL moment for flip eligibility "
                             "(default 50). Below this, BTC has barely crossed the strike and "
                             "reversal is too likely.")
    parser.add_argument("--smart-flip-max-opp-entry", type=int, default=75,
                        help="Max opposite-side bid (\u00a2) at SL moment for flip eligibility "
                             "(default 75). Above this, insufficient upside to the sell target "
                             "(math breaks down).")
    parser.add_argument("--smart-flip-recovery-ratio", type=float, default=0.50,
                        help="Target fraction of primary loss to recover via flip (default 0.50 "
                             "= 50%%). Conservative \u2014 don't try to recover full loss because "
                             "that requires huge flip sizes that amplify cross-strike-twice risk.")
    parser.add_argument("--smart-flip-sl-cents", type=int, default=15,
                        help="Tight stop-loss for the flip position (default 15\u00a2). If flip "
                             "bid drops this much below entry, flip is sold immediately to limit "
                             "cross-strike-twice damage.")
    parser.add_argument("--smart-flip-sell-target", type=int, default=89,
                        help="Sell-target bid (\u00a2) for the flip position (default 89\u00a2). "
                             "Conservative; avoids settlement-timing risk and edge-of-orderbook "
                             "liquidity at 95+.")
    parser.add_argument("--smart-flip-trail-cents", type=int, default=10,
                        help="Trail-stop in cents (default 10). If flip bid drops this many cents "
                             "from observed peak, sell immediately. Catches reversals before they "
                             "wipe out flip profit.")
    parser.add_argument("--smart-flip-max-capital-usd", type=float, default=100.0,
                        help="Hard cap on flip capital deployment (default $100). Prevents "
                             "runaway flip sizes on big primary losses.")
    parser.add_argument("--smart-flip-min-mins-remaining", type=float, default=5.0,
                        help="Minimum minutes remaining at SL moment for flip to fire (default "
                             "5.0). Need enough time for flip to fill, run, and exit before "
                             "settlement.")
    parser.add_argument("--no-smart-flip-futures-confirm", action="store_true",
                        help="Disable the futures-confirmation gate. By default, the flip requires "
                             "the lead-venue futures to have moved >X%% in the FLIP direction over "
                             "the last Y seconds (i.e., continuation of primary-adverse).")
    parser.add_argument("--smart-flip-futures-confirm-pct", type=float, default=0.10,
                        help="Required futures move (%%) in the flip direction over the lookback "
                             "window (default 0.10). Higher = stricter continuation requirement.")
    parser.add_argument("--smart-flip-futures-window-s", type=float, default=30.0,
                        help="Lookback window (seconds) for the futures continuation check "
                             "(default 30).")
    parser.add_argument("--smart-flip-poll-s", type=float, default=2.0,
                        help="Poll interval (seconds) for the flip-position monitor (default 2.0).")
    parser.add_argument("--smart-flip-retry-attempts", type=int, default=3,
                        help="Number of times to retry flip evaluation if first attempt fails any "
                             "eligibility gate (default 3). Catches 'just-missed' cases where "
                             "opposite bid or futures clears within 30-60s post-SL. Set to 1 to "
                             "disable retries (one-and-done).")
    parser.add_argument("--smart-flip-retry-sleep-s", type=float, default=15.0,
                        help="Seconds to wait between flip retry attempts (default 15).")
    parser.add_argument("--smart-flip-tiers", type=str, default="high_conv,strong,late_sure",
                        help="Comma-separated tier names eligible for flip (default: "
                             "high_conv,strong,late_sure). LATE-SURE included 2026-06-01 \u2014 the "
                             "opp_bid band gate [50,75] already filters out deep-ITM cases.")
    # 2026-06-01 NEW: Hurst + TP disagreement veto (Gate B)
    parser.add_argument("--hurst-tp-veto-enabled", action="store_true", default=True,
                        help="Enable Gate B (ON by default in the canonical baseline) "
                             "\u2014 the Hurst + TP-Markov disagreement veto. "
                             "Blocks HC/STRONG trades when Hurst is HIGH (strong-trending regime) "
                             "AND TermProb disagrees with Markov by >= the threshold in the adverse "
                             "direction. Catches the 'chasing a fading top' pattern where Markov "
                             "lags the price reversal but TP (options market) has priced it in. "
                             "7-day backtest: catches 3 disasters (-$402), blocks 3 small wins "
                             "(+$37), net +$365/week.")
    parser.add_argument("--no-hurst-tp-veto", dest="hurst_tp_veto_enabled", action="store_false",
                        help="Disable the Hurst+TP disagreement veto.")
    parser.add_argument("--hurst-tp-veto-min-hurst", type=float, default=0.80,
                        help="Min Hurst exponent to trigger the veto (default 0.80). Hurst >= "
                             "0.80 indicates strong trending behavior. Lower = catches more "
                             "trades; higher = catches fewer.")
    parser.add_argument("--hurst-tp-veto-min-diff", type=float, default=0.05,
                        help="Min |TP - Markov| adverse magnitude to trigger the veto (default "
                             "0.05). 'Adverse' = TP is less bullish than Markov for YES, OR TP "
                             "is more bullish than Markov for NO. Lower = catches more trades.")
    parser.add_argument("--hurst-tp-veto-tiers", type=str, default="high_conv,strong",
                        help="Comma-separated tier names eligible for this veto. Default "
                             "high_conv,strong. The standard tier already has tight EV gates "
                             "and rarely fires at high Hurst, so excluded by default.")
    # 2026-06-01: Directional max-adverse-bar veto
    parser.add_argument("--max-adverse-bar-veto-enabled", action="store_true", default=True,
                        help="Enable the directional max-adverse-bar veto (ON by default in the "
                             "canonical baseline). Blocks trade direction "
                             "X when any of the recent 5m BTC bars moved >= threshold AGAINST X. "
                             "DIRECTIONAL: a -0.40%% bar blocks YES trades (BTC dropped, don't "
                             "chase the bounce) but does NOT block NO trades (NO is going WITH "
                             "the recent crash). 7-day backtest at 0.30%%: catches 2 disasters "
                             "(-$226), blocks 1 win (+$1.28), net +$225/week.")
    parser.add_argument("--no-max-adverse-bar-veto", dest="max_adverse_bar_veto_enabled",
                        action="store_false",
                        help="Disable the directional max-adverse-bar veto.")
    parser.add_argument("--max-adverse-bar-veto-pct", type=float, default=0.30,
                        help="Threshold for the max-adverse-bar veto (default 0.30%%). Any 5m bar "
                             "in the recent window that moved AGAINST the trade direction by >= "
                             "this magnitude triggers the veto.")
    parser.add_argument("--max-adverse-bar-veto-tiers", type=str, default="high_conv,strong",
                        help="Comma-separated tier names eligible for this veto. Default "
                             "high_conv,strong. Standard tier excluded by default since it has "
                             "smaller positions where the cost is less catastrophic.")
    # â”€â”€ 2026-06-02: Cumulative adverse momentum veto â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Catches "slow steady drift against the trade" patterns missed by the
    # max-adverse-BAR veto (which only checks single-bar magnitude). The
    # 2026-06-02 05:09 NO trade lost -$301 with adverse_sum=+0.36%% but no
    # single bar above 0.18%%; orderbook never repriced; SL never fired.
    parser.add_argument("--no-cum-adverse-momentum-veto", action="store_true",
                        help="Disable the cumulative adverse momentum veto. Default ENABLED. "
                             "The veto blocks trade direction X when the SUM of recent_5m_pct "
                             "bars in the direction adverse to X meets/exceeds the threshold. "
                             "Catches multi-bar slow drift that the per-bar veto misses. "
                             "7-day backtest @ 0.35%%: blocks 5 trades (4W/1L), saves $301 "
                             "in losses, $28 in forgone wins, net +$273/week.")
    parser.add_argument("--cum-adverse-momentum-veto-pct", type=float, default=0.35,
                        help="Threshold for the cumulative adverse momentum veto (default 0.35%%). "
                             "If sum(recent_5m_pct) in the ADVERSE direction is >= this, block "
                             "the trade. 0.30%% is more aggressive (catches 9 trades over 7d, "
                             "+$252 net); 0.40%% misses today's failure mode (no losses caught).")
    parser.add_argument("--cum-adverse-momentum-veto-tiers", type=str,
                        default="high_conv,strong,late_sure",
                        help="Comma-separated tier names eligible for this veto. Default "
                             "high_conv,strong,late_sure. Standard tier excluded by default "
                             "(smaller positions, lower cost per failure).")
    # â”€â”€ 2026-06-03: TIER 4 â€” LOW-HURST HC veto â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Block HC trades in mean-reverting regimes where extreme Markov is a trap.
    # 2026-06-03 11:22 NO @ 97c (Markov=0.031, Hurst=0.27) lost $43 with no
    # hedge protection. This veto blocks the same pattern at entry.
    parser.add_argument("--no-hc-low-hurst-veto", action="store_true", default=True,
                        help="Disable the LOW-HURST HC veto (DISABLED by default in the canonical "
                             "baseline; use --hc-low-hurst-veto to enable). When ON, "
                             "HC trades are blocked when Hurst < threshold (default 0.30) "
                             "AND |Markov-0.5| \u2265 extremity (default 0.35). Catches "
                             "mean-reverting-regime traps where strong directional signals "
                             "get reversed by mean reversion. 7-day backtest: prevents "
                             "$300-500 in catastrophic losses.")
    parser.add_argument("--hc-low-hurst-veto", dest="no_hc_low_hurst_veto", action="store_false",
                        help="Enable the LOW-HURST HC veto (off in the canonical baseline).")
    parser.add_argument("--hc-low-hurst-threshold", type=float, default=0.30,
                        help="Hurst threshold for the low-hurst veto. When Hurst is BELOW "
                             "this value, BTC is in a mean-reverting regime. Default 0.30 "
                             "(strict mean-reverting territory). Lower = stricter veto.")
    parser.add_argument("--hc-low-hurst-markov-extremity", type=float, default=0.35,
                        help="Markov extremity threshold (|p_yes - 0.5|) for the low-hurst "
                             "veto. Only fires when the directional signal is extreme. "
                             "Default 0.35 \u2192 fires when Markov \u2264 0.15 or \u2265 0.85.")
    # â”€â”€ 2026-06-03: HC distance-from-strike floor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Blocks HC trades when BTC is too close to strike (where reversals can
    # easily flip the contract). 9-day backtest of strict HC (gap\u22650.40,
    # persist\u22650.95, tp\u22650.90) combined with |dist|\u22650.25% gave 13/13 wins,
    # +$127 PnL (+$98/wk). Default 0.0 = filter off; common values:
    #   0.20 (loose) | 0.25 (recommended) | 0.30 (very strict)
    parser.add_argument("--high-conv-dist-min", type=float, default=0.25,
                        help="Minimum |dist_pct| (BTC vs strike, in percent) required "
                             "for HC trades. When |dist_pct| is below this floor, BTC "
                             "is too close to strike and HC's extreme bet is too risky. "
                             "Backtest: 0.25%% gives 100%% WR with strict HC filters. "
                             "Default 0.25 (canonical baseline); 0.0 turns the filter off.")
    # â”€â”€ 2026-06-03: tunable STANDARD price caps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Backtest of 9-day audit data showed STANDARD trades at 66-75\u00a2 NO entries
    # are net-profitable (100%% WR on the observed sample). Current 65\u00a2 cap
    # leaves money on the table. CLI flags allow tuning without editing
    # run_backtest.py constants.
    parser.add_argument("--standard-price-cap-no", type=int, default=88,
                        help="Override the STANDARD tier NO price cap. Default uses the "
                             "constant from run_backtest.py (currently 65\u00a2). 9-day backtest: "
                             "raising to 70 adds ~$200/wk, 72 adds ~$320/wk, 75 adds ~$410/wk "
                             "(estimates assume observed 100%% WR holds; conservative WR=74%% "
                             "lower bound still gives positive EV).")
    parser.add_argument("--standard-price-cap-yes", type=int, default=88,
                        help="Override the STANDARD tier YES price cap. Default uses the "
                             "constant from run_backtest.py (currently 72\u00a2). 9-day backtest: "
                             "minimal new edge on YES side (\u22481-2 trades/wk).")
    # â”€â”€ 2026-06-03: FADE-BOUNCE dual-entry tier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Captures the "fade the false reversal" pattern: when the Polymarket bid for
    # our side dips into the 40-55Â¢ band mid-window, 85%% of those windows
    # still settle in the original direction (7-day backtest of 73 windows).
    # When fired, a SECOND smaller position is bought at the discount price.
    # Hard-capped by --fade-bounce-max-capital-usd (default $20/attach).
    parser.add_argument("--fade-bounce-enabled", action="store_true",
                        help="Enable FADE-BOUNCE dual-entry. Default DISABLED. When ON, the "
                             "daemon will buy additional contracts on the same side when the "
                             "bid dips into the configured range mid-window. Hard-capped "
                             "by --fade-bounce-max-capital-usd.")
    parser.add_argument("--fade-bounce-no-ask-min", type=int, default=40,
                        help="Lower bound of the discount band (default 40\u00a2).")
    parser.add_argument("--fade-bounce-no-ask-max", type=int, default=55,
                        help="Upper bound of the discount band (default 55\u00a2).")
    parser.add_argument("--fade-bounce-yes-side-enabled", action="store_true",
                        help="Allow YES-side fade-bounce too. Default OFF \u2014 the 7-day audit "
                             "showed YES bets in this band had 0%% WR (sample bias likely).")
    parser.add_argument("--fade-bounce-markov-no-max", type=float, default=0.45,
                        help="For NO-side fade-bounce: cached entry Markov P(YES) must be \u2264 "
                             "this value (default 0.45 \u2192 entry signal still favors NO).")
    parser.add_argument("--fade-bounce-markov-yes-min", type=float, default=0.55,
                        help="For YES-side fade-bounce (if enabled): cached Markov must be \u2265 "
                             "this value. Default 0.55.")
    parser.add_argument("--fade-bounce-hurst-min", type=float, default=0.50,
                        help="Minimum cached Hurst to qualify (default 0.50 \u2014 avoid extreme "
                             "mean-reverting regimes where the signal direction is fragile).")
    parser.add_argument("--fade-bounce-dist-min", type=float, default=0.03,
                        help="Minimum |dist_pct| for cached entry (default 0.03%%). Ensures "
                             "BTC was at least slightly on the favored side at primary entry.")
    parser.add_argument("--fade-bounce-min-mins", type=float, default=3.0,
                        help="Skip fade-bounce when fewer than this many mins remain "
                             "(default 3.0 \u2014 too close to settle, no time to absorb).")
    parser.add_argument("--fade-bounce-max-mins", type=float, default=12.0,
                        help="Skip fade-bounce when more than this many mins remain "
                             "(default 12.0 \u2014 too early, primary trade hasn't settled in).")
    parser.add_argument("--fade-bounce-min-stake-pct", type=float, default=0.015,
                        help="Floor stake (fraction of bankroll) for fade-bounce position. "
                             "Default 0.015 = 1.5%% \u2192 ~$45 at $3000 bankroll.")
    parser.add_argument("--fade-bounce-kelly-frac", type=float, default=0.05,
                        help="Kelly-fraction cap for fade-bounce (default 0.05 \u2014 smaller than "
                             "other tiers since variance is high).")
    parser.add_argument("--fade-bounce-sl-cents", type=int, default=20,
                        help="SL trigger for the fade-bounce leg (default 20\u00a2). Lighter than "
                             "primary because entry is already at a discount.")
    parser.add_argument("--fade-bounce-max-capital-usd", type=float, default=20.0,
                        help="HARD CAP on capital deployed per fade-bounce attach "
                             "(default $20). Strongly recommend keeping low during initial "
                             "validation \u2014 e.g., $20 = max ~$20 loss if the rare 15%% "
                             "reversal case fires and contract settles to $0.")
    parser.add_argument("--sl-trigger-mode", type=str, default="mid",
                        choices=["bid", "mid", "last"],
                        help="Price to use for SL trigger check. 'bid' (conservative â€” "
                             "matches actual exit price), 'mid' (default â€” average of bid+ask, "
                             "reacts faster), 'last' (last-traded). Selling always happens at the bid.")
    parser.add_argument("--no-sl-aggressive-sell", action="store_true",
                        help="Disable aggressive sell on SL exits. Default: ENABLED. "
                             "Aggressive sell submits IOC at extreme limit ($0.01 ask for YES, "
                             "$0.99 bid for NO) so Polymarket fills against ANY remaining bid in "
                             "the book â€” guarantees fill, prevents chasing a falling bid. "
                             "Today's data showed conservative-bid sells losing ~$22 to slippage "
                             "across two SL events; aggressive sell eliminates that.")
    parser.add_argument("--no-hc-block-on-split", action="store_true",
                        help="Disable the HIGH-CONV split-externals block. Default: ENABLED. "
                             "Rule: when about to qualify a trade for HIGH-CONV tier, check "
                             "binance_futures + OKX move directions. If ONE opposes the trade direction "
                             "and the other doesn't (XOR: split signal), block HIGH-CONV "
                             "qualification â€” trade falls through to lower tiers (which usually "
                             "means NO_TRADE if price is too high for them). Today's data: "
                             "would have prevented -$50.46 SL loss and missed +$10.40 win = "
                             "+$40 net. Only affects HIGH-CONV tier â€” STANDARD/strong/late-* "
                             "tiers unaffected because their lower price ceilings provide more "
                             "EV cushion against split-external noise.")
    parser.add_argument("--standard-confirmed-boost", type=float, default=1.5,
                        help="Multiplier applied to Kelly fraction for non-HC tiers (STANDARD, "
                             "strong-floor, late-sure, late-dir) when BOTH external feeds confirm "
                             "direction (neither opposes). Default 1.5 (lowered from 2.0 on "
                             "2026-05-27 PM after a -$119 loss where 2x sizing on a strong-floor "
                             "trade amplified damage). MAX_TRADE_PCT cap (20%%) still applies; "
                             "set to 1.0 to disable boost entirely.")
    parser.add_argument("--high-conv-confirmed-frac", type=float, default=0.10,
                        help="Kelly fraction applied to HIGH-CONV tier when BOTH "
                             "external feeds (binance_futures + OKX) confirm direction "
                             "(neither opposes). Default 0.10 = 2x the anti-Kelly "
                             "baseline of 0.05 for 86Â¢+ trades. Historical 5/26-5/27 "
                             "analysis: all 7 HIGH-CONV WINS had both-confirm; the 1 "
                             "LOSS had both-oppose (now blocked by consensus veto). "
                             "Set to 0.05 to disable the boost (revert to baseline).")
    parser.add_argument("--ev-gate",            action="store_true", default=True,
                        help="Replace flat YES/NO price caps with EV-based override (when only "
                             "price cap blocks). ON by default in the canonical baseline.")
    parser.add_argument("--no-ev-gate", dest="ev_gate", action="store_false",
                        help="Disable the EV gate (revert to flat price caps).")
    parser.add_argument("--ev-floor",           type=float, default=0.05,
                        help="Minimum EV per contract (in $) for EV-gate override (default 0.05)")
    parser.add_argument("--ev-ceiling",         type=int,   default=90,
                        help="Hard price ceiling regardless of EV (cents, default 90)")
    parser.add_argument("--ev-tp-weight",       type=float, default=0.5,
                        help="Weight of TP vs Markov in combined probability for EV gate (default 0.5)")
    parser.add_argument("--ev-strong-floor",    type=float, default=0.0,
                        help="Relaxed EV floor (in $) when strong-signal criteria hold (default 0.0)")
    parser.add_argument("--ev-strong-gap-min",  type=float, default=0.13,
                        help="Min Markov gap to qualify for strong-floor (default 0.13)")
    parser.add_argument("--ev-strong-price-max",type=int,   default=88,
                        help="Max entry price (cents) to qualify for strong-floor (default 88, "
                             "raised from 85â†’88 on 2026-05-28). On choppy days price_cap is "
                             "the top blocker; widening to 88 lets walking retry fill at 86-88Â¢ "
                             "on developing-conviction trades that don't quite reach HIGH-CONV.")
    parser.add_argument("--ev-strong-tp-min",   type=float, default=0.55,
                        help="Minimum TP (Deribit terminal probability) directional confirmation "
                             "for strong-floor qualification (default 0.55). For YES trades TP "
                             "must be â‰¥0.55; for NO trades TP must be â‰¤0.45 (= 1-0.55). Previously "
                             "the gate just required TP same-direction as Markov, which allowed "
                             "TP at 50.1%% to qualify â€” too marginal. 2026-05-27 21:24 loss had "
                             "TP=52.7%% and was a dead-cat bounce; this tighter threshold filters "
                             "such cases. Set to 0.50 to revert to old same-direction-only logic.")
    parser.add_argument("--ev-strong-max-mins", type=float, default=8.0,
                        help="Max mins_left allowed for strong-floor tier (default 8.0). Strong-"
                             "floor was firing at 11-12 min on 2026-05-27 PM, catching mid-market "
                             "Markov bounces that reverse before close. Capping to 8 min forces "
                             "signal to develop further before commitment. Today's loss cluster "
                             "(-$163 on two trades at 11-12 mins left) would have been blocked.")
    parser.add_argument("--ev-strong-max-adverse-momentum", type=float, default=0.10,
                        help="Maximum adverse cumulative recent_5m_pct%% allowed for strong-floor "
                             "(default 0.10 = 0.10%%). Blocks strong-floor when net 6-bar momentum "
                             "is opposite trade direction by more than this. Today's losses both "
                             "had net -0.21%% over 6 bars but were YES trades â€” Markov caught a "
                             "brief bounce in the last 2 bars and missed the underlying downtrend. "
                             "This gate catches 'dead-cat bounce' setups.")
    parser.add_argument("--last-bar-adverse-threshold", type=float, default=0.10,
                        help="LAST 5-min bar adverse-direction threshold (default 0.10 = 0.10%%). "
                             "Universal gate applied to ALL trades: if the MOST RECENT 5m bar "
                             "moved against the trade direction by more than this, block the "
                             "trade. Catches late-stage reversal patterns where Markov is "
                             "averaging over multiple bars and missing the very recent shift. "
                             "2026-05-27 22:07 loss (NO @ 73Â¢): last bar was +0.14%% just as BTC "
                             "started recovering â€” this gate would have BLOCKED it. Treated as "
                             "TRANSIENT (re-polls until bar boundary crosses or trade fires). "
                             "Set to 0 to disable.")
    parser.add_argument("--no-orderbook-signal", dest="orderbook_signal_enabled",
                        action="store_false",
                        help="Disable Polymarket orderbook depth recording (default ON, audit-only). "
                             "Captures top-of-book + depth + imbalance per poll for retrospective "
                             "analysis. Added 2026-05-28 â€” after 1-2 days of data we can validate "
                             "predictive features (imbalance, spread, depth) before promoting to gates.")
    parser.add_argument("--no-trade-flow-signal", dest="trade_flow_signal_enabled",
                        action="store_false",
                        help="Disable Polymarket recent-trades recording (default ON, audit-only). "
                             "Captures taker aggression direction (yes_taker_volume vs no_taker_volume) "
                             "from the last N trades. Complements orderbook depth: orderbook shows "
                             "resting INTENT, trade flow shows actual EXECUTION pressure.")
    parser.add_argument("--trade-flow-lookback-n", type=int, default=20,
                        help="Number of recent Polymarket trades to summarize per poll (default 20). "
                             "Trade-flow aggression is computed across this window: more trades = "
                             "smoother metric but slower to react to fresh aggression bursts.")
    parser.add_argument("--taker-flow-veto-enabled", action="store_true", default=True,
                        help="Block entries when recent Polymarket taker aggression is at or "
                             "above --taker-flow-veto-agg-min AGAINST the bet AND the price is "
                             "within --taker-flow-veto-dist-max of strike (near-strike reversal "
                             "trap). BTC backtest: +$1458 over 20 audit files. ON by default "
                             "in the canonical baseline.")
    parser.add_argument("--no-taker-flow-veto", dest="taker_flow_veto_enabled", action="store_false",
                        help="Disable the taker-flow entry veto.")
    parser.add_argument("--taker-flow-veto-agg-min", type=float, default=0.90,
                        help="Taker-flow veto: fraction of recent taker volume against the bet "
                             "to trigger (default 0.90).")
    parser.add_argument("--taker-flow-veto-dist-max", type=float, default=0.25,
                        help="Taker-flow veto only applies within this distance-from-strike "
                             "(percent) magnitude (default 0.25).")
    parser.add_argument("--taker-flow-veto-min-trades", type=int, default=5,
                        help="Min recent trades for the taker-flow veto to fire (default 5).")
    parser.add_argument("--retry-walk-cents",   type=int, default=8,
                        help="On zero-fill chunked IOC, walk price up to N cents (default 8 per the "
                             "canonical baseline). Each step re-checks EV vs applicable floor; stops on fill / "
                             "floor-breach / ceiling. Wider walk catches gaps in the bid stack at fast-"
                             "moving HC-lockin prices.")
    parser.add_argument("--retry-plus-cent",    action="store_true",
                        help="Back-compat alias for --retry-walk-cents 1")
    parser.add_argument("--max-window-fill-attempts", type=int, default=10,
                        help="Max IOC retry attempts per window when first fill misses (default 10, "
                             "raised from 5 on 2026-05-29). Daemon re-evaluates signal on each attempt; "
                             "gives up when cap is hit. More attempts = more chances to catch the next "
                             "orderbook refresh.")
    parser.add_argument("--refill-retry-sleep-s",     type=float, default=10.0,
                        help="Seconds to wait between retry attempts within a window (default 10s, "
                             "lowered from 30 on 2026-05-29). 30s was too long for short 15min windows "
                             "\u2014 a 30s sleep after a failed fill misses ~3 normal poll opportunities. "
                             "10s gives time for the orderbook to refresh while still keeping the daemon "
                             "responsive. With --max-window-fill-attempts 10, you get 10\u00d710s = 100s "
                             "of retry coverage, more than enough for most window lifetimes.")
    parser.add_argument("--late-window-mins",      type=float, default=0.0,
                        help="Mins-left threshold for late-window-sure tier (default 0.0 = "
                             "LATE-SURE tier disabled, per the canonical baseline)")
    parser.add_argument("--late-window-price-max", type=int,   default=98,
                        help="Max bid price (Â¢) under late-window-sure tier (default 98). "
                             "2026-05-28: raised 96 â†’ 98 after 3-day audit found EVERY late-window "
                             "lock-in candidate (104 polls, mins<5 with TP+Markov agreed) was "
                             "asking 97-100Â¢. Cap at 96 made LATE-SURE unreachable. With the "
                             "-$0.10/c EV floor an entry at 98Â¢ self-filters to p_win â‰¥ 0.90.")
    parser.add_argument("--no-late-sure-vol-bypass", dest="late_sure_vol_bypass",
                        action="store_false",
                        help="Disable LATE-SURE bypassing the high-vol gate (default ON). "
                             "2026-05-28: LATE-SURE's own gates (mins<5 + TP-direction + Markov "
                             "agreement) already guarantee strong directional conviction, so the "
                             "vol-regime caution is redundant for late-window lock-ins. Pre-change "
                             "behavior: only HIGH-CONV / LATE-DIR could bypass vol.")
    parser.add_argument("--no-orderbook-lockin", dest="orderbook_lockin_enabled",
                        action="store_false",
                        help="Disable orderbook-confirmed lock-in bypasses (default ON, 2026-05-28 PM). "
                             "Lock-in detection: spread\u2264N AND top-of-book in trade direction \u2265N\u00a2 "
                             "AND Markov gap \u2265N. When confirmed, (a) HIGH-CONV bypasses its TP "
                             "threshold and (b) LATE-SURE\u2019s effective cap rises 98 \u2192 99. Orthogonal "
                             "confirmation from Polymarket\u2019s own microstructure compensates for Deribit\u2019s "
                             "13h IV undershoot on 15min binaries.")
    parser.add_argument("--orderbook-lockin-spread-max", type=int, default=2,
                        help="Max spread (\u00a2) to qualify as lock-in (default 2). Spread \u2264 this AND "
                             "top-of-book in trade direction \u2265 --orderbook-lockin-price-min defines lock-in.")
    parser.add_argument("--orderbook-lockin-price-min", type=int, default=85,
                        help="Min top-of-book bid (\u00a2) in trade direction for lock-in (default 85, "
                             "lowered from 95 \u2192 90 \u2192 85 on 2026-05-28 across the day). 95 was too strict, "
                             "90 still missed moderate-conviction markets. At 85 market still implies 85%% "
                             "confidence; spread\u22642 + gap\u22650.20 provide additional confirmation.")
    parser.add_argument("--orderbook-lockin-gap-min", type=float, default=0.20,
                        help="Min Markov gap to combine with orderbook for lock-in confirmation (default 0.20, "
                             "lowered from 0.30 on 2026-05-28 PM2 to match the new --high-conv-gap-min default. "
                             "Symmetric so HC's lock-in bypass works for the new 0.20-0.30 gap range.")
    parser.add_argument("--late-window-price-max-lockin", type=int, default=99,
                        help="LATE-SURE effective cap (\u00a2) when orderbook lock-in is confirmed (default 99 "
                             "\u2014 1\u00a2 above the standard 98). At 99\u00a2 a fill needs ~99%% accuracy to break "
                             "even; the orderbook confirmation provides this confidence band.")
    parser.add_argument("--hc-lockin-ev-floor", type=float, default=-0.10,
                        help="HIGH-CONV EV floor ($/c) when orderbook lock-in is confirmed (default -0.10, "
                             "matching LATE-SURE). When lock-in confirms direction the standard tighter "
                             "HC floor (default -0.05 via --high-conv-ev-floor) is too conservative â€” the "
                             "orderbook is doing the work TP would otherwise do.")
    parser.add_argument("--hc-lockin-min-stake-pct", type=float, default=0.015,
                        help="Floor-stake (%% of bankroll) for HC trades fired under the lock-in path "
                             "when Kelly returns 0 (i.e., slight negative EV within the -$0.10 floor). "
                             "Default 0.015 (1.5%%, matches LATE-SURE). Without this, lock-in HC fires "
                             "only 1 contract because Kelly is 0 on negative EV.")
    parser.add_argument("--no-strong-floor-hurst-bypass", dest="strong_floor_hurst_bypass",
                        action="store_false",
                        help="Disable STRONG-FLOOR bypassing the Hurst (mean-reverting) gate. Default ON "
                             "(2026-05-28 PM3). When ON, strong-floor can fire on Hurst-blocked moderate "
                             "setups when its own gates hold (gap\u22650.13, price\u226488, TP-meaningful, mins\u22648, "
                             "direction match). Combine with --ev-strong-floor -0.10 for wider EV cushion.")
    parser.add_argument("--late-window-min-tp",    type=float, default=0.75,
                        help="TP probability threshold confirming direction (default 0.75, "
                             "lowered from 0.85 on 2026-05-28). Deribit's options expire 13+ hours "
                             "out â€” BS-based TP applied to a 3-min binary structurally undershoots "
                             "true probability. Audit confirmed TP_bs clusters at 0.65-0.80 in the "
                             "late window even when Polymarket markets show 85-98%% consensus. The 0.85 "
                             "threshold made LATE-SURE unreachable (0 fires in 5 days).")
    parser.add_argument("--late-window-ev-floor",  type=float, default=-0.10,
                        help="EV floor ($ per contract) under late-window-sure tier (default -0.10)")
    parser.add_argument("--strong-floor-min-stake-pct", type=float, default=0.020,
                        help="Min bankroll fraction to risk on strong-floor approved trades when "
                             "Kelly=0 (default 0.020 = 2.0%%). Prevents 1-contract floor on negative-EV approvals.")
    parser.add_argument("--late-sure-min-stake-pct",    type=float, default=0.015,
                        help="Min bankroll fraction to risk on late-window-sure trades when Kelly=0 "
                             "(default 0.015 = 1.5%%)")
    parser.add_argument("--standard-min-entry-contracts", type=int, default=2,
                        help="Skip STANDARD-tier entries Kelly-sized below this "
                             "many contracts (default 2 => skip 1-contract token "
                             "entries). The window is NOT marked traded, so it "
                             "stays open for a better-priced or high-conv entry. "
                             "Set 1 to disable.")
    parser.add_argument("--golden-price-lo", type=int, default=65,
                        help="Golden-zone price band LOW (Â¢). Default 65. "
                             "Backtest favors 55.")
    parser.add_argument("--golden-price-hi", type=int, default=73,
                        help="Golden-zone price band HIGH (Â¢). Default 73. "
                             "Backtest favors 80 (85+ degrades).")
    parser.add_argument("--golden-no-dist", action="store_true",
                        help="Drop the near-strike distance gate FOR golden-zone "
                             "entries (the 65-73Â¢ band is itself the conviction "
                             "filter). Backtest: +volume, WR holds 80%%+.")
    parser.add_argument("--golden-no-hurst", action="store_true",
                        help="Drop the mean-reverting Hurst gate FOR golden-zone "
                             "entries. Backtest: minimal WR impact, +volume.")
    parser.add_argument("--okx-boost-enabled", action="store_true", default=True,
                        help="Nudge combined_p toward OKX 6s futures momentum (the "
                             "one auxiliary signal that stably predicts the model's "
                             "residual: corr +0.09, +5pp split, both temporal "
                             "halves). ON by default in the canonical baseline.")
    parser.add_argument("--no-okx-boost", dest="okx_boost_enabled", action="store_false",
                        help="Disable the OKX momentum boost.")
    parser.add_argument("--okx-boost-weight", type=float, default=0.06,
                        help="Max combined_p nudge from OKX boost (default 0.06 "
                             "= Â±6pp). 0 disables.")
    parser.add_argument("--okx-boost-scale", type=float, default=0.015,
                        help="OKX 6s move%% at which the boost ~saturates "
                             "(default 0.015).")
    parser.add_argument("--high-conv-gap-min",     type=float, default=0.40,
                        help="Min Markov gap for high-conviction tier (default 0.40 per the "
                             "canonical baseline â€” strict HC filter, ~70/30 Markov confidence).")
    parser.add_argument("--high-conv-persist-min", type=float, default=0.95,
                        help="Min Markov persistence for high-conviction tier (default 0.95)")
    parser.add_argument("--high-conv-tp-strong",   type=float, default=0.90,
                        help="TP probability threshold confirming direction for high-conv "
                             "(default 0.90 â†’ TP â‰¥ 90%% for YES or â‰¤ 10%% for NO). "
                             "The ChainVector probability engine computes exact-TTE 15m "
                             "probabilities (no 13h-IV undershoot), so the strict 0.90 "
                             "threshold is reachable.")
    parser.add_argument("--high-conv-price-max",   type=int,   default=97,
                        help="Max bid (Â¢) under high-conviction tier (default 97)")
    parser.add_argument("--high-conv-ev-floor",    type=float, default=0.005,
                        help="EV floor ($/c) under high-conviction tier (default +0.005, positive-EV only)")
    parser.add_argument("--high-conv-max-mins",    type=float, default=13.0,
                        help="Max mins_left allowed for high-conv tier (default 13.0). "
                             "Allows extending past the standard 10-min entry cap when signals "
                             "are extreme. Set to 10.0 to disable timing bypass.")
    parser.add_argument("--no-high-conv-vol-bypass", action="store_true",
                        help="Disable high-conv's ability to bypass the vol gate. By default "
                             "high-conv can bypass vol when BTC is far from strike and moving "
                             "directionally in our favor (directional vol, not noisy vol).")
    parser.add_argument("--high-conv-vol-bypass-momentum", type=float, default=0.10,
                        help="Min |recent 5m return|%% in trade direction to bypass vol (default 0.10)")
    parser.add_argument("--high-conv-vol-bypass-distance", type=float, default=0.15,
                        help="Min |BTC - strike|/strike * 100 (%%) to bypass vol (default 0.15)")
    parser.add_argument("--high-conv-vol-bypass-strong-distance", type=float, default=0.25,
                        help="If |BTC-strike|/strike * 100 â‰¥ this %%, skip the 5m momentum check "
                             "(distance alone is directional signal). Default 0.25.")
    parser.add_argument("--no-late-dir", action="store_true",
                        help="Disable the LATE-DIR tier (late-window-directional vol bypass). "
                             "Default: enabled.")
    parser.add_argument("--late-dir-mins",            type=float, default=5.0,
                        help="Max mins_left for late-dir tier (default 5.0)")
    parser.add_argument("--late-dir-gap-min",         type=float, default=0.25,
                        help="Min Markov gap for late-dir tier (default 0.25)")
    parser.add_argument("--late-dir-persist-min",     type=float, default=0.95,
                        help="Min Markov persist for late-dir tier (default 0.95)")
    parser.add_argument("--late-dir-distance-min",    type=float, default=0.05,
                        help="Min |BTC-strike|/strike (%%) for late-dir (default 0.05)")
    parser.add_argument("--late-dir-momentum-min",    type=float, default=0.05,
                        help="Min |recent 5m return|%% in trade direction (default 0.05). "
                             "Skipped when distance â‰¥ strong-distance threshold.")
    parser.add_argument("--late-dir-strong-distance", type=float, default=0.08,
                        help="If distance â‰¥ this %%, skip 5m momentum check (default 0.08)")
    parser.add_argument("--late-dir-ev-floor",        type=float, default=-0.05,
                        help="EV floor ($/c) for late-dir tier (default -0.05)")
    parser.add_argument("--late-dir-price-max",       type=int,   default=89,
                        help="Max bid price (Â¢) for late-dir tier (default 89)")
    parser.add_argument("--order-lead-cents",         type=int,   default=2,
                        help="Submit initial order at +NÂ¢ above observed orderbook ask "
                             "(default 2). 'Sweeps the book' to capture fast-moving orderbooks. "
                             "EV-checked vs strong-floor: leads are blocked if EV at +N would "
                             "fall below ev_strong_floor. Recommended: 2-3Â¢ to handle typical "
                             "1c/sec orderbook drift.")
    parser.add_argument("--no-okx", action="store_true",
                        help="Disable the OKX venue view of the ChainVector lead feed "
                             "(consensus-veto partner + OKX boost source). Default: enabled.")
    parser.add_argument("--okx-poll-interval-s", type=float, default=3.0,
                        help="ChainVector /momentum poll interval in seconds for the lead "
                             "feed (default 3.0 = 20 req/min, inside the Developer plan's "
                             "60/min with room for the rest of the signal stack).")
    # â”€â”€ 2026-06-09: Rolling-WR adaptive throttle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Detects regime shifts via observed rolling win rate. When the WR of the
    # last N trades drops below threshold, blocks "standard" and "strong"
    # tier trades (configurable) until the next win OR a timeout. HC and
    # late-window tiers (which require extreme signals) still fire.
    # Resets at restart (in-memory state). OFF by default \u2014 opt-in.
    parser.add_argument("--rolling-wr-enabled", action="store_true",
                        help="Enable rolling-WR adaptive throttle. When recent "
                             "win rate drops below threshold, blocks weaker "
                             "tiers until WR recovers or timeout expires.")
    parser.add_argument("--rolling-wr-window", type=int, default=5,
                        help="Number of most-recent trade outcomes to track "
                             "for the rolling WR calculation. Default 5.")
    parser.add_argument("--rolling-wr-threshold", type=float, default=0.40,
                        help="WR threshold (0.0-1.0). Defensive mode engages "
                             "when rolling WR \u2264 this. Default 0.40.")
    parser.add_argument("--rolling-wr-timeout-mins", type=float, default=120.0,
                        help="Max time in defensive mode (minutes) before "
                             "automatic exit. Default 120 min. 0 = no timeout.")
    parser.add_argument("--rolling-wr-defensive-tiers", type=str,
                        default="standard,strong",
                        help="Comma-separated tiers blocked during defensive "
                             "mode. Default 'standard,strong'. HC/late tiers "
                             "stay enabled. To skip all standard-and-below: "
                             "'standard,strong,late_dir'.")
    # â”€â”€ 2026-06-11: ADAPTIVE BANKROLL (risk throttle) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # On a loss cluster (drawdown >= trigger since reference) or weak rolling
    # WR, shrink position sizing to reduced-frac of bankroll instead of
    # stopping. All tiers keep trading (live data preserved); recovers to
    # 100% on demonstrated WR recovery. State is in-memory: a manual restart
    # resets to NORMAL. When enabled, the daily hard-stop extends to $700
    # (the throttle brakes first; the cap becomes the catastrophe floor).
    parser.add_argument("--adaptive-bankroll-enabled", action="store_true",
                        help="Enable the adaptive bankroll risk throttle.")
    parser.add_argument("--adaptive-br-reduced-frac", type=float, default=0.15,
                        help="Fraction of bankroll used while REDUCED (default 0.15).")
    parser.add_argument("--adaptive-br-loss-trigger-usd", type=float, default=300.0,
                        help="Drawdown (USD, from peak since reference) that "
                             "triggers REDUCED mode. Default 300.")
    parser.add_argument("--adaptive-br-wr-trigger", type=float, default=0.50,
                        help="Rolling WR floor that triggers REDUCED. Default 0.50.")
    parser.add_argument("--adaptive-br-wr-window-h", type=float, default=3.0,
                        help="Rolling WR window in hours. Default 3.")
    parser.add_argument("--adaptive-br-wr-min-trades", type=int, default=5,
                        help="Min trades in the WR window before it can trigger. Default 5.")
    parser.add_argument("--adaptive-br-recover-wr", type=float, default=0.75,
                        help="WR over the recovery window required to return "
                             "to 100%% sizing. Default 0.75.")
    parser.add_argument("--adaptive-br-recover-window", type=int, default=6,
                        help="Number of most-recent trades for the recovery WR. Default 6.")
    parser.add_argument("--adaptive-br-recover-min-wins", type=int, default=3,
                        help="Min wins while REDUCED before recovery allowed. Default 3.")
    # â”€â”€ 2026-06-12: PATIENT TOP-UP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    parser.add_argument("--patient-topup-enabled", action="store_true", default=True,
                        help="When a position filled below its intended size "
                             "(walk stopped at EV floor), keep watching the ask "
                             "for the rest of the window and top up to the "
                             "intended size when a FRESH signal approves at the "
                             "improved price. ON by default in the canonical baseline.")
    parser.add_argument("--no-patient-topup", dest="patient_topup_enabled", action="store_false",
                        help="Disable patient top-up.")
    parser.add_argument("--patient-topup-interval-s", type=float, default=20.0,
                        help="Min seconds between top-up attempts. Default 20.")
    parser.add_argument("--patient-topup-min-mins", type=float, default=2.5,
                        help="Stop attempting top-ups when less than this many "
                             "minutes remain. Default 2.5.")
    parser.add_argument("--patient-topup-dynamic-kelly", action="store_true",
                        help="Let the patient top-up grow the position toward "
                             "the FRESH Kelly size (recomputed at current price) "
                             "instead of capping at the frozen entry-time intent. "
                             "Only ever adds when fresh EV>0 â€” so a trade that "
                             "was Kelly-floored to ~1 contract at entry (EV "
                             "negative then) can build up if it later becomes "
                             "genuinely +EV at a better price.")
    # â”€â”€ 2026-06-15: PERP-CONFIRMED TAKE-PROFIT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    parser.add_argument("--take-profit-enabled", action="store_true", default=True,
                        help="Exit at entry + --take-profit-cents. ON by default in the "
                             "canonical baseline (with --take-profit-all-trades).")
    parser.add_argument("--no-take-profit", dest="take_profit_enabled", action="store_false",
                        help="Disable the take-profit exit.")
    parser.add_argument("--take-profit-cents", type=int, default=10,
                        help="Profit target in cents above entry. Default 10.")
    parser.add_argument("--take-profit-all-trades", action="store_true", default=True,
                        help="Apply TP to ALL trades, not just perp-confirmed. ON by default "
                             "in the canonical baseline.")
    parser.add_argument("--take-profit-perp-confirmed-only", dest="take_profit_all_trades",
                        action="store_false",
                        help="Gate the TP to perp-momentum-confirmed entries only.")
    parser.add_argument("--take-profit-perp-min", type=float, default=0.0,
                        help="Min entry perp_m30s (bp) to arm the TP when "
                             "perp-confirmed-only. Default 0.0.")
    parser.add_argument("--resting-tp-enabled", action="store_true", default=True,
                        help="ON by default in the canonical baseline. "
                             "Place a REAL resting GTC limit sell at entry+TP on "
                             "fill (captures spikes the poll misses). SL monitor "
                             "tracks fills, cancels before any other exit and "
                             "near settlement; startup cancels all resting orders "
                             "(orphan cleanup). Requires --take-profit-enabled for "
                             "the gate/threshold. HIGH-RISK live order mgmt â€” "
                             "verify with small bankroll first.")
    parser.add_argument("--no-resting-tp", dest="resting_tp_enabled", action="store_false",
                        help="Disable the resting GTC take-profit order.")
    parser.add_argument("--tp-reentry-enabled", action="store_true",
                        help="After a resting-TP order FULLY sells (whole "
                             "position exits in profit), re-open the window for "
                             "a fresh signal evaluation so the daemon may "
                             "re-enter (re-buy the same side or flip) IF signals "
                             "still align. Fires ONLY on a complete sell â€” never "
                             "on partial fills, SL/RRM exits, flips, or "
                             "settlement. Re-entry re-runs the full gate stack "
                             "and is repeatable. Requires --resting-tp-enabled. "
                             "LIVE-TEST FEATURE.")
    parser.add_argument("--sl-reentry-enabled", action="store_true",
                        help="After a position is FULLY closed by an adverse "
                             "exit (stop-loss / RRM reversal / futures fast-"
                             "exit), re-open the window for a fresh signal "
                             "evaluation so the daemon may re-enter (re-buy or "
                             "flip) IF signals still align. Independent of "
                             "--tp-reentry-enabled. Skipped on partial fills, "
                             "hedge-held positions, and when a smart-flip "
                             "already fired. Re-runs the full gate stack. "
                             "LIVE-TEST FEATURE \u2014 re-evaluates right after a "
                             "losing exit, so it can catch a real flip OR re-buy "
                             "into a continuing reversal.")
    parser.add_argument("--high-price-tp-enabled", action="store_true",
                        help="Ceiling TP for high-price entries: entries >= "
                             "--high-price-tp-min-cents rest a sell at "
                             "--high-price-tp-target-cents (instead of entry+TP, "
                             "which would exceed 99Â¢). Price-based, ALL tiers, "
                             "not perp-gated. Requires --resting-tp-enabled.")
    parser.add_argument("--high-price-tp-min-cents", type=int, default=89,
                        help="Entry price (Â¢) at/above which the ceiling TP "
                             "applies. Default 89.")
    parser.add_argument("--high-price-tp-target-cents", type=int, default=98,
                        help="Resting sell price (Â¢) for the ceiling TP. "
                             "Default 98 (99 fills less often).")
    # â”€â”€ 2026-06-27: HIGH-RISK TIGHT-TP (conditional) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    parser.add_argument("--highrisk-tp-enabled", action="store_true", default=True,
                        help="Conditional tight TP for high_conv trades that show "
                             "the last-minute-reversal signature: drew down >= "
                             "--highrisk-tp-dd-cents below entry AND entered within "
                             "--highrisk-tp-dist-max%% of strike. Rests a sell at "
                             "entry+--highrisk-tp-cents and keeps it alive to the "
                             "close. Requires --resting-tp-enabled. ON by default "
                             "in the canonical baseline.")
    parser.add_argument("--no-highrisk-tp", dest="highrisk_tp_enabled", action="store_false",
                        help="Disable the high-risk tight TP.")
    parser.add_argument("--highrisk-tp-dd-cents", type=int, default=12,
                        help="Min drawdown (Â¢ below entry) to flag a high_conv "
                             "position as high-risk. Default 12.")
    parser.add_argument("--highrisk-tp-dist-max", type=float, default=0.35,
                        help="Max |entry distance to strike| (%%) for the high-risk "
                             "gate. Default 0.35.")
    parser.add_argument("--highrisk-tp-cents", type=int, default=5,
                        help="Tight TP offset (Â¢ above entry) to lock in. Default 5.")
    # â”€â”€ 2026-06-12: PERP-MOMENTUM ENTRY VETO â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    parser.add_argument("--perp-veto-enabled", action="store_true", default=True,
                        help="Enforce the perp-momentum entry veto: skip entry "
                             "when the perp 30s tape is moving against "
                             "the position by more than the threshold. "
                             "ON by default in the canonical baseline. "
                             "Fail-open when perp feed is down.")
    parser.add_argument("--no-perp-veto", dest="perp_veto_enabled", action="store_false",
                        help="Disable the perp-momentum entry veto.")
    parser.add_argument("--perp-veto-m30s-threshold", type=float, default=-10.0,
                        help="Signed perp 30s momentum (bp, toward the trade "
                             "side) at or below which entry is vetoed. "
                             "Default -10.0 (canonical baseline).")
    # â”€â”€ 2026-07-02: POLYMARKET BID-STABILITY ENTRY VETO â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    parser.add_argument("--bid-stab-veto-enabled", action="store_true", default=True,
                        help="Skip entries unless our side's Polymarket bid "
                             "(100 - opposite ask) has been stable or rising "
                             "over the recent lookback. BTC 30d backtest: "
                             "+$1,133 net (saved $3,626 losses vs $2,493 "
                             "winners clipped), positive in both halves. "
                             "ON by default in the canonical baseline; the local "
                             "read is cross-checked against ChainVector "
                             "/predictions/stability at fire time.")
    parser.add_argument("--no-bid-stab-veto", dest="bid_stab_veto_enabled", action="store_false",
                        help="Disable the bid-stability entry veto.")
    parser.add_argument("--bid-stab-lookback-s", type=float, default=60.0,
                        help="Bid-stability lookback window in seconds. Default 60.")
    parser.add_argument("--bid-stab-max-fade-cents", type=int, default=4,
                        help="Max cents our side's bid may sit below its "
                             "lookback peak. Default 4 (canonical baseline).")
    parser.add_argument("--bid-stab-min-samples", type=int, default=3,
                        help="Minimum bid samples in the lookback before the "
                             "gate is active (fewer = fail-open). Default 3.")
    parser.add_argument("--bid-stab-burst-samples", type=int, default=3,
                        help="Extra LIVE re-samples of the book at the entry "
                             "moment (real-time micro-stability confirm over "
                             "the last ~5-10s). Any downtick vetoes. "
                             "0 disables the burst. Default 3.")
    parser.add_argument("--bid-stab-burst-interval-s", type=float, default=2.0,
                        help="Seconds between burst re-samples. Default 2.0 "
                             "(3 samples = ~6s confirm window).")
    # â”€â”€ 2026-06-14: BOOK-SKEW ENTRY VETO (golden-only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    parser.add_argument("--book-skew-veto-enabled", action="store_true",
                        help="Skip golden-zone (65-73c) entries when Binance "
                             "futures resting-depth skew (signed toward the "
                             "trade side) is <= threshold. Validated golden-"
                             "only: -0.15 gate cut 4 losers (incl -$446/-$417) "
                             "vs 7 small winners. HURTS non-golden tiers, so "
                             "scoped to golden by default. Fail-open if ChainVector down.")
    parser.add_argument("--book-skew-threshold", type=float, default=-0.15,
                        help="Signed book_skew at or below which a golden entry "
                             "is vetoed. Default -0.15.")
    parser.add_argument("--book-skew-all-tiers", action="store_true",
                        help="Apply the book-skew veto to ALL tiers, not just "
                             "golden. NOT recommended â€” backtest shows it hurts "
                             "non-golden tiers.")
    # â”€â”€ 2026-06-15: PERP-IMB DEEP VETO (all tiers) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    parser.add_argument("--perp-imb-veto", action="store_true",
                        help="Skip entries when the Polymarket perp book imbalance "
                             "(signed toward trade side) is <= threshold â€” i.e. "
                             "the perp book is STRONGLY stacked against the "
                             "position. All tiers, transient re-poll. Backtest: "
                             "deep threshold net-positive, cuts only small "
                             "winners, catches some (not all) reversal losers.")
    parser.add_argument("--perp-imb-veto-threshold", type=float, default=-0.50,
                        help="Signed perp_imb at or below which entry is vetoed. "
                             "Default -0.50 (deep â€” conservative on winner-cuts).")
    # â”€â”€ 2026-06-17: GOLDEN NEAR + HIGH-VOL VETO â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    parser.add_argument("--golden-near-vol-veto", action="store_true", default=True,
                        help="ON by default in the canonical baseline. "
                             "Skip golden-band entries that are BOTH near the "
                             "strike (|dist%%| < --golden-near-vol-dist-max) AND "
                             "in elevated short-term vol (GK_vol >= "
                             "--golden-near-vol-gk-min). Targets the near+high-vol "
                             "knife-edge bucket (~36%% loss vs ~20%% for far "
                             "entries). Golden-only, transient re-poll, fail-open "
                             "if dist/GK missing.")
    parser.add_argument("--no-golden-near-vol-veto", dest="golden_near_vol_veto",
                        action="store_false",
                        help="Disable the golden near+high-vol veto.")
    parser.add_argument("--golden-near-vol-dist-max", type=float, default=0.08,
                        help="|dist%%|-from-strike below which a golden entry is "
                             "'near' for the near+high-vol veto. Default 0.08.")
    parser.add_argument("--golden-near-vol-gk-min", type=float, default=0.0020,
                        help="GK_vol at/above which a golden entry is 'high-vol' "
                             "for the near+high-vol veto. Default 0.0020 (~1.2x "
                             "the 10-day golden median 0.00168).")
    parser.add_argument("--min-entry-price", type=int, default=0,
                        help="Hard entry-price floor in cents (all tiers, "
                             "final pre-order veto). 0=off. Recommended 89-90 "
                             "to keep the deep/high-conv band and cut golden "
                             "(65-73c) + standard (74-88c) cheap entries.")
    parser.add_argument("--max-trade-usd", type=float, default=800.0,
                        help="Hard ceiling on a single ticket's capital "
                             "(contracts x entry price), all tiers. Caps only "
                             "oversized tickets; small/medium trades unaffected. "
                             "Default 800 (canonical baseline); 0 = uncapped.")
    # â”€â”€ 2026-06-15: SURE-TRADE EV WALK-UP OVERRIDE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    parser.add_argument("--ev-walkup-override-enabled", action="store_true", default=True,
                        help="ON by default in the canonical baseline. "
                             "On high-conviction trades where book_skew agrees, "
                             "relax the EV walk floor (to --ev-override-floor) so "
                             "the price walk can win fills on 'sure' markets. "
                             "Backtest: strong + book_skew>0 + 78-86c was +EV; "
                             "book_skew<=0 same band was -EV (book_skew is the gate).")
    parser.add_argument("--no-ev-walkup-override", dest="ev_walkup_override_enabled",
                        action="store_false",
                        help="Disable the sure-trade EV walk-up override.")
    parser.add_argument("--ev-override-pwin-min", type=float, default=0.70,
                        help="Min p_win for the walk-up override. Default 0.70.")
    parser.add_argument("--ev-override-price-max", type=int, default=85,
                        help="Max price the override will walk to. Default 85c.")
    parser.add_argument("--ev-override-book-skew-min", type=float, default=0.0,
                        help="Min book_skew (signed toward trade) to allow the "
                             "override. Default 0.0 (book must be supportive).")
    parser.add_argument("--ev-override-floor", type=float, default=-0.10,
                        help="Relaxed EV floor used during override walk. "
                             "Default -0.10/c.")
    # â”€â”€ 2026-06-12: RRM LIVE EXIT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    parser.add_argument("--rrm-exit-enabled", action="store_true", default=True,
                        help="ON by default in the canonical baseline. "
                             "Sell positions when the reversal-risk monitor "
                             "fires (strike breach + multi-signal confluence "
                             "score). Uses the proven SL sell path. Positions "
                             "below --rrm-exit-min-contracts stay log-only.")
    parser.add_argument("--no-rrm-exit", dest="rrm_exit_enabled", action="store_false",
                        help="Disable RRM live exits (log-only).")
    parser.add_argument("--rrm-exit-min-score", type=int, default=6,
                        help="Minimum RRM score (of 10) to trigger a live "
                             "exit. Default 6.")
    parser.add_argument("--rrm-exit-min-contracts", type=int, default=25,
                        help="Minimum position size (contracts) for a live "
                             "RRM exit; smaller positions log-only. Default 25.")
    parser.add_argument("--rrm-exit-cushion-max", type=float, default=0.15,
                        help="Only fire a LIVE RRM exit when the entry was "
                             "within this |dist pct| of the strike (0 = no "
                             "cushion gate). RRM is net-additive only on "
                             "thin-cushion entries. Default 0.15 (canonical baseline).")
    parser.add_argument("--min-entry-cushion-pct", type=float, default=0.10,
                        help="Skip entries with |BTC-strike|/strike below "
                             "this pct (0 = disabled). Near-money entries are "
                             "net-losing on BTC 15m. Default 0.10 (canonical baseline).")
    # â”€â”€ 2026-06-18: PREDICT-CROSS-EXIT (drift-aware) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    parser.add_argument("--predict-cross-exit", dest="predict_cross_exit_enabled",
                        action="store_true",
                        help="Sell pre-emptively when the drift-aware P(end "
                             "OTM) from the perp oracle exceeds --pcross-prob "
                             "for --pcross-confirm-polls consecutive polls in "
                             "the final --pcross-max-mins. Uses the proven SL "
                             "sell path (kept alive into the last 2 min). "
                             "Positions below --pcross-min-contracts stay "
                             "log-only. Default OFF.")
    parser.add_argument("--pcross-prob", type=float, default=0.40,
                        help="P(end OTM) threshold to trigger predict-cross-"
                             "exit. Default 0.40.")
    parser.add_argument("--pcross-max-mins", type=float, default=2.0,
                        help="Only arm predict-cross-exit when minutes "
                             "remaining <= this. Default 2.0 (last 2 min).")
    parser.add_argument("--pcross-confirm-polls", type=int, default=2,
                        help="Consecutive polls above the prob threshold "
                             "required before selling. Default 2.")
    parser.add_argument("--pcross-min-contracts", type=int, default=25,
                        help="Minimum position size for a live predict-cross-"
                             "exit; smaller positions log-only. Default 25.")
    parser.add_argument("--pcross-keep-alive-mins", type=float, default=0.4,
                        help="Keep the SL sell path alive down to this many "
                             "minutes before close so predict-cross-exit can "
                             "act late. Default 0.4.")
    # â”€â”€ 2026-06-24: HOLD-TO-WIN (cancel resting TP on high-conviction) â”€â”€â”€
    parser.add_argument("--holdwin-enabled", action="store_true", default=True,
                        help="On golden winners that are clearly "
                             "working, cancel the resting +Nc TP and ride to "
                             "settlement for the full (100-entry)c. Re-arms the "
                             "TP if conviction deteriorates. $-stop/pcross/RRM "
                             "stay active as backstop. ON by default in the "
                             "canonical baseline.")
    parser.add_argument("--no-holdwin", dest="holdwin_enabled", action="store_false",
                        help="Disable hold-to-win.")
    parser.add_argument("--holdwin-tiers", type=str, default="golden",
                        help="Comma list of tiers eligible for hold-to-win. "
                             "Default 'golden' (canonical baseline; high_conv "
                             "excluded: its hold upside is < the TP).")
    parser.add_argument("--holdwin-min-profit-cents", type=int, default=3,
                        help="Only cancel the TP once the position is in profit "
                             "by >= this many cents. Default 3 (canonical baseline).")
    parser.add_argument("--holdwin-min-dist-pct", type=float, default=0.10,
                        help="Only cancel the TP when the live cushion "
                             "|BTC-strike|/strike >= this pct. Default 0.10.")
    parser.add_argument("--holdwin-max-potm", type=float, default=0.20,
                        help="Only cancel the TP when the drift-aware P(end OTM) "
                             "from the perp oracle <= this. Default 0.20.")
    parser.add_argument("--holdwin-rearm-dist-pct", type=float, default=0.05,
                        help="Re-arm the resting TP if the live cushion falls "
                             "below this pct. Default 0.05.")
    parser.add_argument("--holdwin-rearm-potm", type=float, default=1.01,
                        help="Re-arm the resting TP if P(end OTM) rises to/above "
                             "this. Default 1.01 (canonical baseline = the P(OTM) "
                             "re-arm path is effectively off; the trail-cents "
                             "re-arm still protects).")
    parser.add_argument("--holdwin-trail-cents", type=int, default=8,
                        help="Re-arm the resting TP if the favorable bid retraces "
                             "this many cents from its peak while holding "
                             "(locks the gain). Default 8.")
    parser.add_argument("--holdwin-min-gap", type=float, default=0.30,
                        help="Min entry Markov gap to hold a golden/standard "
                             "winner to settlement (skip resting TP). Default 0.30.")
    args = parser.parse_args()

    # ── Strategy templates (--strategy) ──────────────────────────────────────
    # Named presets over the existing gate stack. A preset only touches knobs
    # the user did NOT pass explicitly on the command line, so e.g.
    # `--strategy momentum --cv-mom-boost-weight 0.02` keeps the user's 0.02.
    STRATEGY_PRESETS = {
        # Canonical production defaults — no overrides.
        "baseline": {},
        # Tighten every ChainVector veto + cut per-trade risk. Fewer entries,
        # each with more confirmation behind it.
        "conservative": {
            "cv_prob_veto_max":       0.35,   # prob-engine floor 0.22 -> 0.35
            "cv_mom_veto_score":      50.0,   # momentum against-veto fires sooner
            "cv_mom_veto_breadth":    0.50,
            "cv_stab_mom_against":    30.0,   # stability veto fires sooner
            "cv_cascade_veto_score":  60.0,   # cascade veto fires sooner
            "futures_lead_veto_bps":  3.0,    # 6s lead veto stricter (5 -> 3 bps)
            "min_entry_cushion_pct":  0.15,   # demand more strike cushion
            "max_loss_per_trade":     50.0,
            "max_loss_per_trade_high_conv": 50.0,
        },
        # Relax the ChainVector vetoes for more entries; core Markov/EV gates
        # still apply. Higher variance.
        "aggressive": {
            "cv_prob_veto_max":       0.15,
            "cv_mom_veto_score":      80.0,
            "cv_stab_mom_against":    60.0,
            "cv_cascade_veto_score":  85.0,
            "futures_lead_veto_bps":  8.0,
            "min_entry_cushion_pct":  0.0,
            "term_prob_relax":        True,   # TP-confirmed relaxed gap floor
        },
        # Momentum-scorecard-led: bigger capped /momentum EV boost, boost
        # saturates earlier, against-veto much stricter, OKX boost heavier.
        "momentum": {
            "cv_mom_boost_weight":    0.10,   # ±10pp combined_p nudge
            "cv_mom_boost_scale":     35.0,   # saturates at aggregate ±35
            "cv_mom_veto_score":      45.0,   # any solid against-reading blocks
            "cv_mom_veto_breadth":    0.50,
            "okx_boost_weight":       0.08,
            "futures_lead_veto_bps":  4.0,
        },
        # Probability-engine-led: the six-estimator ensemble dominates
        # combined_p and its floor veto is demanding.
        "probability": {
            "ev_tp_weight":           0.70,   # prob engine 70% of combined_p
            "cv_prob_veto_max":       0.35,
            "term_prob_relax":        True,
        },
        # Prediction-quote-microstructure-led: strict /predictions/stability
        # veto + hardened local bid-stability gate (both burst and history).
        "stability": {
            "cv_stab_mom_against":    25.0,
            "bid_stab_lookback_s":    90.0,
            "bid_stab_max_fade_cents": 2,
            "bid_stab_burst_samples": 3,
            "min_entry_cushion_pct":  0.12,
        },
    }
    _opt_to_dest = {}
    for _a in parser._actions:
        for _opt in _a.option_strings:
            _opt_to_dest[_opt] = _a.dest
    _explicit = {_opt_to_dest[t.split("=", 1)[0]]
                 for t in sys.argv[1:]
                 if t.startswith("--") and t.split("=", 1)[0] in _opt_to_dest}
    _preset = STRATEGY_PRESETS.get(args.strategy, {})
    _applied = []
    for _dest, _val in _preset.items():
        if _dest in _explicit:
            continue
        setattr(args, _dest, _val)
        _applied.append(f"{_dest}={_val}")
    if _applied:
        log.info(f"[STRATEGY] '{args.strategy}' preset applied: "
                 + ", ".join(_applied))
    else:
        log.info(f"[STRATEGY] '{args.strategy}' (no overrides beyond defaults)")

    MIN_ENTRY_CUSHION_PCT = args.min_entry_cushion_pct
    RRM_EXIT_CUSHION_MAX = args.rrm_exit_cushion_max
    # Reconcile the back-compat flag
    if args.retry_plus_cent and args.retry_walk_cents == 0:
        args.retry_walk_cents = 1

    try:
        asyncio.run(main_loop(
            dry_run=args.dry_run,
            bankroll=args.bankroll,
            term_prob_disabled=args.no_term_prob,
            term_prob_relax=args.term_prob_relax,
            futures_lead_disabled=args.no_futures_lead,
            futures_lead_lookback_s=args.futures_lead_lookback,
            futures_lead_veto_bps=args.futures_lead_veto_bps,
            cv_mom_boost_enabled=args.cv_mom_boost_enabled,
            cv_mom_boost_weight=args.cv_mom_boost_weight,
            cv_mom_boost_scale=args.cv_mom_boost_scale,
            cv_mom_veto_enabled=args.cv_mom_veto_enabled,
            cv_mom_veto_score=args.cv_mom_veto_score,
            cv_mom_veto_breadth=args.cv_mom_veto_breadth,
            cv_prob_veto_enabled=args.cv_prob_veto_enabled,
            cv_prob_veto_max=args.cv_prob_veto_max,
            cv_stab_veto_enabled=args.cv_stab_veto_enabled,
            cv_stab_mom_against=args.cv_stab_mom_against,
            cv_cascade_veto_enabled=args.cv_cascade_veto_enabled,
            cv_cascade_veto_score=args.cv_cascade_veto_score,
            consensus_veto_enabled=not args.no_consensus_veto,
            consensus_min_move_pct=args.consensus_min_move_pct,
            consensus_okx_lookback_s=args.consensus_okx_lookback_s,
            consensus_smart_bypass=not args.no_consensus_smart_bypass,
            consensus_long_window_s=args.consensus_long_window_s,
            consensus_long_min_pct=args.consensus_long_min_pct,
            consensus_5m_favor_pct=args.consensus_5m_favor_pct,
            consensus_far_dist_pct=args.consensus_far_dist_pct,
            consensus_far_max_move_pct=args.consensus_far_max_move_pct,
            high_conv_confirmed_frac=args.high_conv_confirmed_frac,
            hc_block_on_split_externals=not args.no_hc_block_on_split,
            standard_confirmed_boost=args.standard_confirmed_boost,
            sl_enabled=not args.no_sl,
            sl_loss_cents=args.sl_loss_cents,
            sl_loss_cents_high_conv=args.sl_loss_cents_high_conv,
            max_loss_per_trade=args.max_loss_per_trade,
            max_loss_per_trade_high_conv=args.max_loss_per_trade_high_conv,
            sl_grace_mins=args.sl_grace_mins,
            sl_disable_late_mins=args.sl_disable_late_mins,
            sl_poll_interval_s=args.sl_poll_interval_s,
            sl_poll_interval_hc_s=args.sl_poll_interval_hc_s,
            futures_fast_exit_enabled=not args.no_futures_fast_exit,
            futures_fast_exit_window_s=args.futures_fast_exit_window_s,
            futures_fast_exit_threshold_pct=args.futures_fast_exit_threshold_pct,
            futures_fast_exit_sanity_max_pct=args.futures_fast_exit_sanity_max_pct,
            hedge_enabled=args.hedge_enabled,
            hedge_tiers=tuple(t.strip() for t in args.hedge_tiers.split(",") if t.strip()),
            hedge_min_yes_entry=args.hedge_min_yes_entry,
            hedge_max_yes_entry=args.hedge_max_yes_entry,
            hedge_max_no_cost=args.hedge_max_no_cost,
            hedge_no_settle_assumed=args.hedge_no_settle_assumed,
            hedge_max_capital_mult=args.hedge_max_capital_mult,
            hedge_widened_sl_cents=args.hedge_widened_sl_cents,
            hedge_no_sell_target=args.hedge_no_sell_target,
            hedge_no_sell_trail=args.hedge_no_sell_trail,
            hedge_post_sl_poll_s=args.hedge_post_sl_poll_s,
            smart_flip_enabled=args.smart_flip_enabled,
            smart_flip_tiers=tuple(t.strip() for t in args.smart_flip_tiers.split(",") if t.strip()),
            smart_flip_min_opp_entry=args.smart_flip_min_opp_entry,
            smart_flip_max_opp_entry=args.smart_flip_max_opp_entry,
            smart_flip_recovery_ratio=args.smart_flip_recovery_ratio,
            smart_flip_sl_cents=args.smart_flip_sl_cents,
            smart_flip_sell_target=args.smart_flip_sell_target,
            smart_flip_trail_cents=args.smart_flip_trail_cents,
            smart_flip_max_capital_usd=args.smart_flip_max_capital_usd,
            smart_flip_min_mins_remaining=args.smart_flip_min_mins_remaining,
            smart_flip_require_futures_confirm=not args.no_smart_flip_futures_confirm,
            smart_flip_futures_confirm_pct=args.smart_flip_futures_confirm_pct,
            smart_flip_futures_window_s=args.smart_flip_futures_window_s,
            smart_flip_poll_s=args.smart_flip_poll_s,
            smart_flip_retry_attempts=args.smart_flip_retry_attempts,
            smart_flip_retry_sleep_s=args.smart_flip_retry_sleep_s,
            hurst_tp_veto_enabled=args.hurst_tp_veto_enabled,
            hurst_tp_veto_min_hurst=args.hurst_tp_veto_min_hurst,
            hurst_tp_veto_min_diff=args.hurst_tp_veto_min_diff,
            hurst_tp_veto_tiers=tuple(t.strip() for t in args.hurst_tp_veto_tiers.split(",") if t.strip()),
            max_adverse_bar_veto_enabled=args.max_adverse_bar_veto_enabled,
            max_adverse_bar_veto_pct=args.max_adverse_bar_veto_pct,
            max_adverse_bar_veto_tiers=tuple(t.strip() for t in args.max_adverse_bar_veto_tiers.split(",") if t.strip()),
            cum_adverse_momentum_veto_enabled=not args.no_cum_adverse_momentum_veto,
            cum_adverse_momentum_veto_pct=args.cum_adverse_momentum_veto_pct,
            cum_adverse_momentum_veto_tiers=tuple(t.strip() for t in args.cum_adverse_momentum_veto_tiers.split(",") if t.strip()),
            hc_low_hurst_veto_enabled=not args.no_hc_low_hurst_veto,
            hc_low_hurst_threshold=args.hc_low_hurst_threshold,
            hc_low_hurst_markov_extremity=args.hc_low_hurst_markov_extremity,
            hc_dist_min=args.high_conv_dist_min,
            standard_price_cap_yes=args.standard_price_cap_yes,
            standard_price_cap_no=args.standard_price_cap_no,
            fade_bounce_enabled=args.fade_bounce_enabled,
            fade_bounce_no_ask_min=args.fade_bounce_no_ask_min,
            fade_bounce_no_ask_max=args.fade_bounce_no_ask_max,
            fade_bounce_yes_side_enabled=args.fade_bounce_yes_side_enabled,
            fade_bounce_markov_no_max=args.fade_bounce_markov_no_max,
            fade_bounce_markov_yes_min=args.fade_bounce_markov_yes_min,
            fade_bounce_hurst_min=args.fade_bounce_hurst_min,
            fade_bounce_dist_min=args.fade_bounce_dist_min,
            fade_bounce_min_mins=args.fade_bounce_min_mins,
            fade_bounce_max_mins=args.fade_bounce_max_mins,
            fade_bounce_min_stake_pct=args.fade_bounce_min_stake_pct,
            fade_bounce_kelly_frac=args.fade_bounce_kelly_frac,
            fade_bounce_sl_cents=args.fade_bounce_sl_cents,
            fade_bounce_max_capital_usd=args.fade_bounce_max_capital_usd,
            sl_trigger_mode=args.sl_trigger_mode,
            sl_aggressive_sell=not args.no_sl_aggressive_sell,
            ev_gate=args.ev_gate,
            ev_floor=args.ev_floor,
            ev_ceiling=args.ev_ceiling,
            ev_tp_weight=args.ev_tp_weight,
            ev_strong_floor=args.ev_strong_floor,
            ev_strong_gap_min=args.ev_strong_gap_min,
            ev_strong_price_max=args.ev_strong_price_max,
            ev_strong_tp_min=args.ev_strong_tp_min,
            ev_strong_max_mins=args.ev_strong_max_mins,
            ev_strong_max_adverse_momentum=args.ev_strong_max_adverse_momentum,
            last_bar_adverse_threshold=args.last_bar_adverse_threshold,
            orderbook_signal_enabled=args.orderbook_signal_enabled,
            trade_flow_signal_enabled=args.trade_flow_signal_enabled,
            trade_flow_lookback_n=args.trade_flow_lookback_n,
            orderbook_lockin_enabled=args.orderbook_lockin_enabled,
            orderbook_lockin_spread_max=args.orderbook_lockin_spread_max,
            orderbook_lockin_price_min=args.orderbook_lockin_price_min,
            orderbook_lockin_gap_min=args.orderbook_lockin_gap_min,
            late_window_price_max_lockin=args.late_window_price_max_lockin,
            hc_lockin_ev_floor=args.hc_lockin_ev_floor,
            hc_lockin_min_stake_pct=args.hc_lockin_min_stake_pct,
            strong_floor_hurst_bypass=args.strong_floor_hurst_bypass,
            retry_walk_cents=args.retry_walk_cents,
            max_window_fill_attempts=args.max_window_fill_attempts,
            refill_retry_sleep_s=args.refill_retry_sleep_s,
            late_window_mins=args.late_window_mins,
            late_window_price_max=args.late_window_price_max,
            late_window_min_tp=args.late_window_min_tp,
            late_sure_vol_bypass=args.late_sure_vol_bypass,
            late_window_ev_floor=args.late_window_ev_floor,
            strong_floor_min_stake_pct=args.strong_floor_min_stake_pct,
            late_sure_min_stake_pct=args.late_sure_min_stake_pct,
            standard_min_entry_contracts=args.standard_min_entry_contracts,
            golden_price_lo=args.golden_price_lo,
            golden_price_hi=args.golden_price_hi,
            golden_no_dist=args.golden_no_dist,
            golden_no_hurst=args.golden_no_hurst,
            okx_boost_enabled=args.okx_boost_enabled,
            okx_boost_weight=args.okx_boost_weight,
            okx_boost_scale=args.okx_boost_scale,
            high_conv_gap_min=args.high_conv_gap_min,
            high_conv_persist_min=args.high_conv_persist_min,
            high_conv_tp_strong=args.high_conv_tp_strong,
            high_conv_price_max=args.high_conv_price_max,
            high_conv_ev_floor=args.high_conv_ev_floor,
            high_conv_max_mins=args.high_conv_max_mins,
            high_conv_vol_bypass=not args.no_high_conv_vol_bypass,
            high_conv_vol_bypass_momentum=args.high_conv_vol_bypass_momentum,
            high_conv_vol_bypass_distance=args.high_conv_vol_bypass_distance,
            high_conv_vol_bypass_strong_distance=args.high_conv_vol_bypass_strong_distance,
            late_dir_enabled=not args.no_late_dir,
            late_dir_mins=args.late_dir_mins,
            late_dir_gap_min=args.late_dir_gap_min,
            late_dir_persist_min=args.late_dir_persist_min,
            late_dir_distance_min=args.late_dir_distance_min,
            late_dir_momentum_min=args.late_dir_momentum_min,
            late_dir_strong_distance=args.late_dir_strong_distance,
            late_dir_ev_floor=args.late_dir_ev_floor,
            late_dir_price_max=args.late_dir_price_max,
            order_lead_cents=args.order_lead_cents,
            okx_disabled=args.no_okx,
            okx_poll_interval_s=args.okx_poll_interval_s,
            rolling_wr_enabled=args.rolling_wr_enabled,
            rolling_wr_window=args.rolling_wr_window,
            rolling_wr_threshold=args.rolling_wr_threshold,
            rolling_wr_timeout_mins=args.rolling_wr_timeout_mins,
            rolling_wr_defensive_tiers=tuple(t.strip() for t in args.rolling_wr_defensive_tiers.split(",") if t.strip()),
            adaptive_br_enabled=args.adaptive_bankroll_enabled,
            adaptive_br_reduced_frac=args.adaptive_br_reduced_frac,
            adaptive_br_loss_trigger_usd=args.adaptive_br_loss_trigger_usd,
            adaptive_br_wr_trigger=args.adaptive_br_wr_trigger,
            adaptive_br_wr_window_h=args.adaptive_br_wr_window_h,
            adaptive_br_wr_min_trades=args.adaptive_br_wr_min_trades,
            adaptive_br_recover_wr=args.adaptive_br_recover_wr,
            adaptive_br_recover_window=args.adaptive_br_recover_window,
            adaptive_br_recover_min_wins=args.adaptive_br_recover_min_wins,
            patient_topup_enabled=args.patient_topup_enabled,
            patient_topup_interval_s=args.patient_topup_interval_s,
            patient_topup_min_mins=args.patient_topup_min_mins,
            patient_topup_dynamic_kelly=args.patient_topup_dynamic_kelly,
            take_profit_enabled=args.take_profit_enabled,
            take_profit_cents=args.take_profit_cents,
            take_profit_perp_confirmed_only=not args.take_profit_all_trades,
            take_profit_perp_min=args.take_profit_perp_min,
            resting_tp_enabled=args.resting_tp_enabled,
            tp_reentry_enabled=args.tp_reentry_enabled,
            sl_reentry_enabled=args.sl_reentry_enabled,
            high_price_tp_enabled=args.high_price_tp_enabled,
            high_price_tp_min_cents=args.high_price_tp_min_cents,
            high_price_tp_target_cents=args.high_price_tp_target_cents,
            highrisk_tp_enabled=args.highrisk_tp_enabled,
            highrisk_tp_dd_cents=args.highrisk_tp_dd_cents,
            highrisk_tp_dist_max=args.highrisk_tp_dist_max,
            highrisk_tp_cents=args.highrisk_tp_cents,
            perp_veto_enabled=args.perp_veto_enabled,
            perp_veto_m30s_threshold=args.perp_veto_m30s_threshold,
            bid_stab_veto_enabled=args.bid_stab_veto_enabled,
            bid_stab_lookback_s=args.bid_stab_lookback_s,
            bid_stab_max_fade_cents=args.bid_stab_max_fade_cents,
            bid_stab_min_samples=args.bid_stab_min_samples,
            bid_stab_burst_samples=args.bid_stab_burst_samples,
            bid_stab_burst_interval_s=args.bid_stab_burst_interval_s,
            taker_flow_veto_enabled=args.taker_flow_veto_enabled,
            taker_flow_veto_agg_min=args.taker_flow_veto_agg_min,
            taker_flow_veto_dist_max=args.taker_flow_veto_dist_max,
            taker_flow_veto_min_trades=args.taker_flow_veto_min_trades,
            book_skew_veto_enabled=args.book_skew_veto_enabled,
            book_skew_threshold=args.book_skew_threshold,
            book_skew_golden_only=not args.book_skew_all_tiers,
            perp_imb_veto_enabled=args.perp_imb_veto,
            perp_imb_veto_threshold=args.perp_imb_veto_threshold,
            golden_near_vol_veto_enabled=args.golden_near_vol_veto,
            golden_near_vol_dist_max=args.golden_near_vol_dist_max,
            golden_near_vol_gk_min=args.golden_near_vol_gk_min,
            min_entry_price=args.min_entry_price,
            max_trade_usd=args.max_trade_usd,
            ev_walkup_override_enabled=args.ev_walkup_override_enabled,
            ev_override_pwin_min=args.ev_override_pwin_min,
            ev_override_price_max=args.ev_override_price_max,
            ev_override_book_skew_min=args.ev_override_book_skew_min,
            ev_override_floor=args.ev_override_floor,
            rrm_exit_enabled=args.rrm_exit_enabled,
            rrm_exit_min_score=args.rrm_exit_min_score,
            rrm_exit_min_contracts=args.rrm_exit_min_contracts,
            predict_cross_exit_enabled=args.predict_cross_exit_enabled,
            pcross_prob=args.pcross_prob,
            pcross_max_mins=args.pcross_max_mins,
            pcross_confirm_polls=args.pcross_confirm_polls,
            pcross_min_contracts=args.pcross_min_contracts,
            pcross_keep_alive_mins=args.pcross_keep_alive_mins,
            holdwin_enabled=args.holdwin_enabled,
            holdwin_tiers=tuple(t.strip() for t in args.holdwin_tiers.split(",") if t.strip()),
            holdwin_min_profit_cents=args.holdwin_min_profit_cents,
            holdwin_min_dist_pct=args.holdwin_min_dist_pct,
            holdwin_max_potm=args.holdwin_max_potm,
            holdwin_rearm_dist_pct=args.holdwin_rearm_dist_pct,
            holdwin_rearm_potm=args.holdwin_rearm_potm,
            holdwin_trail_cents=args.holdwin_trail_cents,
            holdwin_min_gap=args.holdwin_min_gap,
        ))
    except KeyboardInterrupt:
        log.info("Daemon stopped by user.")

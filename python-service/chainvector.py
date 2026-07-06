"""
chainvector.py — ChainVector market-intelligence client (SOLE signals provider).

Every derived market signal this daemon consumes comes from the ChainVector
API (https://chainvector.com/docs). Polymarket remains the execution venue
(and its own orderbook/trades are read natively); Coinbase/OKX/Deribit index
reads remain only as the raw settlement price oracle. Everything else —
momentum, probabilities, liquidations, order flow, funding, whales,
prediction-market stability/edge — is ChainVector.

Endpoints wrapped here:
  • momentum(...)             — /momentum: cross-venue futures momentum
                                scorecard (−100..+100 per venue + aggregate,
                                breadth, dispersion). EV weight + veto + the
                                futures lead-lag feed (cv_lead.py).
  • probability(...)          — /probability: P(price > target at horizon end)
                                from a six-estimator ensemble. Called with
                                target=strike and close_ts=the contract's
                                exact close time (no bucket rounding). This is
                                the daemon's terminal-probability signal
                                (terminal_prob.py) and a weighted EV input.
  • stability(market_id)      — /predictions/stability: Polymarket YES-quote
                                stability (0..100) + repricing momentum
                                (−100..+100) over 30s/1m/5m windows. Entry
                                bid-price-stability gate.
  • edge(market_id)           — /predictions/edge: ChainVector model prob vs
                                the live Polymarket quote (model_prob,
                                buy_edge, sell_edge). Recorded signal,
                                promotable to a gate via --strategy edge.
  • prediction_markets/quotes/trades/results — Polymarket series metadata,
                                quote history, taker tape, settled outcomes.

Polymarket market ids: ChainVector's /predictions endpoints key markets by
the venue's own id. For robustness stability_any/edge_any accept every
identifier we know for a contract (condition_id, slug) and remember which
style the API resolved, so subsequent calls go straight to the winning form.
  • signals_snapshot(...)     — /signals/snapshot: spot, 1m vol, momentum,
                                book imbalance, whale pressure, cascade risk,
                                funding composite + squeeze in ONE call.
                                Backs signal_feeds.py.
  • cascade_risk(...)         — /liquidations/cascade-risk (incl. recent
                                long/short liquidation USD totals).
  • heatmap(...)              — /liquidations/heatmap: resting liquidation
                                clusters (liq_heatmap.py features).
  • cvd(...)/long_short(...)  — /orderflow/cvd + /long-short: taker flow.
  • orderbook_imbalance(...)  — /orderbook/imbalance: depth imbalance.
  • whale_pressure(...)       — /whale-pressure: signed large-trade flow.
  • funding_weighted(...)     — /funding/weighted: OI-weighted funding.
  • open_interest(...)        — /open-interest/history (5m buckets).
  • positioning(...)          — /positioning: exchange-reported long/short.
  • volatility(...)/regime(...)/risk_index(...) — recorded context signals.
  • candles(...)              — /candles: 1m OHLCV bars (Markov stack input,
                                aggregated client-side to 5m/15m).

All methods are FAIL-OPEN: any error / missing coverage returns None and the
caller treats the signal as unavailable (neutral). Results are cached with
per-endpoint TTLs so the poll loop respects plan rate limits, and a soft
circuit-breaker backs off after repeated API errors / 429s.

Auth: set CHAINVECTOR_API_KEY in .env.local (key looks like cv_live_...).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

import httpx

CHAINVECTOR_BASE = os.environ.get("CHAINVECTOR_BASE",
                                  "https://chainvector.com/api/v1")

# Per-endpoint cache TTLs (seconds). Tuned for the Developer plan
# (60 req/min): the fast loop leans on /momentum (cv_lead poller) and
# /signals/snapshot; everything else refreshes on slower cycles.
TTL_MOMENTUM   = 2.0     # cv_lead poller cadence handles the real limit
TTL_SNAPSHOT   = 12.0    # /signals/snapshot — one call bundles 7 signals
TTL_PROB       = 12.0    # /probability per (target, close_ts)
TTL_STABILITY  = 10.0    # /predictions/stability per market
TTL_EDGE       = 20.0    # /predictions/edge per market
TTL_CASCADE    = 20.0    # /liquidations/cascade-risk
TTL_CVD        = 20.0    # /orderflow/cvd
TTL_LONGSHORT  = 30.0    # /long-short
TTL_BOOK_IMB   = 10.0    # /orderbook/imbalance
TTL_WHALE      = 20.0    # /whale-pressure
TTL_FUNDING    = 60.0    # /funding/weighted
TTL_OI         = 60.0    # /open-interest/history
TTL_POSITIONING = 120.0  # /positioning
TTL_HEATMAP    = 120.0   # /liquidations/heatmap (slow-moving)
TTL_VOL        = 30.0    # /volatility
TTL_REGIME     = 30.0    # /regime
TTL_RISK       = 60.0    # /risk-index


class ChainVectorClient:
    """Thread-safe, cached, fail-open ChainVector REST client."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key
                        or os.environ.get("CHAINVECTOR_API_KEY", "").strip())
        self._lock = threading.Lock()
        self._caches: dict[str, dict] = {}   # name -> {key: (ts, value)}
        self._id_winners: dict = {}          # (id, id, ...) -> resolved id
        self._client = httpx.Client(
            timeout=12,
            headers=({"Authorization": f"Bearer {self.api_key}"}
                     if self.api_key else {}),
        )
        self._cooldown_until = 0.0

    @property
    def enabled(self) -> bool:
        """True when an API key is available. Re-checks os.environ lazily so
        a client constructed at import time (module-level singletons in
        signal_feeds/terminal_prob) picks up a key that .env.local loading
        or the host injects later in startup."""
        if not self.api_key:
            key = os.environ.get("CHAINVECTOR_API_KEY", "").strip()
            if key:
                self.api_key = key
                self._client.headers["Authorization"] = f"Bearer {key}"
        return bool(self.api_key)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    # ── Plumbing ─────────────────────────────────────────────────────────────
    def _get(self, path: str, params: dict) -> Optional[Any]:
        """One authenticated GET. Returns the envelope's `data` or None."""
        if not self.enabled or time.time() < self._cooldown_until:
            return None
        try:
            r = self._client.get(f"{CHAINVECTOR_BASE}{path}", params=params)
            if r.status_code == 429:
                retry = float(r.headers.get("Retry-After", 30))
                self._cooldown_until = time.time() + max(5.0, retry)
                return None
            if r.status_code in (400, 403, 404):
                return None  # bad param / plan / no coverage — fail open
            r.raise_for_status()
            payload = r.json()
        except Exception:
            self._cooldown_until = time.time() + 60.0
            return None
        if not payload.get("success"):
            return None
        return payload.get("data")

    def _cached(self, name: str, key, ttl_s: float, fetch) -> Optional[Any]:
        """Per-endpoint cache. `fetch` is a zero-arg callable hitting the API.
        Negative results (None) are cached too, at 1/3 TTL, so a downed
        endpoint doesn't get hammered."""
        if not self.enabled:
            return None
        now = time.time()
        with self._lock:
            cache = self._caches.setdefault(name, {})
            hit = cache.get(key)
            if hit is not None:
                age = now - hit[0]
                if hit[1] is not None and age < ttl_s:
                    return hit[1]
                if hit[1] is None and age < max(2.0, ttl_s / 3.0):
                    return None
        value = fetch()
        with self._lock:
            self._caches.setdefault(name, {})[key] = (now, value)
        return value

    # ── Futures momentum scorecard ────────────────────────────────────────────
    def momentum(self, coin: str = "BTC",
                 exchanges: Optional[str] = None,
                 ttl_s: float = TTL_MOMENTUM) -> Optional[dict]:
        """Cross-venue momentum snapshot. Full payload:
        {venues: [{exchange, last_mid, ret_bps{10s,30s,1m,5m,15m},
        z{...}, momentum_score}], aggregate_score (−100..+100),
        breadth_up, score_dispersion, sigma_1m_bps}."""
        return self._cached(
            "momentum", (coin, exchanges), ttl_s,
            lambda: self._fetch_momentum(coin, exchanges))

    def _fetch_momentum(self, coin: str, exchanges: Optional[str]) -> Optional[dict]:
        params = {"symbol": coin, "type": "perp"}
        if exchanges:
            params["exchanges"] = exchanges
        data = self._get("/momentum", params)
        return data if isinstance(data, dict) else None

    @staticmethod
    def momentum_signed(mom: Optional[dict], direction: str) -> Optional[float]:
        """Signed momentum agreement in [−100, +100]: positive = the
        cross-venue futures momentum agrees with our side (YES wants up)."""
        if not mom or mom.get("aggregate_score") is None:
            return None
        try:
            s = float(mom["aggregate_score"])
        except (TypeError, ValueError):
            return None
        return s if direction in ("YES", "UP") else -s

    # ── Probability engine (terminal P(above) ensemble) ──────────────────────
    _PROB_ESTIMATORS = ("gaussian", "markov", "bs_d2", "mc_gbm",
                        "mc_student_t", "bootstrap")

    def probability(self, target: float,
                    close_ts_ms: Optional[int] = None,
                    horizon_min: Optional[float] = None,
                    coin: str = "BTC",
                    kind: str = "terminal") -> Optional[dict]:
        """Terminal probability P(price > target at horizon end) from the
        six-estimator ensemble. Pass close_ts_ms (Unix ms of the contract's
        close_time) for an exact TTE — no bucket rounding. Returns
        {ensemble, gaussian, markov, bs_d2, mc_gbm, mc_student_t, bootstrap,
        sigma_horizon, sigma_1m, spot, horizon_minutes} or None."""
        key = (coin, kind, round(float(target), 8), int(close_ts_ms or 0),
               round(horizon_min or 0.0, 2))
        return self._cached(
            "probability", key, TTL_PROB,
            lambda: self._fetch_probability(coin, target, close_ts_ms,
                                            horizon_min, kind))

    def _fetch_probability(self, coin: str, target: float,
                           close_ts_ms: Optional[int],
                           horizon_min: Optional[float],
                           kind: str) -> Optional[dict]:
        params: dict = {"symbol": f"{coin}-USDT", "target": float(target),
                        "kind": kind}
        if close_ts_ms:
            params["close_ts"] = int(close_ts_ms)
        elif horizon_min is not None and horizon_min > 0:
            params["horizon"] = f"{max(0.5, horizon_min):g}m"
        data = self._get("/probability", params)
        if not isinstance(data, dict) or data.get("ensemble") is None:
            return None
        try:
            out = {"ensemble": float(data["ensemble"])}
        except (TypeError, ValueError):
            return None
        for k in self._PROB_ESTIMATORS:
            try:
                out[k] = float(data[k]) if data.get(k) is not None else None
            except (TypeError, ValueError):
                out[k] = None
        inputs = data.get("inputs") or {}
        out["sigma_horizon"] = inputs.get("sigma_horizon")
        out["sigma_1m"] = inputs.get("sigma_1m")
        out["sample_bars"] = inputs.get("sample_bars")
        return out

    @staticmethod
    def prob_side(prob: Optional[dict], side: str) -> Optional[float]:
        """P(our side wins) from a probability() result: ensemble P(above)
        for YES, its complement for NO. None if unavailable."""
        if not prob or prob.get("ensemble") is None:
            return None
        p = float(prob["ensemble"])
        return p if side in ("YES", "UP") else (1.0 - p)

    # ── Polymarket prediction-market signals ─────────────────────────────────
    def stability(self, market_id: str) -> Optional[dict]:
        """Quote stability & momentum for one Polymarket market. Raw
        per-window dict {"30s": {...}, "1m": {...}, "5m": {...}} where each
        window has stability_score (0..100, 100 = frozen tight quote),
        momentum_score (−100..+100 signed YES-mid repricing speed), churn,
        consistency, avg_spread. None if the market has no live quotes."""
        return self._cached(
            "stability", market_id, TTL_STABILITY,
            lambda: self._get("/predictions/stability",
                              {"venue": "polymarket", "market_id": market_id}))

    def stability_any(self, market_ids: list) -> Optional[dict]:
        """stability() across every known id form for the market
        (condition_id, slug); first hit wins and is remembered."""
        return self._try_ids(self.stability, market_ids)

    def edge(self, market_id: str) -> Optional[dict]:
        """Model-vs-market edge for one Polymarket market at full MC
        precision: {model_prob, market_prob, yes_bid, yes_ask, edge,
        buy_edge, sell_edge, minutes_to_close}. Recorded signal."""
        return self._cached("edge", market_id, TTL_EDGE,
                            lambda: self._fetch_edge(market_id))

    def edge_any(self, market_ids: list) -> Optional[dict]:
        """edge() across every known id form for the market."""
        return self._try_ids(self.edge, market_ids)

    def _try_ids(self, fn, market_ids: list) -> Optional[dict]:
        """Call `fn` with each candidate id until one resolves. Remembers the
        winning id style per id-set so subsequent calls skip the misses."""
        ids = [str(m) for m in (market_ids or []) if m]
        if not ids:
            return None
        key = tuple(ids)
        winner = self._id_winners.get(key)
        if winner:
            return fn(winner)
        for mid in ids:
            out = fn(mid)
            if out is not None:
                self._id_winners[key] = mid
                if len(self._id_winners) > 64:
                    for k in list(self._id_winners)[:-64]:
                        self._id_winners.pop(k, None)
                return out
        return None

    def _fetch_edge(self, market_id: str) -> Optional[dict]:
        data = self._get("/predictions/edge",
                         {"venue": "polymarket", "market_id": market_id})
        if isinstance(data, list):
            data = data[0] if data else None
        return data if isinstance(data, dict) else None

    def prediction_markets(self, series: str) -> Optional[list]:
        """Market metadata for a Polymarket series (discovery / research)."""
        return self._cached(
            "pred_markets", series, 60.0,
            lambda: self._get("/predictions/markets",
                              {"venue": "polymarket", "series": series}))

    def prediction_quotes(self, market_id: str, limit: int = 50) -> Optional[list]:
        """Recent YES top-of-book history for one Polymarket market."""
        return self._cached(
            "pred_quotes", (market_id, limit), 10.0,
            lambda: self._get("/predictions/quotes",
                              {"venue": "polymarket", "market_id": market_id,
                               "limit": limit}))

    def prediction_trades(self, series: str = "btc-updown-15m",
                          limit: int = 100) -> Optional[list]:
        """Recent taker tape across a Polymarket series (order-flow research)."""
        return self._cached(
            "pred_trades", (series, limit), 15.0,
            lambda: self._get("/predictions/trades",
                              {"venue": "polymarket", "series": series,
                               "limit": limit}))

    def prediction_results(self, series: str = "btc-updown-15m",
                           limit: int = 200) -> Optional[list]:
        """Settled outcomes for a series — calibration ground truth."""
        return self._cached(
            "pred_results", (series, limit), 300.0,
            lambda: self._get("/predictions/results",
                              {"venue": "polymarket", "series": series,
                               "limit": limit}))

    # ── One-call bot snapshot ─────────────────────────────────────────────────
    def signals_snapshot(self, coin: str = "BTC") -> Optional[dict]:
        """Every fast signal in one request: {spot, sigma_1m_bps, momentum,
        book, whale_pressure, cascade_risk, funding, squeeze}. Components
        degrade independently (unavailable blocks come back null)."""
        return self._cached(
            "snapshot", coin, TTL_SNAPSHOT,
            lambda: self._get("/signals/snapshot", {"symbol": coin}))

    # ── Liquidations ──────────────────────────────────────────────────────────
    def cascade_risk(self, coin: str = "BTC") -> Optional[dict]:
        """Cascade-risk composite: {risk_score 0-100, cascade_side,
        components, recent{long_liq_15m_usd, short_liq_15m_usd,
        long_liq_1h_usd, short_liq_1h_usd, ...}, oi_change_1h_pct}."""
        return self._cached("cascade", coin, TTL_CASCADE,
                            lambda: self._get("/liquidations/cascade-risk",
                                              {"symbol": coin}))

    def heatmap(self, coin: str = "BTC", window: str = "12h") -> Optional[dict]:
        """Raw /liquidations/heatmap data (spot + bins of long/short resting
        liquidation USD)."""
        return self._cached(
            "heatmap", (coin, window), TTL_HEATMAP,
            lambda: self._get("/liquidations/heatmap",
                              {"symbol": coin, "window": window}))

    # ── Order flow ────────────────────────────────────────────────────────────
    def cvd(self, coin: str = "BTC", interval: str = "1m") -> Optional[list]:
        """Per-bucket taker buy/sell notional + delta + running CVD,
        newest-first: [{bucket, buy_usd, sell_usd, delta_usd, cvd_usd}]."""
        return self._cached(
            "cvd", (coin, interval), TTL_CVD,
            lambda: self._get("/orderflow/cvd",
                              {"symbol": coin, "interval": interval}))

    def taker_flow_1m(self, coin: str = "BTC") -> Optional[float]:
        """Latest 1m taker-flow imbalance in [−1, +1]
        ((buy−sell)/(buy+sell)); None if unavailable."""
        rows = self.cvd(coin, "1m")
        if not rows:
            return None
        try:
            row = rows[0]
            buy = float(row.get("buy_usd") or 0)
            sell = float(row.get("sell_usd") or 0)
        except (TypeError, ValueError, AttributeError):
            return None
        total = buy + sell
        if total <= 0:
            return None
        return (buy - sell) / total

    def long_short(self, coin: str = "BTC",
                   interval: str = "5m") -> Optional[list]:
        """Taker long/short buy-share per interval bucket, newest-first."""
        return self._cached(
            "long_short", (coin, interval), TTL_LONGSHORT,
            lambda: self._get("/long-short",
                              {"symbol": coin, "interval": interval}))

    # ── Order book / whales / funding / OI / positioning ─────────────────────
    def orderbook_imbalance(self, coin: str = "BTC") -> Optional[dict]:
        """{mid, imbalance{d5,d10,d20}, microprice_tilt_bps}."""
        return self._cached(
            "book_imb", coin, TTL_BOOK_IMB,
            lambda: self._get("/orderbook/imbalance",
                              {"symbol": f"{coin}-USDT"}))

    def whale_pressure(self, coin: str = "BTC") -> Optional[dict]:
        """{windows{1m,5m,15m,1h: {net_usd, pressure, ...}},
        pressure_score −100..+100, direction}."""
        return self._cached(
            "whale", coin, TTL_WHALE,
            lambda: self._get("/whale-pressure", {"symbol": coin}))

    def funding_weighted(self, coin: str = "BTC") -> Optional[dict]:
        """OI-weighted composite funding for the asset:
        {venues, simple_avg, oi_weighted, simple_avg_8h_normalized,
        total_oi_usd}."""
        return self._cached("funding", coin, TTL_FUNDING,
                            lambda: self._fetch_funding(coin))

    def _fetch_funding(self, coin: str) -> Optional[dict]:
        data = self._get("/funding/weighted", {"base": coin})
        if isinstance(data, list):
            data = data[0] if data else None
        return data if isinstance(data, dict) else None

    def open_interest(self, coin: str = "BTC",
                      exchange: str = "binance_futures",
                      limit: int = 12) -> Optional[list]:
        """5-minute OI buckets for one venue, newest-first:
        [{bucket, open_interest, open_interest_usd}]."""
        return self._cached(
            "oi", (coin, exchange, limit), TTL_OI,
            lambda: self._get("/open-interest/history",
                              {"exchange": exchange,
                               "symbol": f"{coin}-USDT-PERP",
                               "limit": limit}))

    def positioning(self, coin: str = "BTC") -> Optional[list]:
        """Exchange-reported long/short positioning snapshot rows."""
        return self._cached(
            "positioning", coin, TTL_POSITIONING,
            lambda: self._get("/positioning", {"symbol": coin}))

    def global_ls_ratio(self, coin: str = "BTC") -> Optional[float]:
        """Global-account long/short ratio (retail crowd), first venue with
        the metric. None if unavailable."""
        rows = self.positioning(coin)
        if not rows:
            return None
        for row in rows:
            try:
                if row.get("metric") == "global_account":
                    return float(row["long_short_ratio"])
            except (TypeError, ValueError, KeyError):
                continue
        return None

    # ── Recorded context signals ──────────────────────────────────────────────
    def volatility(self, coin: str = "BTC",
                   horizon_min: Optional[float] = None) -> Optional[dict]:
        """Realized-vol battery + expected move over `horizon_min` minutes
        (rounded to 0.5m for cache friendliness). Recorded signal."""
        hz = None
        if horizon_min is not None and horizon_min > 0:
            hz = f"{max(0.5, round(horizon_min * 2) / 2.0):g}m"
        params = {"symbol": f"{coin}-USDT", "window": "4h",
                  "percentile": "true"}
        if hz:
            params["horizon"] = hz
        return self._cached("volatility", (coin, hz), TTL_VOL,
                            lambda: self._get("/volatility", params))

    def regime(self, coin: str = "BTC") -> Optional[dict]:
        """Market regime classifier (trending/choppy/quiet/squeeze_setup/
        cascading + confidence + inputs incl. Hurst). Recorded signal."""
        return self._cached("regime", coin, TTL_REGIME,
                            lambda: self._get("/regime", {"symbol": coin}))

    def risk_index(self, coin: str = "BTC") -> Optional[dict]:
        """ChainVector Risk Index read for one asset. Recorded signal."""
        return self._cached("risk", coin, TTL_RISK,
                            lambda: self._get("/risk-index", {"symbol": coin}))

    # ── Candles (Markov stack bar source) ─────────────────────────────────────
    def candles_1m(self, coin: str = "BTC", exchange: str = "binance",
                   limit: int = 1000,
                   from_ms: Optional[int] = None) -> Optional[list]:
        """Raw 1m OHLCV rows (newest-first dicts) from the tick-derived bars.
        NOT cached here — run_backtest.py does its own file-level caching."""
        params: dict = {"exchange": exchange, "symbol": f"{coin}-USDT",
                        "interval": "1m", "limit": min(int(limit), 10000)}
        if from_ms:
            params["from"] = int(from_ms)
        rows = self._get("/candles", params)
        return rows if isinstance(rows, list) and rows else None


# ── Module-level shared client ────────────────────────────────────────────────
_CLIENT: Optional[ChainVectorClient] = None
_CLIENT_LOCK = threading.Lock()


def get_client() -> ChainVectorClient:
    """Process-wide shared ChainVectorClient singleton."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = ChainVectorClient()
        return _CLIENT

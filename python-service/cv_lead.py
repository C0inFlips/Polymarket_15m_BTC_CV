"""
cv_lead.py — ChainVector futures lead-lag feed for the 15m Polymarket daemon.

Replaces the old Databento (CME MBT WebSocket) and OKX (REST ticker poll)
lead feeds with ONE background poller against ChainVector's
GET /api/v1/momentum endpoint, filtered to the venues we care about
(binance_futures as the primary lead venue, okx as the secondary).

Each /momentum response carries, per venue, the live perp mid (`last_mid`)
plus vol-normalized momentum scores. The poller appends every venue's mid to
a rolling in-memory tick buffer, so the daemon can ask "how much did futures
move in the last N seconds?" exactly the way it asked Databento/OKX — and the
latest full momentum scorecard (aggregate score, breadth, dispersion) is kept
for the EV weight / veto logic, at zero extra API cost.

Usage:
    feed = CVLeadFeed(poll_interval_s=3.0)
    feed.start()
    primary = feed.view("binance_futures")   # drop-in for DatabentoLeadFeed
    alt     = feed.view("okx")               # drop-in for OKXLeadFeed
    move = primary.get_recent_move(lookback_s=6.0)
    veto = primary.is_signal_consistent("YES", lookback_s=6.0,
                                        veto_threshold_bps=5)
    mom  = feed.latest_momentum()   # full scorecard dict (or None)

Rate budget: one /momentum call per poll. At the default 3s cadence that is
20 req/min — inside the Developer plan's 60/min with room for the rest of
the signal stack. If the API rate-limits (429) the ChainVector client backs
off automatically and the feed reports stale until it recovers (the daemon's
vetoes fail-open on stale data, same as with the old feeds).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional

from chainvector import get_client

log = logging.getLogger("daemon")   # share the daemon's logger

PRIMARY_VENUE = "binance_futures"
ALT_VENUE = "okx"


class CVLeadVenueView:
    """Read-only view over one venue's tick buffer. Interface-compatible with
    the old DatabentoLeadFeed / OKXLeadFeed query API (get_recent_move +
    is_signal_consistent)."""

    def __init__(self, feed: "CVLeadFeed", venue: str):
        self._feed = feed
        self.venue = venue

    # ── Public read API (same shape the daemon always consumed) ─────────────
    def get_recent_move(self, lookback_s: float = 6.0) -> dict:
        """Snapshot current mid + the mid `lookback_s` seconds ago and the
        % move between them. Non-blocking (lock + deque scan)."""
        out: dict = {
            "valid":       False,
            "n_ticks":     0,
            "stale_s":     None,
            "current_mid": None,
            "past_mid":    None,
            "move_pct":    None,
            "connected":   self._feed.connected,
            "front_month": self.venue,   # legacy key: venue id stands in
            "venue":       self.venue,
            "momentum_score": None,
        }
        ticks, last_ts, mom_score = self._feed._venue_state(self.venue)
        n = len(ticks)
        out["n_ticks"] = n
        out["momentum_score"] = mom_score
        if n == 0:
            return out
        now = time.time()
        out["stale_s"] = round(now - last_ts, 3) if last_ts else None
        cur_ts, cur_mid = ticks[-1]
        target_ts = now - lookback_s
        past_mid: Optional[float] = None
        for ts, mid in ticks:
            if ts >= target_ts:
                past_mid = mid
                break
        if past_mid is None:
            # Buffer doesn't span the lookback yet — use the oldest tick.
            past_mid = ticks[0][1]
            out["partial"] = True
        if past_mid and past_mid > 0:
            out["valid"] = True
            out["current_mid"] = round(cur_mid, 2)
            out["past_mid"] = round(past_mid, 2)
            out["move_pct"] = round((cur_mid - past_mid) / past_mid * 100, 4)
        return out

    def is_signal_consistent(self, direction: str,
                             lookback_s: float = 6.0,
                             veto_threshold_bps: float = 5.0) -> dict:
        """Check whether the recent futures move CONTRADICTS the proposed
        Polymarket trade direction. Fail-open on missing/stale data."""
        m = self.get_recent_move(lookback_s)
        out = dict(m)
        if not m["valid"]:
            out["consistent"] = True
            out["reason"] = ("no data yet (connected, building buffer)"
                             if m["connected"] else "feed disconnected")
            return out
        if m["stale_s"] is not None and m["stale_s"] > 15.0:
            out["consistent"] = True
            out["reason"] = (f"stale data ({m['stale_s']:.0f}s since last "
                             f"tick — feed may be rate-limited; veto disabled)")
            return out

        threshold_pct = veto_threshold_bps / 100.0  # bps → %
        move_pct = m["move_pct"]

        if direction == "YES":
            if move_pct < -threshold_pct:
                out["consistent"] = False
                out["reason"] = (f"futures dropped {move_pct:+.3f}% in "
                                 f"{lookback_s:.0f}s (threshold -{threshold_pct:.2f}%)")
            else:
                out["consistent"] = True
                out["reason"] = (f"futures {move_pct:+.3f}% in "
                                 f"{lookback_s:.0f}s — not against YES")
        elif direction == "NO":
            if move_pct > threshold_pct:
                out["consistent"] = False
                out["reason"] = (f"futures rose {move_pct:+.3f}% in "
                                 f"{lookback_s:.0f}s (threshold +{threshold_pct:.2f}%)")
            else:
                out["consistent"] = True
                out["reason"] = (f"futures {move_pct:+.3f}% in "
                                 f"{lookback_s:.0f}s — not against NO")
        else:
            out["consistent"] = True
            out["reason"] = f"unknown direction {direction!r}, abstaining"
        return out


class CVLeadFeed:
    """Background thread polling ChainVector /momentum for the lead venues,
    maintaining per-venue rolling mid-price buffers + the latest scorecard."""

    def __init__(self,
                 venues: tuple = (PRIMARY_VENUE, ALT_VENUE),
                 poll_interval_s: float = 3.0,
                 buffer_s: float = 180.0,
                 stale_threshold_s: float = 60.0):
        self.venues = tuple(venues)
        self.poll_interval_s = poll_interval_s
        self.buffer_s = buffer_s
        self.stale_threshold_s = stale_threshold_s

        self._cv = get_client()
        self._lock = threading.Lock()
        self._ticks: dict[str, deque] = {v: deque() for v in self.venues}
        self._venue_scores: dict[str, Optional[float]] = {v: None for v in self.venues}
        self._momentum: Optional[dict] = None      # latest full scorecard
        self._momentum_ts: float = 0.0
        self._last_tick_ts: dict[str, float] = {v: 0.0 for v in self.venues}
        self._stop = False
        self._poll_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self.connected = False
        self._consec_errors = 0

    # ── Lifecycle ────────────────────────────────────────────────────────────
    def start(self):
        if self._poll_thread is not None:
            return
        self._stop = False
        self._poll_thread = threading.Thread(
            target=self._poll_loop, name="cv-lead-poller", daemon=True)
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="cv-lead-watchdog", daemon=True)
        self._poll_thread.start()
        self._watchdog_thread.start()
        log.info(f"[CVLead] background poller started "
                 f"(venues={','.join(self.venues)}, "
                 f"poll_interval={self.poll_interval_s:.1f}s, "
                 f"buffer={self.buffer_s:.0f}s)")

    def stop(self):
        self._stop = True
        for t in (self._poll_thread, self._watchdog_thread):
            if t is not None:
                t.join(timeout=2)
        self._poll_thread = None
        self._watchdog_thread = None

    def view(self, venue: str) -> CVLeadVenueView:
        """A per-venue view exposing the legacy lead-feed query API."""
        return CVLeadVenueView(self, venue)

    # ── Public reads ─────────────────────────────────────────────────────────
    def latest_momentum(self, max_age_s: float = 30.0) -> Optional[dict]:
        """Latest full /momentum scorecard {venues, aggregate_score,
        breadth_up, score_dispersion, sigma_1m_bps}, or None if stale."""
        with self._lock:
            if self._momentum is None:
                return None
            if time.time() - self._momentum_ts > max_age_s:
                return None
            return self._momentum

    def latest_mid(self, venue: str = PRIMARY_VENUE,
                   max_age_s: float = 30.0) -> Optional[float]:
        """Most recent perp mid for a venue, or None if stale/missing."""
        ticks, last_ts, _ = self._venue_state(venue)
        if not ticks or (time.time() - last_ts) > max_age_s:
            return None
        return ticks[-1][1]

    def mid_series(self, venue: str = PRIMARY_VENUE,
                   window_s: float = 120.0) -> list:
        """[(ts, mid), ...] oldest-first for the trailing window."""
        cutoff = time.time() - window_s
        ticks, _, _ = self._venue_state(venue)
        return [(ts, mid) for ts, mid in ticks if ts >= cutoff]

    # ── Internals ────────────────────────────────────────────────────────────
    def _venue_state(self, venue: str):
        with self._lock:
            ticks = list(self._ticks.get(venue, ()))
            last_ts = self._last_tick_ts.get(venue, 0.0)
            score = self._venue_scores.get(venue)
        return ticks, last_ts, score

    def _poll_loop(self):
        exchanges = ",".join(self.venues)
        while not self._stop:
            poll_start = time.time()
            try:
                data = self._cv.momentum("BTC", exchanges=exchanges,
                                         ttl_s=max(0.5, self.poll_interval_s - 0.5))
                if isinstance(data, dict) and data.get("venues"):
                    now = time.time()
                    with self._lock:
                        self._momentum = data
                        self._momentum_ts = now
                        for vrow in data.get("venues") or []:
                            venue = str(vrow.get("exchange") or "")
                            if venue not in self._ticks:
                                continue
                            try:
                                mid = float(vrow.get("last_mid") or 0)
                            except (TypeError, ValueError):
                                mid = 0.0
                            try:
                                self._venue_scores[venue] = (
                                    float(vrow["momentum_score"])
                                    if vrow.get("momentum_score") is not None
                                    else None)
                            except (TypeError, ValueError):
                                self._venue_scores[venue] = None
                            if mid > 0:
                                dq = self._ticks[venue]
                                dq.append((now, mid))
                                self._last_tick_ts[venue] = now
                                cutoff = now - self.buffer_s
                                while dq and dq[0][0] < cutoff:
                                    dq.popleft()
                    self.connected = True
                    self._consec_errors = 0
                else:
                    self._consec_errors += 1
                    if self._consec_errors >= 10:
                        self.connected = False
            except Exception as e:
                self._consec_errors += 1
                if self._consec_errors in (1, 10) or self._consec_errors % 60 == 0:
                    log.debug(f"[CVLead] poll error #{self._consec_errors}: {e!r}")
                if self._consec_errors >= 10:
                    self.connected = False

            elapsed = time.time() - poll_start
            time.sleep(max(0.0, self.poll_interval_s - elapsed))

    def _watchdog_loop(self):
        """Warn when the feed goes stale (rate-limited / API down). The
        poller keeps retrying on its own; this is purely observability."""
        while not self._stop:
            time.sleep(15.0)
            last = max(self._last_tick_ts.values() or [0.0])
            if last <= 0:
                continue
            stale = time.time() - last
            if stale > self.stale_threshold_s:
                log.warning(
                    f"[CVLead] WATCHDOG: no venue ticks for {stale:.0f}s "
                    f"(threshold {self.stale_threshold_s:.0f}s) — momentum "
                    f"feed degraded; lead vetoes fail-open until it recovers.")
                self.connected = False


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from pathlib import Path

    _env_path = Path(__file__).parent.parent / ".env.local"
    if _env_path.exists():
        for line in _env_path.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if not os.environ.get("CHAINVECTOR_API_KEY"):
        raise SystemExit("CHAINVECTOR_API_KEY not set in .env.local")

    print("Starting CV lead feed...")
    feed = CVLeadFeed()
    feed.start()
    primary = feed.view(PRIMARY_VENUE)
    alt = feed.view(ALT_VENUE)
    try:
        for i in range(8):
            time.sleep(5)
            mv = primary.get_recent_move(lookback_s=6.0)
            print(f"\n[poll {i+1}] connected={mv['connected']} "
                  f"n_ticks={mv['n_ticks']} stale_s={mv['stale_s']}")
            if mv["valid"]:
                print(f"  {PRIMARY_VENUE}: mid=${mv['current_mid']:,.2f} "
                      f"move={mv['move_pct']:+.4f}% score={mv['momentum_score']}")
                y = primary.is_signal_consistent("YES")
                n = primary.is_signal_consistent("NO")
                print(f"  YES: consistent={y['consistent']} ({y['reason']})")
                print(f"  NO:  consistent={n['consistent']} ({n['reason']})")
            amv = alt.get_recent_move(lookback_s=6.0)
            if amv["valid"]:
                print(f"  {ALT_VENUE}: mid=${amv['current_mid']:,.2f} "
                      f"move={amv['move_pct']:+.4f}%")
            mom = feed.latest_momentum()
            if mom:
                print(f"  scorecard: agg={mom.get('aggregate_score')} "
                      f"breadth={mom.get('breadth_up')} "
                      f"disp={mom.get('score_dispersion')}")
    finally:
        feed.stop()
        print("\nStopped.")

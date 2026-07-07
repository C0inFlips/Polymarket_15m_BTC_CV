"""
cv_collector.py — ChainVector state collector for the CV veto/exit stack.

Writes `latest_{ASSET}.json` into --state-dir every ~5s. The trade daemon's
CV-FLOW / CV-EV / CV-REV entry vetoes, the CV-REV in-trade exit, the
CV-confirmed SL deferral and the --cv-shadow-enabled audit stamp all read
this file (read-only, fail-open on staleness), so the daemon never blocks
on the ChainVector API inside its decision path.

File schema (top-level `ts_ms` is the write time; every block carries its
own `<block>_age_ms` = how stale that block already was AT write time, so
readers compute true age as `(now - ts_ms) + block_age_ms`):

  ts_ms                 int   — write timestamp (Unix ms)
  probability           dict  — /probability at the CURRENT window's strike
                                with exact close_ts TTE:
                                {ensemble, market_id, inputs{log_moneyness}}
  probability_age_ms    int
  prob_roll             dict  — rolling window over our own probability
                                samples: {market_id, n_3m, ens_min_3m,
                                ens_max_3m, lm_cross_4m}
  prob_roll_age_ms      int
  momentum              dict  — /momentum scorecard
                                {aggregate_score, breadth_up, ...}
  momentum_age_ms       int
  snapshot              dict  — /signals/snapshot {whale_pressure, ...}
  snapshot_age_ms       int
  cascade               dict  — /liquidations/cascade-risk
                                {risk_score, cascade_side, ...}
  cascade_age_ms        int
  edge                  dict  — /predictions/edge for the current market
                                {market_id, model_prob, market_prob, ...}
  edge_age_ms           int
  volatility            dict  — /volatility with horizon = time-to-close
                                {expected_move{sigma_pct}, ...}
  volatility_age_ms     int

`market_id` is the Polymarket slug (the daemon's `ticker`), so the daemon's
`market_id == ticker` guards match. All fetches go through the shared
ChainVectorClient (per-endpoint TTL caches + 429 backoff), so a 5s loop
stays inside the Developer plan budget.

Run alongside the daemon:

    python cv_collector.py --asset BTC --state-dir cv_state

then start the daemon with the CV gates armed, e.g.:

    python trade_daemon.py --cv-asset BTC --cv-state-dir cv_state \
        --cv-shadow-enabled --cv-flow-veto-enabled --cv-ev-veto-enabled \
        --cv-rev-veto-enabled --cv-rev-exit-enabled --cv-sl-defer-enabled
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

# Load .env.local (CHAINVECTOR_API_KEY etc) before the chainvector import,
# same as trade_daemon.py.
_env_path = Path(__file__).parent.parent / ".env.local"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from chainvector import get_client
import polymarket_client as pm_client

log = logging.getLogger("cv_collector")

WINDOW_S = 900  # 15m windows


class _Block:
    """One fetched block + the time it was last refreshed."""

    __slots__ = ("value", "fetched_ms")

    def __init__(self):
        self.value = None
        self.fetched_ms = 0.0

    def set(self, value) -> None:
        if value is not None:
            self.value = value
            self.fetched_ms = time.time() * 1000.0

    def age_ms(self, now_ms: float) -> Optional[int]:
        if self.value is None or not self.fetched_ms:
            return None
        return int(now_ms - self.fetched_ms)


def _current_window(now: Optional[float] = None) -> tuple[str, int, int]:
    """(slug, start_ts, close_ts) of the currently-open 15m window."""
    ts = int(now if now is not None else time.time())
    start = (ts // WINDOW_S) * WINDOW_S
    return pm_client.slug_for_window_start(start), start, start + WINDOW_S


class Collector:
    def __init__(self, asset: str, state_dir: str, interval_s: float):
        self.asset = asset.upper()
        self.state_dir = state_dir
        self.interval_s = interval_s
        self.cv = get_client()
        self.prob = _Block()
        self.mom = _Block()
        self.snap = _Block()
        self.casc = _Block()
        self.edge = _Block()
        self.vol = _Block()
        # Rolling probability samples for prob_roll:
        # [(unix_s, ensemble, log_moneyness)]
        self._roll: list = []
        self._roll_market: Optional[str] = None
        self._strike_cache: dict = {}

    async def _strike_for(self, slug: str) -> Optional[float]:
        if slug in self._strike_cache:
            return self._strike_cache[slug]
        try:
            strike = await pm_client.get_strike(slug)
        except Exception:
            strike = None
        if strike:
            self._strike_cache[slug] = strike
            if len(self._strike_cache) > 8:
                for k in list(self._strike_cache)[:-8]:
                    self._strike_cache.pop(k, None)
        return strike

    def _update_roll(self, slug: str, ensemble: float,
                     lm: Optional[float]) -> dict:
        now_s = time.time()
        if self._roll_market != slug:
            self._roll = []
            self._roll_market = slug
        self._roll.append((now_s, float(ensemble),
                           float(lm) if lm is not None else None))
        cut = now_s - 240.0  # keep 4m of samples
        self._roll = [r for r in self._roll if r[0] >= cut]
        r3 = [r for r in self._roll if r[0] >= now_s - 180.0]
        ens3 = [r[1] for r in r3]
        # lm sign changes over 4m = spot crossing the strike
        lms = [r[2] for r in self._roll if r[2] is not None and r[2] != 0.0]
        crosses = sum(1 for a, b in zip(lms, lms[1:])
                      if math.copysign(1, a) != math.copysign(1, b))
        return {
            "market_id": slug,
            "n_3m": len(r3),
            "ens_min_3m": min(ens3) if ens3 else None,
            "ens_max_3m": max(ens3) if ens3 else None,
            "lm_cross_4m": crosses,
        }

    async def poll_once(self) -> dict:
        slug, _start, close_ts = _current_window()
        mins_left = max(0.5, (close_ts - time.time()) / 60.0)
        loop = asyncio.get_running_loop()

        strike = await self._strike_for(slug)

        def _fetch_sync():
            out = {}
            if strike:
                out["prob"] = self.cv.probability(
                    target=float(strike), close_ts_ms=close_ts * 1000,
                    coin=self.asset)
            out["mom"] = self.cv.momentum(self.asset)
            out["snap"] = self.cv.signals_snapshot(self.asset)
            out["casc"] = self.cv.cascade_risk(self.asset)
            out["edge"] = self.cv.edge_any([slug])
            out["vol"] = self.cv.volatility(self.asset,
                                            horizon_min=mins_left)
            return out

        fetched = await loop.run_in_executor(None, _fetch_sync)

        prob = fetched.get("prob")
        prob_roll = None
        if prob and prob.get("ensemble") is not None:
            lm = prob.get("log_moneyness")
            prob_out = {
                "ensemble": prob["ensemble"],
                "market_id": slug,
                "inputs": {"log_moneyness": lm},
            }
            self.prob.set(prob_out)
            prob_roll = self._update_roll(slug, prob["ensemble"], lm)
        mom = fetched.get("mom")
        if mom:
            self.mom.set(mom)
        snap = fetched.get("snap")
        if snap:
            self.snap.set(snap)
        casc = fetched.get("casc")
        if casc:
            self.casc.set(casc)
        edge = fetched.get("edge")
        if edge:
            edge = dict(edge)
            edge.setdefault("market_id", slug)
            edge["market_id"] = slug
            self.edge.set(edge)
        vol = fetched.get("vol")
        if vol:
            self.vol.set(vol)

        now_ms = time.time() * 1000.0
        state = {"ts_ms": int(now_ms), "asset": self.asset,
                 "market_id": slug}
        for key, block in (("probability", self.prob),
                           ("momentum", self.mom),
                           ("snapshot", self.snap),
                           ("cascade", self.casc),
                           ("edge", self.edge),
                           ("volatility", self.vol)):
            state[key] = block.value
            age = block.age_ms(now_ms)
            if age is not None:
                state[f"{key}_age_ms"] = age
        if prob_roll:
            state["prob_roll"] = prob_roll
            state["prob_roll_age_ms"] = 0
        return state

    def write(self, state: dict) -> None:
        os.makedirs(self.state_dir, exist_ok=True)
        path = os.path.join(self.state_dir, f"latest_{self.asset}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, separators=(",", ":"))
        os.replace(tmp, path)

    async def run(self) -> None:
        log.info(f"cv_collector: asset={self.asset} "
                 f"state_dir={self.state_dir} interval={self.interval_s}s")
        while True:
            t0 = time.time()
            try:
                state = await self.poll_once()
                self.write(state)
                p = state.get("probability") or {}
                log.info(
                    f"wrote {state['market_id']}  "
                    f"ens={p.get('ensemble')}  "
                    f"mom={(state.get('momentum') or {}).get('aggregate_score')}  "
                    f"casc={(state.get('cascade') or {}).get('risk_score')}")
            except Exception as e:
                log.warning(f"poll failed (non-fatal): {e!r}")
            await asyncio.sleep(max(0.5, self.interval_s - (time.time() - t0)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--asset", type=str, default="BTC",
                    help="ChainVector asset symbol (default BTC).")
    ap.add_argument("--state-dir", type=str,
                    default=os.environ.get("CV_STATE_DIR", "cv_state"),
                    help="Directory for latest_{asset}.json "
                         "(default ./cv_state or CV_STATE_DIR).")
    ap.add_argument("--interval-s", type=float, default=5.0,
                    help="Write cadence in seconds (default 5).")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(Collector(args.asset, args.state_dir, args.interval_s).run())

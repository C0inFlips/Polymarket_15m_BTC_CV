"""signal_feeds.py — ChainVector-backed live signal snapshots (was: perp +
CoinGlass JSONL recorder feeds).

The original program tailed two recorder JSONL files (perp_recorder /
coinglass_recorder). This version sources every feature LIVE from the
ChainVector API — no recorder sidecars, no files:

  perp_m6s/m30s/m60s/m5m — futures mid momentum (bps, SIGNED toward side),
                            from the cv_lead in-memory tick buffer
                            (ChainVector /momentum poller, binance_futures)
  perp_imb                — order-book depth imbalance d10 (SIGNED),
                            /signals/snapshot book block
  book_skew               — depth-20 imbalance (SIGNED), same block
  liq_adverse_5m_k        — $k of SAME-side liquidations in the last ~5 min
                            (long-liqs hurt YES, short-liqs hurt NO),
                            /liquidations/cascade-risk `recent` deltas
  liq_support_5m_k        — $k of OPPOSITE-side liquidations, same source
  taker_1m                — cross-venue 1m taker flow imbalance (SIGNED),
                            /orderflow/cvd
  oi_d5m_pct              — market-wide OI change over ~5 min (%),
                            /signals/snapshot funding.total_oi_usd history
  funding                 — OI-weighted composite funding, same block
  whale                   — whale-pressure score (−100..+100), same block
  ls_ratio                — global-account long/short ratio, /positioning
  liq_heatmap / liq_skew  — resting-liquidation features,
                            /liquidations/heatmap via liq_heatmap.py
  cascade                 — {risk_score, cascade_side} for the cascade veto

STRICTLY fail-open — never raises, never blocks a trade on its own. If the
API is down/stale, features come back None and verdicts abstain.

Call attach_lead_feed(feed) once at daemon startup (after CVLeadFeed.start())
so momentum/mid features read from the shared in-memory buffer, and so the
background context sampler (20s cadence) starts building the short history
needed for the 5m delta features.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from chainvector import get_client
from liq_heatmap import heatmap_features


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


PERP_STALE_S = 30      # lead-feed sample older than this = feed down
CV_STALE_S = 90        # snapshot older than this = feed down

_cv = get_client()
_lead_feed = None                 # CVLeadFeed, set by attach_lead_feed()
_hist_lock = threading.Lock()
_oi_hist: deque = deque(maxlen=60)        # (ts, total_oi_usd)
_liq_hist: deque = deque(maxlen=60)       # (ts, long_liq_1h_usd, short_liq_1h_usd)
_sampler_started = False


def attach_lead_feed(feed) -> None:
    """Wire in the shared CVLeadFeed and start the context sampler."""
    global _lead_feed, _sampler_started
    _lead_feed = feed
    if not _sampler_started:
        _sampler_started = True
        t = threading.Thread(target=_sampler_loop, name="cv-context-sampler",
                             daemon=True)
        t.start()


def _sampler_loop():
    """Every 20s, touch the snapshot + cascade endpoints (client-cached) and
    append the values needed for delta features (OI, liquidation totals)."""
    while True:
        try:
            _sample_context()
        except Exception:
            pass
        time.sleep(20.0)


def _sample_context():
    now = time.time()
    snap = _cv.signals_snapshot("BTC")
    if isinstance(snap, dict):
        funding = snap.get("funding") or {}
        oi = funding.get("total_oi_usd")
        if oi:
            with _hist_lock:
                _oi_hist.append((now, float(oi)))
    casc = _cv.cascade_risk("BTC")
    if isinstance(casc, dict):
        recent = casc.get("recent") or {}
        ll, sl = recent.get("long_liq_1h_usd"), recent.get("short_liq_1h_usd")
        if ll is not None and sl is not None:
            with _hist_lock:
                _liq_hist.append((now, float(ll), float(sl)))


def _hist_at(hist: deque, ago_s: float, tol_s: float):
    """Sample from a (ts, ...) deque closest to `ago_s` seconds ago."""
    target = time.time() - ago_s
    best = None
    with _hist_lock:
        rows = list(hist)
    for row in rows:
        d = abs(row[0] - target)
        if d <= tol_s and (best is None or d < best[0]):
            best = (d, row)
    return best[1] if best else None


def _lead_move_bps(window_s: float) -> Optional[float]:
    """Raw (unsigned-side) primary-venue move over window_s, in bps."""
    if _lead_feed is None:
        return None
    try:
        m = _lead_feed.view("binance_futures").get_recent_move(lookback_s=window_s)
        if not m.get("valid") or m.get("partial"):
            return None
        if m.get("stale_s") is not None and m["stale_s"] > PERP_STALE_S:
            return None
        return (m["current_mid"] - m["past_mid"]) / m["past_mid"] * 10000.0
    except Exception:
        return None


def snapshot_new_signals(side: str) -> dict:
    """Compute the candidate-signal snapshot for a trade on `side` (YES/NO).

    Never raises. Returns a dict of features (None where unavailable) plus
    log-only veto verdicts and a one-line human summary. Signed features are
    positive when they SUPPORT the trade side.
    """
    out: dict = {
        "perp_m6s": None, "perp_m30s": None, "perp_m60s": None, "perp_m5m": None,
        "perp_imb": None, "perp_spread": None,
        "liq_adverse_5m_k": None, "liq_support_5m_k": None,
        "taker_1m": None, "oi_d5m_pct": None,
        "funding": None, "whale": None, "ls_ratio": None, "book_skew": None,
        "liq_heatmap": None,
        "liq_skew": None,
        "cv_momentum": None, "cascade": None,
        "perp_feed_ok": False, "cv_feed_ok": False,
        "perp_veto": None, "liq_veto": None,
        "summary": "feeds unavailable",
    }
    try:
        sgn = 1 if side == "YES" else -1

        # ── Futures lead feed (ChainVector /momentum poller) ─────────────────
        if _lead_feed is not None:
            mid_now = _lead_feed.latest_mid("binance_futures",
                                            max_age_s=PERP_STALE_S)
            if mid_now:
                out["perp_feed_ok"] = True
                for w_s, key in [(6, "perp_m6s"), (30, "perp_m30s"),
                                 (60, "perp_m60s"), (300, "perp_m5m")]:
                    bp = _lead_move_bps(w_s)
                    if bp is not None:
                        out[key] = round(sgn * bp, 3)
            mom = _lead_feed.latest_momentum()
            if mom:
                out["cv_momentum"] = {
                    "aggregate_score": mom.get("aggregate_score"),
                    "breadth_up": mom.get("breadth_up"),
                    "score_dispersion": mom.get("score_dispersion"),
                    "sigma_1m_bps": mom.get("sigma_1m_bps"),
                }

        # ── ChainVector snapshot block ───────────────────────────────────────
        snap = _cv.signals_snapshot("BTC")
        if isinstance(snap, dict):
            out["cv_feed_ok"] = True
            book = snap.get("book") or {}
            imb = book.get("imbalance") or {}
            if imb.get("d10") is not None:
                out["perp_imb"] = round(sgn * float(imb["d10"]), 4)
            if imb.get("d20") is not None:
                out["book_skew"] = round(sgn * float(imb["d20"]), 4)
            whale = snap.get("whale_pressure") or {}
            if whale.get("pressure_score") is not None:
                out["whale"] = whale["pressure_score"]
            funding = snap.get("funding") or {}
            if funding.get("oi_weighted") is not None:
                out["funding"] = funding["oi_weighted"]
            # OI change over ~5 min from the sampler history
            oi_now = funding.get("total_oi_usd")
            ago5 = _hist_at(_oi_hist, 300.0, tol_s=90.0)
            if oi_now and ago5 and ago5[1] > 0:
                out["oi_d5m_pct"] = round(
                    (float(oi_now) - ago5[1]) / ago5[1] * 100.0, 4)
            casc = snap.get("cascade_risk") or {}
            if casc.get("risk_score") is not None:
                out["cascade"] = {
                    "risk_score": casc.get("risk_score"),
                    "cascade_side": casc.get("cascade_side"),
                }

        # ── Liquidation flow (cascade-risk recent totals, 5m deltas) ─────────
        casc_full = _cv.cascade_risk("BTC")
        if isinstance(casc_full, dict):
            if out["cascade"] is None and casc_full.get("risk_score") is not None:
                out["cascade"] = {
                    "risk_score": casc_full.get("risk_score"),
                    "cascade_side": casc_full.get("cascade_side"),
                }
            recent = casc_full.get("recent") or {}
            ll_now, sl_now = (recent.get("long_liq_1h_usd"),
                              recent.get("short_liq_1h_usd"))
            ago5 = _hist_at(_liq_hist, 300.0, tol_s=90.0)
            dlong = dshort = None
            if ll_now is not None and sl_now is not None and ago5:
                dlong = max(0.0, float(ll_now) - ago5[1])
                dshort = max(0.0, float(sl_now) - ago5[2])
            elif recent.get("long_liq_15m_usd") is not None:
                # No history yet — approximate 5m as a third of the 15m totals
                dlong = float(recent.get("long_liq_15m_usd") or 0) / 3.0
                dshort = float(recent.get("short_liq_15m_usd") or 0) / 3.0
            if dlong is not None:
                adverse = dlong if side == "YES" else dshort
                support = dshort if side == "YES" else dlong
                out["liq_adverse_5m_k"] = round(adverse / 1000.0, 1)
                out["liq_support_5m_k"] = round(support / 1000.0, 1)

        # ── Taker flow / positioning / heatmap ───────────────────────────────
        flow = _cv.taker_flow_1m("BTC")
        if flow is not None:
            out["taker_1m"] = round(sgn * flow, 4)
        ls = _cv.global_ls_ratio("BTC")
        if ls is not None:
            out["ls_ratio"] = ls
        hm = _cv.heatmap("BTC")
        if hm:
            feats = heatmap_features(hm)
            if feats:
                out["liq_heatmap"] = feats
                if feats.get("liq_skew") is not None:
                    out["liq_skew"] = feats["liq_skew"]

        # ── Provisional LOG-ONLY verdicts (never enforced here) ──────────────
        if out["perp_m30s"] is not None:
            out["perp_veto"] = out["perp_m30s"] <= -2.0
        if out["liq_adverse_5m_k"] is not None:
            adv, sup = out["liq_adverse_5m_k"], out["liq_support_5m_k"] or 0.0
            out["liq_veto"] = adv > 250.0 and sup < adv / 2

        parts = []
        if out["perp_feed_ok"]:
            parts.append(f"perp m30s={out['perp_m30s']}bp imb={out['perp_imb']}")
        if out["cv_feed_ok"]:
            parts.append(f"liq_adv={out['liq_adverse_5m_k']}k "
                         f"taker1m={out['taker_1m']} oi_d5m={out['oi_d5m_pct']}%")
        if out["cv_momentum"] and out["cv_momentum"].get("aggregate_score") is not None:
            parts.append(f"cv_mom={out['cv_momentum']['aggregate_score']}")
        vetoes = []
        if out["perp_veto"]: vetoes.append("PERP-VETO")
        if out["liq_veto"]:  vetoes.append("LIQ-VETO")
        verdict = (" | would-block: " + "+".join(vetoes)) if vetoes else " | would-allow"
        out["summary"] = ("; ".join(parts) or "feeds unavailable") + verdict
    except Exception as e:   # absolute safety: never break the daemon
        out["summary"] = f"signal snapshot error: {e!r}"
    return out


def latest_perp_mid(max_age_s: float = 30.0) -> Optional[float]:
    """Most recent primary-venue perp mid from the lead feed, or None."""
    try:
        if _lead_feed is None:
            return None
        return _lead_feed.latest_mid("binance_futures", max_age_s=max_age_s)
    except Exception:
        return None


def rrm_evaluate(*, side: str, strike: float, btc_entry: float,
                 perp_mid_entry: float, mins_remaining: float) -> dict:
    """Reversal-Risk Monitor score for an ACTIVE position.

    Basis-free spot estimate: anchor on the oracle spot at entry, move it by
    the perp's relative change since entry:
        est_spot = btc_entry * (perp_mid_now / perp_mid_entry)

    MANDATORY GATE: breach — est_spot crossed >= 0.02% onto the LOSING side
    of the strike. Without breach the score is reported but capped.

    Score (max 10), would-exit at >= 6 WITH breach:
      +3 breach (the gate)
      +2 velocity: adverse perp move >= 8bp over 30s
      +2 liq cascade: adverse-side liquidations >= $50k over ~90s
      +1 taker surge: 1m taker flow >= 2:1 against position
      +1 OI rising over ~2 min (new money driving the move)
      +1 mins_remaining < 4 (no time left to mean-revert)

    Never raises. Returns dict(score, breach, would_exit, components, ...).
    """
    out = {"ok": False, "score": 0, "breach": False, "would_exit": False,
           "est_spot": None, "components": {}, "summary": "feeds unavailable"}
    try:
        if _lead_feed is None or not perp_mid_entry:
            return out
        mid_now = _lead_feed.latest_mid("binance_futures",
                                        max_age_s=PERP_STALE_S)
        if not mid_now:
            return out
        est_spot = btc_entry * (mid_now / perp_mid_entry)
        out["est_spot"] = round(est_spot, 2)
        out["ok"] = True

        comp = {}
        # ── Gate: strike breach on the losing side ────────────────────────
        margin = strike * 0.0002   # 0.02%
        if side == "NO":
            breach = est_spot >= strike + margin
        else:
            breach = est_spot <= strike - margin
        comp["breach"] = breach
        out["breach"] = breach

        # ── Velocity: adverse perp move over 30s ──────────────────────────
        move30 = _lead_move_bps(30.0)
        vel_bp = None
        if move30 is not None:
            vel_bp = move30 if side == "NO" else -move30   # adverse-positive
        comp["velocity_bp"] = round(vel_bp, 2) if vel_bp is not None else None
        comp["velocity"] = bool(vel_bp is not None and vel_bp >= 8.0)

        # ── ChainVector: liq cascade, taker, OI ───────────────────────────
        comp["liq_cascade"] = False
        comp["taker_surge"] = False
        comp["oi_rising"] = False
        casc = _cv.cascade_risk("BTC")
        if isinstance(casc, dict):
            recent = casc.get("recent") or {}
            ago90 = _hist_at(_liq_hist, 90.0, tol_s=45.0)
            key = "short_liq_1h_usd" if side == "NO" else "long_liq_1h_usd"
            now_val = recent.get(key)
            if now_val is not None and ago90:
                prev_val = ago90[2] if side == "NO" else ago90[1]
                adverse = max(0.0, float(now_val) - prev_val)
                comp["liq_adverse_90s_k"] = round(adverse / 1000.0, 1)
                comp["liq_cascade"] = adverse >= 50000.0
        flow = _cv.taker_flow_1m("BTC")
        if flow is not None:
            signed = flow if side == "YES" else -flow
            comp["taker_signed"] = round(signed, 3)
            comp["taker_surge"] = signed <= -(1.0 / 3.0)   # >= 2:1 against
        ago120 = _hist_at(_oi_hist, 120.0, tol_s=60.0)
        with _hist_lock:
            oi_latest = _oi_hist[-1] if _oi_hist else None
        if ago120 and oi_latest and ago120[1] > 0:
            comp["oi_d2m_pct"] = round(
                (oi_latest[1] - ago120[1]) / ago120[1] * 100.0, 4)
            comp["oi_rising"] = oi_latest[1] > ago120[1]

        comp["time_critical"] = mins_remaining < 4.0

        # ── Drift-aware P(end OTM) for predict-cross-exit ───────────────────
        try:
            tte_sec = max(0.0, mins_remaining * 60.0)
            if side == "NO":
                dist = (strike - est_spot) / est_spot
            else:
                dist = (est_spot - strike) / est_spot
            adverse_frac_30s = (vel_bp / 10000.0) if vel_bp is not None else 0.0
            # Horizon-capped + damped drift (see original hardening notes)
            _horizon = min(tte_sec, 120.0)
            drift = adverse_frac_30s * (_horizon / 30.0) * 0.6
            mean_dist = dist - drift
            # realized per-second perp vol over the last ~120s of mids
            sig_per_s = None
            mids = _lead_feed.mid_series("binance_futures", window_s=120.0)
            if len(mids) >= 3:
                sq = 0.0; dt = 0.0
                for (t0, m0), (t1, m1) in zip(mids, mids[1:]):
                    if m0 and m1 and t1 > t0:
                        rr = math.log(m1 / m0)
                        sq += rr * rr
                        dt += (t1 - t0)
                if dt > 0:
                    sig_per_s = math.sqrt(sq / dt)
            sigma_rem = (sig_per_s * math.sqrt(max(1.0, tte_sec))
                         if sig_per_s else 0.0)
            # Deep-ITM cushion floor — a comfortably ITM position never fires.
            if dist >= 0.0015:
                p_otm = 0.0
            elif sigma_rem > 0:
                p_otm = _norm_cdf(-mean_dist / sigma_rem)
            else:
                p_otm = 1.0 if mean_dist <= 0 else 0.0
            out["dist_pct"] = round(dist, 6)
            out["mean_dist_pct"] = round(mean_dist, 6)
            out["sigma_rem"] = round(sigma_rem, 6)
            out["p_otm"] = round(p_otm, 4)
            out["tte_min"] = round(mins_remaining, 3)
        except Exception:
            pass

        score = 0
        if comp["breach"]:        score += 3
        if comp["velocity"]:      score += 2
        if comp["liq_cascade"]:   score += 2
        if comp["taker_surge"]:   score += 1
        if comp["oi_rising"]:     score += 1
        if comp["time_critical"]: score += 1

        out["score"] = score
        out["components"] = comp
        out["would_exit"] = bool(comp["breach"] and score >= 6)
        flags = "+".join(k for k in ("breach", "velocity", "liq_cascade",
                                     "taker_surge", "oi_rising", "time_critical")
                         if comp.get(k))
        out["summary"] = (f"score={score}/10 [{flags or 'none'}] "
                          f"est_spot=${est_spot:,.0f} vs strike=${strike:,.0f}")
    except Exception as e:
        out["summary"] = f"rrm error: {e!r}"
    return out


def cv_composite(side: str, strike: float, spot: float, mins_left: float,
                 raw: dict) -> dict:
    """LOG-ONLY composite reversal/maturation score (never raises, never
    gates). Blends the leading ChainVector signals (liq_skew + book skew lead
    bullish over ~3-5m; OI momentum leads contrarian) into a directional
    conviction, signs it AGAINST the held side, then estimates the
    probability the contract FLIPS before expiry given the cushion to strike,
    realized spot drift (maturation) and short-horizon vol.

    raw holds RAW signals (+ = bullish for the asset):
      liq_skew, book_skew, oi_mom (%), cb_mom (%/min), perp (bp/min),
      sigma_pct_min (per-minute vol proxy).

    Weights/coupling are PROVISIONAL; every normalized component is logged so
    they can be re-fit offline as data accumulates.
    """
    out = {"ok": False}
    try:
        from math import erf, sqrt
        if not strike or not spot or spot <= 0 or mins_left is None:
            return out
        sgn = 1.0 if side == "YES" else -1.0
        W = {"liq_skew": 0.43, "book_skew": 0.27, "oi_mom": -0.35,
             "cb_mom": 0.50, "perp": 0.30}
        S = {"liq_skew": 0.30, "book_skew": 0.15, "oi_mom": 1.0,
             "cb_mom": 0.05, "perp": 5.0}
        comp, acc, wsum = {}, 0.0, 0.0
        for k in ("liq_skew", "book_skew", "oi_mom", "cb_mom", "perp"):
            v = raw.get(k)
            if v is None:
                continue
            z = v / S[k]
            comp[k] = round(z, 3)
            acc += W[k] * z
            wsum += abs(W[k])
        cv_dir = (acc / wsum) if wsum > 0 else 0.0            # + = bullish
        adverse = -sgn * cv_dir                                # + = against us
        cushion_pct = sgn * (spot - strike) / strike * 100.0   # + = ITM cushion
        cb = raw.get("cb_mom")
        drift_adv = (-sgn * cb) if cb is not None else 0.0     # %/min against us
        sigma = raw.get("sigma_pct_min") or 0.05
        T = max(0.05, float(mins_left))
        std_move = max(1e-6, sigma * sqrt(T))
        phi = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
        flip_prob = phi((drift_adv * T - cushion_pct) / std_move)
        k_sig = 0.04   # %/min adverse drift per unit adverse composite
        drift_sig = drift_adv + k_sig * max(0.0, adverse)
        flip_prob_sig = phi((drift_sig * T - cushion_pct) / std_move)
        out = {
            "ok": True,
            "cv_dir": round(cv_dir, 4),
            "adverse": round(adverse, 4),
            "cushion_pct": round(cushion_pct, 4),
            "mins_left": round(T, 2),
            "drift_adv_pct_min": round(drift_adv, 5),
            "sigma_pct_min": round(sigma, 5),
            "flip_prob": round(flip_prob, 4),
            "flip_prob_sig": round(flip_prob_sig, 4),
            "components": comp,
        }
    except Exception as e:
        out = {"ok": False, "err": repr(e)}
    return out


def context_signals(mins_left: Optional[float] = None) -> dict:
    """Recorded ChainVector context bundle (regime / volatility / risk index /
    prediction results are slower-moving reads). Fail-open; keys None when
    unavailable. Written into the audit stream at trade time for offline
    signal research."""
    out = {"regime": None, "volatility": None, "risk_index": None}
    try:
        reg = _cv.regime("BTC")
        if isinstance(reg, dict):
            out["regime"] = {
                "regime": reg.get("regime"),
                "direction": reg.get("direction"),
                "confidence": reg.get("confidence"),
                "hurst": (reg.get("inputs") or {}).get("hurst"),
                "variance_ratio": (reg.get("inputs") or {}).get("variance_ratio"),
            }
        vol = _cv.volatility("BTC", horizon_min=mins_left)
        if isinstance(vol, dict):
            em = vol.get("expected_move") or {}
            vp = vol.get("vol_percentile") or {}
            out["volatility"] = {
                "sigma_pct": em.get("sigma_pct"),
                "sigma_usd": em.get("sigma_usd"),
                "vol_percentile": vp.get("percentile"),
                "hurst": (vol.get("regime") or {}).get("hurst_exponent"),
            }
        risk = _cv.risk_index("BTC")
        if isinstance(risk, dict):
            assets = risk.get("assets") or []
            row = assets[0] if assets else None
            if isinstance(row, dict):
                out["risk_index"] = {"index": row.get("index"),
                                     "label": row.get("label")}
    except Exception:
        pass
    return out

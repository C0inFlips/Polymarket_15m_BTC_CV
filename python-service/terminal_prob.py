"""
terminal_prob.py — ChainVector-implied terminal probability for the 15m daemon
(was: Deribit options-chain SSVI/BL/MC calculator).

Computes the probability that BTC ends above a given strike at the Polymarket
close using the ChainVector Probability Engine (GET /api/v1/probability):
six estimators — Gaussian CDF, 5-state Markov transition matrix,
Black-Scholes digital Φ(d₂), GBM Monte Carlo, fat-tailed Student-t Monte
Carlo and an empirical bootstrap — blended into a convex ensemble, computed
on tick-derived 1-minute bars with the market's EXACT time-to-close
(close_ts, no bucket rounding).

This is a huge structural upgrade over the old Deribit approach: Deribit's
shortest option expiry is 3-24h out, so its IV structurally undershot a
15-minute binary. The ChainVector engine is built for 30s-24h horizons and
takes close_ts directly.

Output keys keep the ORIGINAL names so the daemon's 40+ read sites work
unchanged (the estimator mapping is documented below):
  bs_p_above / bs_p_below   — the ENSEMBLE probability (primary signal)
  bl_p_above / bl_p_below   — Gaussian-CDF estimator (cross-check #1)
  mc_p_above / mc_p_below   — Monte-Carlo mean (GBM + Student-t, #2)
  cross_method_max_disagreement — max spread across ALL six estimators
  deribit_iv_at_strike / deribit_iv_atm / sigma_used
                            — annualized vol implied by the engine's
                              sigma_horizon (kept under the legacy key names)
  deribit_expiry            — "cv-ensemble" tag (legacy key, no options leg)
  hours_to_deribit_expiry   — equals hours_to_venue_expiry (exact TTE!)
  cv_estimators             — the raw six-estimator dict (new, for audit)
  error                     — set when the API is unavailable (caller treats
                              TP as missing; all gates fail toward safety)

Public functions:
  compute_terminal_prob(spot, strike, venue_close_dt) -> dict
  save_snapshot(snapshot, ticker, log_dir) -> Optional[Path]
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from chainvector import get_client

_MINUTES_PER_YEAR = 365.25 * 24 * 60


def compute_terminal_prob(spot: float, strike: float,
                          venue_close_dt: datetime,
                          *, currency: str = "BTC") -> dict:
    """Compute the ChainVector ensemble probability that BTC > strike at
    venue_close_dt. Returns a flat dict (legacy-compatible keys, see module
    docstring). Sets `error` and returns early when the API is unavailable —
    the daemon treats that exactly like the old "Deribit fetch failed" path.
    """
    now = datetime.now(timezone.utc)
    out: dict = {
        "spot":   spot,
        "strike": strike,
        "venue_close_dt": venue_close_dt.isoformat(),
        "computed_at":     now.isoformat(),
        "prob_source":     "chainvector",
        "error":  None,
    }
    tte_s = (venue_close_dt - now).total_seconds()
    T_years = max(0.0, tte_s / (365.25 * 24 * 3600))
    out["hours_to_venue_expiry"] = T_years * 365.25 * 24
    out["hours_to_deribit_expiry"] = out["hours_to_venue_expiry"]  # exact TTE
    out["deribit_expiry"] = "cv-ensemble"

    cv = get_client()
    if not cv.enabled:
        out["error"] = "chainvector_disabled: CHAINVECTOR_API_KEY not set"
        return out
    if tte_s <= 25.0:
        out["error"] = f"tte_too_short: {tte_s:.0f}s to close"
        return out

    close_ts_ms = int(venue_close_dt.timestamp() * 1000)
    prob = cv.probability(target=float(strike), close_ts_ms=close_ts_ms,
                          coin=currency)
    if not prob:
        out["error"] = "chainvector_probability_unavailable"
        return out

    ensemble = float(prob["ensemble"])
    gaussian = prob.get("gaussian")
    mc_vals = [v for v in (prob.get("mc_gbm"), prob.get("mc_student_t"))
               if v is not None]
    mc_p = (sum(mc_vals) / len(mc_vals)) if mc_vals else ensemble
    bl_p = gaussian if gaussian is not None else ensemble

    out.update({
        "bs_p_above": ensemble, "bs_p_below": 1.0 - ensemble,
        "bl_p_above": bl_p,     "bl_p_below": 1.0 - bl_p,
        "mc_p_above": mc_p,     "mc_p_below": 1.0 - mc_p,
        "cv_estimators": {k: prob.get(k) for k in
                          ("gaussian", "markov", "bs_d2", "mc_gbm",
                           "mc_student_t", "bootstrap", "ensemble")},
        "cv_sample_bars": prob.get("sample_bars"),
    })

    # Cross-estimator consistency — max spread across available estimators.
    pvals = [v for v in out["cv_estimators"].values() if v is not None]
    max_disagreement = (max(pvals) - min(pvals)) if len(pvals) >= 2 else 0.0
    out["cross_method_max_disagreement"] = max_disagreement
    if max_disagreement > 0.15:
        out["warning"] = f"estimators disagree by {max_disagreement:.3f}"

    # Annualized vol implied by the engine's horizon sigma — kept under the
    # legacy IV key names so status lines / audit joins stay meaningful.
    iv = 0.0
    sigma_h = prob.get("sigma_horizon")
    horizon_min = max(tte_s / 60.0, 1e-6)
    if sigma_h:
        try:
            iv = float(sigma_h) * math.sqrt(_MINUTES_PER_YEAR / horizon_min)
        except (TypeError, ValueError):
            iv = 0.0
    out.update({
        "deribit_iv_atm":       iv,
        "deribit_iv_at_strike": iv,
        "sigma_used":           iv,
        "sigma_horizon":        sigma_h,
        "sigma_1m":             prob.get("sigma_1m"),
        "deribit_spot":         spot,
    })
    return out


# ── Snapshot persistence ──────────────────────────────────────────────────────
def save_snapshot(snapshot: dict, ticker: str, log_dir: Path) -> Optional[Path]:
    """Persist a terminal-prob snapshot to disk for future backtest analysis.
    Filename: cv_prob_snapshots/{YYYYMMDD_HHMMSS}_{ticker}.json
    """
    snap_dir = Path(log_dir) / "cv_prob_snapshots"
    try:
        snap_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_ticker = ticker.replace("/", "_")[:48]
    path = snap_dir / f"{ts}_{safe_ticker}.json"
    try:
        path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        return path
    except Exception:
        return None


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from datetime import timedelta

    _env_path = Path(__file__).parent.parent / ".env.local"
    if _env_path.exists():
        for line in _env_path.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    print("=" * 65)
    print(" Terminal Probability — ChainVector ensemble smoke test")
    print("=" * 65)

    cv = get_client()
    if not cv.enabled:
        raise SystemExit("CHAINVECTOR_API_KEY not set")

    snap = cv.signals_snapshot("BTC") or {}
    spot = float(snap.get("spot") or 0)
    if spot <= 0:
        raise SystemExit("could not read spot from /signals/snapshot")
    print(f"\nSpot: ${spot:,.2f}")

    now = datetime.now(timezone.utc)
    next_close = (now + timedelta(minutes=15)).replace(second=0, microsecond=0)
    nxt = ((next_close.minute // 15) + 1) * 15
    if nxt >= 60:
        next_close = next_close.replace(minute=0) + timedelta(hours=1)
    else:
        next_close = next_close.replace(minute=nxt)

    for offset_pct in [-0.10, -0.05, 0.00, +0.05, +0.10]:
        strike = round(spot * (1 + offset_pct / 100.0), 2)
        print(f"\n--- Strike ${strike:,.2f}  ({offset_pct:+.2f}% from spot) ---")
        r = compute_terminal_prob(spot, strike, next_close)
        if r.get("error"):
            print(f"  ERROR: {r['error']}")
            continue
        print(f"  TTE: {r['hours_to_venue_expiry']*60:.1f} min (exact close_ts)")
        print(f"  IV (annualized from sigma_horizon): {r['deribit_iv_at_strike']*100:.2f}%")
        print(f"  P(YES) ensemble={r['bs_p_above']*100:.2f}% "
              f"gaussian={r['bl_p_above']*100:.2f}% "
              f"mc={r['mc_p_above']*100:.2f}%  "
              f"(estimator spread: {r['cross_method_max_disagreement']*100:.2f}pp)")
        if "warning" in r:
            print(f"  WARN: {r['warning']}")

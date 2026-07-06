"""liq_heatmap.py — ChainVector liquidation-heatmap feature extractor.

Derives spot- and strike-relative features from the ChainVector estimated
resting liquidation heatmap (GET /api/v1/liquidations/heatmap). Positive
cross-venue OI deltas locate when positions were opened; an assumed leverage
mix places their liquidation prices above/below each entry; levels already
traded through are removed. Long stops cluster below spot, short stops above
— the big levels are "magnets" price tends to gravitate toward.

Heatmap shape (ChainVector):
  data = {
    "spot": float,
    "bins": [ { "price": float, "long_liq_usd": float, "short_liq_usd": float }, ... ],
    "max_long_cluster":  { "price": ..., "usd": ... },
    "max_short_cluster": { "price": ..., "usd": ... },
    "total_long_usd": float, "total_short_usd": float,
  }

Consumed by signal_feeds.snapshot_new_signals (liq_heatmap / liq_skew
features) which feed the flip-probability composite. Shadow/logging only.
"""
from __future__ import annotations

from typing import Optional


def _latest_levels(data: dict):
    """Return (spot, [(price, usd), ...] sorted by price asc) from a
    ChainVector heatmap response, or (None, []) if unusable. Each level's USD
    is the combined long+short resting liquidation notional at that price."""
    if not isinstance(data, dict):
        return None, []
    bins = data.get("bins") or []
    try:
        spot = float(data.get("spot") or 0.0)
    except (TypeError, ValueError):
        return None, []
    if spot <= 0 or not bins:
        return None, []
    levels: dict[float, float] = {}
    for b in bins:
        try:
            price = float(b.get("price"))
            usd = float(b.get("long_liq_usd") or 0) + float(b.get("short_liq_usd") or 0)
        except (TypeError, ValueError):
            continue
        if price > 0 and usd > 0:
            levels[price] = levels.get(price, 0.0) + usd
    if not levels:
        return None, []
    return spot, sorted(levels.items())


def _pct(price: Optional[float], ref: float) -> Optional[float]:
    if price is None or ref <= 0:
        return None
    return round((price - ref) / ref * 100.0, 4)


def heatmap_features(data: dict, top_n: int = 15, cluster_frac: float = 0.5) -> Optional[dict]:
    """Spot-relative standing-liquidation features. Returns a compact dict
    (safe to log every poll) or None if the heatmap is unusable.

    Keys:
      spot, n_levels
      liq_above_usd/liq_below_usd/liq_skew   (skew>0 => more fuel above spot)
      max_cluster_{price,usd,dist_pct}       (largest cluster overall)
      above_cluster_{price,usd,dist_pct}     (largest cluster strictly above)
      below_cluster_{price,usd,dist_pct}     (largest cluster strictly below)
      near_sig_above_pct/near_sig_below_pct  (nearest *significant* cluster each side)
      top_levels: [[price, usd], ...]        (top_n by usd; for strike joins)
    """
    try:
        spot, items = _latest_levels(data)
        if spot is None or not items:
            return None
        above = [(p, v) for p, v in items if p > spot]
        below = [(p, v) for p, v in items if p < spot]
        liq_above = sum(v for _, v in above)
        liq_below = sum(v for _, v in below)
        denom = liq_above + liq_below

        def biggest(seq):
            return max(seq, key=lambda t: t[1]) if seq else (None, 0.0)

        ap, av = biggest(above)
        bp, bv = biggest(below)
        gp, gv = biggest(items)
        thr = cluster_frac * gv if gv else 0.0
        sig_above = [(p, v) for p, v in above if v >= thr]
        sig_below = [(p, v) for p, v in below if v >= thr]
        nap = min((p for p, _ in sig_above), default=None)  # closest above
        nbp = max((p for p, _ in sig_below), default=None)  # closest below
        top = sorted(items, key=lambda t: -t[1])[:top_n]
        return {
            "spot": round(spot, 8),
            "n_levels": len(items),
            "liq_above_usd": round(liq_above, 0),
            "liq_below_usd": round(liq_below, 0),
            "liq_skew": round((liq_above - liq_below) / denom, 4) if denom > 0 else 0.0,
            "max_cluster_price": round(gp, 8) if gp else None,
            "max_cluster_usd": round(gv, 0),
            "max_cluster_dist_pct": _pct(gp, spot),
            "above_cluster_price": round(ap, 8) if ap else None,
            "above_cluster_usd": round(av, 0),
            "above_cluster_dist_pct": _pct(ap, spot),
            "below_cluster_price": round(bp, 8) if bp else None,
            "below_cluster_usd": round(bv, 0),
            "below_cluster_dist_pct": _pct(bp, spot),
            "near_sig_above_pct": _pct(nap, spot),
            "near_sig_below_pct": _pct(nbp, spot),
            "top_levels": [[round(p, 8), round(v, 0)] for p, v in top],
        }
    except Exception:
        return None


def strike_metrics(data: dict, strike: float, side: Optional[str] = None,
                   cluster_frac: float = 0.5) -> Optional[dict]:
    """Nearest-cluster-vs-STRIKE features. `side`, if given ("YES" wants
    price ABOVE strike, "NO" wants BELOW), adds a directional `cluster_pull`
    read: + = the dominant nearby magnet pulls toward our winning side,
    − = against us. Returns compact dict or None.
    """
    try:
        spot, items = _latest_levels(data)
        if spot is None or not items or strike <= 0:
            return None
        liq_above = sum(v for p, v in items if p > strike)
        liq_below = sum(v for p, v in items if p < strike)
        denom = liq_above + liq_below
        gp, gv = max(items, key=lambda t: t[1])
        thr = cluster_frac * gv if gv else 0.0
        sig = [(p, v) for p, v in items if v >= thr]
        above = [(p, v) for p, v in sig if p > strike]
        below = [(p, v) for p, v in sig if p < strike]
        na = min(above, key=lambda t: t[0] - strike) if above else (None, 0.0)
        nb = max(below, key=lambda t: t[0]) if below else (None, 0.0)
        magnet_side = "above" if gp > strike else ("below" if gp < strike else "at")
        out = {
            "spot": round(spot, 8),
            "strike": round(float(strike), 8),
            "spot_vs_strike_pct": _pct(spot, strike),
            "liq_above_strike_usd": round(liq_above, 0),
            "liq_below_strike_usd": round(liq_below, 0),
            "liq_skew_strike": round((liq_above - liq_below) / denom, 4) if denom > 0 else 0.0,
            "magnet_price": round(gp, 8),
            "magnet_usd": round(gv, 0),
            "magnet_dist_from_strike_pct": _pct(gp, strike),
            "magnet_side": magnet_side,
            "near_above_strike_price": round(na[0], 8) if na[0] else None,
            "near_above_strike_usd": round(na[1], 0),
            "near_above_strike_pct": _pct(na[0], strike),
            "near_below_strike_price": round(nb[0], 8) if nb[0] else None,
            "near_below_strike_usd": round(nb[1], 0),
            "near_below_strike_pct": _pct(nb[0], strike),
        }
        if side:
            s = side.upper()
            wants_above = s in ("YES", "UP", "ABOVE", "Y")
            skew = out["liq_skew_strike"]
            out["cluster_pull"] = round(skew if wants_above else -skew, 4)
            out["magnet_favors_us"] = (magnet_side == ("above" if wants_above else "below"))
        return out
    except Exception:
        return None

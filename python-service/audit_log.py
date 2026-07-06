"""
audit_log.py — Structured per-decision JSONL audit log for the 15m daemon

For every signal evaluation, write one JSON object to
``logs/audit_YYYYMMDD.jsonl`` capturing the FULL state of every gate input
so we can do post-hoc analysis without parsing the human-readable daemon log.

Each line is a self-contained JSON object. Decision types:
  • "POLL"       — daemon evaluated a signal (may or may not have traded)
  • "TRADE"      — order was submitted (includes fill details)
  • "SETTLED"    — window closed and Polymarket settled, we recorded P&L
  • "SAFETY_SKIP" — daemon skipped before evaluating (history-building, etc.)

Loading for analysis:
    import pandas as pd
    df = pd.read_json("logs/audit_20260526.jsonl", lines=True)
    df.query('decision == "TRADE"').describe()
    df[df.outcome == "WIN"].groupby('side').mean()

Or via jq:
    cat logs/audit_20260526.jsonl | jq -c 'select(.type == "TRADE")'

Crash-safe: every write is flushed to disk immediately. File handle is closed
properly on daemon shutdown via atexit.
"""

from __future__ import annotations

import atexit
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class AuditLogger:
    """Append-only JSONL audit log, thread-safe, crash-safe (flush per write)."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self._disabled = False
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            # Non-root-safe: degrade to no-op audit rather than crash the
            # daemon in a read-only / root-owned deployment directory.
            print(f"[audit_log] cannot create {self.log_dir} ({e}); "
                  f"audit logging DISABLED. Set BTC15M_DATA_DIR to a "
                  f"writable path to enable.", flush=True)
            self._disabled = True
        self._lock = threading.Lock()
        self._file_handle = None
        self._current_date_str: Optional[str] = None
        atexit.register(self.close)

    def _ensure_handle(self):
        """Open today's audit file, rolling over at midnight UTC."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        if today != self._current_date_str:
            if self._file_handle:
                try:
                    self._file_handle.close()
                except Exception:
                    pass
            path = self.log_dir / f"audit_{today}.jsonl"
            self._file_handle = open(path, "a", encoding="utf-8", buffering=1)
            self._current_date_str = today

    def write(self, record: dict):
        """Serialize and append a single JSON object. Adds ISO timestamp + epoch."""
        if self._disabled:
            return
        with self._lock:
            try:
                self._ensure_handle()
            except (PermissionError, OSError) as e:
                print(f"[audit_log] open failed: {e}; audit logging DISABLED.",
                      flush=True)
                self._disabled = True
                return
            now = datetime.now(timezone.utc)
            record["ts"]    = now.isoformat()
            record["ts_ms"] = int(now.timestamp() * 1000)
            try:
                self._file_handle.write(json.dumps(record, default=str))
                self._file_handle.write("\n")
                self._file_handle.flush()
            except Exception as e:
                # Don't let audit failures break the daemon
                print(f"[audit_log] write failed: {e}", flush=True)

    def close(self):
        with self._lock:
            if self._file_handle:
                try:
                    self._file_handle.close()
                except Exception:
                    pass
                self._file_handle = None


# ── Helper builders ───────────────────────────────────────────────────────────
def build_poll_record(*, window_id: str, ticker: str, close_dt: datetime,
                      decision: str, rec: Optional[str], reasons: list,
                      signal: dict, market: dict, recent_5m_pct: list,
                      tp_cached: Optional[dict], db_check: Optional[dict],
                      ev_calc: Optional[dict],
                      params: dict,
                      okx_check: Optional[dict] = None,
                      orderbook: Optional[dict] = None,
                      trade_flow: Optional[dict] = None,
                      extra: Optional[dict] = None) -> dict:
    """Compose a structured POLL record. All daemon state at decision time."""
    rec_dict: dict[str, Any] = {
        "type":            "POLL",
        "window_id":       window_id,
        "ticker":          ticker,
        "close_dt":        close_dt.isoformat() if close_dt else None,
        "decision":        decision,        # "TRADE" | "NO_TRADE" | "SKIP"
        "recommendation":  rec,             # "YES" | "NO" | None
        "rejection_reasons": list(reasons or []),
        "market": {
            "btc_price":      market.get("btc_price"),
            "strike":         market.get("strike"),
            "dist_pct":       market.get("dist_pct"),
            "dist_dollars":   round(market["btc_price"] - market["strike"], 2)
                              if market.get("btc_price") and market.get("strike") else None,
            "yes_ask":        market.get("yes_ask"),
            "no_ask":         market.get("no_ask"),
            "yes_bid":        market.get("yes_bid"),
            "no_bid":         market.get("no_bid"),
        },
        "signal": {
            "minutes_left":  signal.get("minutes_left"),
            "is_golden":     signal.get("is_golden"),
            "utc_hour":      signal.get("utc_hour"),
            "p_yes":         signal.get("p_yes"),
            "gap":           signal.get("gap"),
            "persist":       signal.get("persist"),
            "hurst":         signal.get("hurst"),
            "gk_vol":        signal.get("gk_vol"),
            "d_score":       signal.get("d_score"),
            "history_len":   signal.get("history_len"),
            "recent_5m_pct": [round(p, 4) for p in (recent_5m_pct or [])],
        },
        "params":            params,
    }

    if tp_cached and not tp_cached.get("error"):
        age_s = time.time() - tp_cached.get("_computed_at_unix", time.time())
        rec_dict["tp"] = {
            "available":       True,
            "fresh":           age_s < 5,
            "age_s":           round(age_s, 1),
            "bs_p_above":      tp_cached.get("bs_p_above"),
            "bl_p_above":      tp_cached.get("bl_p_above"),
            "mc_p_above":      tp_cached.get("mc_p_above"),
            "iv_atm":          tp_cached.get("deribit_iv_atm"),
            "iv_at_strike":    tp_cached.get("deribit_iv_at_strike"),
            "deribit_expiry":  tp_cached.get("deribit_expiry"),
            "deribit_hours":   tp_cached.get("hours_to_deribit_expiry"),
            "cross_method_disagreement": tp_cached.get("cross_method_max_disagreement"),
            "smile_a":         tp_cached.get("smile_a"),
            "smile_b":         tp_cached.get("smile_b"),
            "smile_c":         tp_cached.get("smile_c"),
            "smile_n_strikes": tp_cached.get("smile_n_strikes"),
            "smile_rms":       tp_cached.get("smile_rms_residual"),
        }
    else:
        rec_dict["tp"] = {
            "available": False,
            "error":     (tp_cached or {}).get("error") if tp_cached else "not_computed",
        }

    if db_check:
        # Futures lead-lag consistency check (ChainVector /momentum,
        # binance_futures venue).
        rec_dict["futures_lead"] = {
            "consistent":  db_check.get("consistent"),
            "reason":      db_check.get("reason"),
            "connected":   db_check.get("connected"),
            "venue":       db_check.get("venue") or db_check.get("front_month"),
            "stale_s":     db_check.get("stale_s"),
            "n_ticks":     db_check.get("n_ticks"),
            "move_pct":    db_check.get("move_pct"),
            "current_mid": db_check.get("current_mid"),
            "past_mid":    db_check.get("past_mid"),
            "momentum_score": db_check.get("momentum_score"),
        }
    else:
        rec_dict["futures_lead"] = {"enabled": False}

    if ev_calc:
        rec_dict["ev_calc"] = ev_calc

    if okx_check:
        # OKX lead-lag snapshot. Mirrors the futures_lead structure.
        rec_dict["okx"] = {
            "valid":       okx_check.get("valid"),
            "connected":   okx_check.get("connected"),
            "n_ticks":     okx_check.get("n_ticks"),
            "stale_s":     okx_check.get("stale_s"),
            "current_mid": okx_check.get("current_mid"),
            "past_mid":    okx_check.get("past_mid"),
            "move_pct":    okx_check.get("move_pct"),
        }
    else:
        rec_dict["okx"] = {"enabled": False}

    # Polymarket orderbook depth snapshot (added 2026-05-28, record-only).
    # Captures top-of-book + depth + imbalance for later analysis.
    if orderbook:
        rec_dict["orderbook"] = orderbook
    if trade_flow:
        rec_dict["trade_flow"] = trade_flow

    # Free-form extra signal payloads (e.g. ChainVector momentum/stability
    # snapshots attached at veto time).
    if extra:
        for k, v in extra.items():
            rec_dict.setdefault(k, v)

    return rec_dict


def build_trade_record(*, window_id: str, ticker: str, side: str,
                       limit_price_cents: int, contracts_requested: int,
                       contracts_filled: int, avg_fill_yes_leg: float,
                       order_ids: list, chunks: list,
                       retry_fired: bool, retry_filled: bool,
                       max_loss: float, expected_value: float,
                       params: dict) -> dict:
    return {
        "type":               "TRADE",
        "window_id":          window_id,
        "ticker":             ticker,
        "side":               side,
        "limit_price_cents":  limit_price_cents,
        "contracts_requested": contracts_requested,
        "contracts_filled":   contracts_filled,
        "avg_fill_yes_leg":   round(avg_fill_yes_leg, 4) if avg_fill_yes_leg else None,
        "max_loss_usd":       max_loss,
        "expected_value_usd": expected_value,
        "order_ids":          list(order_ids or []),
        "chunks":             list(chunks or []),
        "retry_fired":        retry_fired,
        "retry_filled":       retry_filled,
        "params":             params,
    }


def build_settlement_record(*, window_id: str, ticker: str, side: str,
                            result: str, contracts: int,
                            pnl_usd: float, daily_pnl_usd: float,
                            limit_price_cents: int) -> dict:
    return {
        "type":              "SETTLED",
        "window_id":         window_id,
        "ticker":            ticker,
        "side":              side,
        "result":            result,                   # "YES" | "NO"
        "won":               (side == result),
        "contracts":         contracts,
        "limit_price_cents": limit_price_cents,
        "pnl_usd":           pnl_usd,
        "daily_pnl_usd":     daily_pnl_usd,
    }

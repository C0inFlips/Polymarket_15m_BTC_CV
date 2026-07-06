"""paths.py — writable data/log directory selection.

The program is often deployed inside containers/VMs where the package
directory is owned by root but the process runs as a non-root user. Writing
logs/caches to Path(__file__).parent would raise PermissionError there.

Resolution order:
  1. $BTC15M_DATA_DIR (explicit override)
  2. <package dir>          (normal local runs)
  3. ~/.polymarket_btc_15m_cv   (home fallback)
  4. <tmp>/polymarket_btc_15m_cv (last resort)

Every candidate is verified with a real write probe before being chosen.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

_ENV_VAR = "BTC15M_DATA_DIR"
_cached: Path | None = None


def _writable(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def data_dir() -> Path:
    """First writable data directory from the resolution order (cached)."""
    global _cached
    if _cached is not None:
        return _cached
    candidates = []
    env = os.environ.get(_ENV_VAR, "").strip()
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parent)
    home = Path.home()
    if str(home) not in ("", "/"):
        candidates.append(home / ".polymarket_btc_15m_cv")
    candidates.append(Path(tempfile.gettempdir()) / "polymarket_btc_15m_cv")
    for cand in candidates:
        if _writable(cand):
            _cached = cand
            return cand
    _cached = candidates[-1]
    return _cached


def logs_dir() -> Path:
    """logs/ under the data dir (created, may still fail on exotic setups —
    callers must degrade gracefully)."""
    p = data_dir() / "logs"
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p

"""
Portfolio Persistence Store — Multi-Portfolio Edition
Supports multiple named portfolios stored as individual JSON files.

Storage layout:
  ~/.portfolio_manager/portfolios/{name}.json   — named portfolio files
  ~/.portfolio_manager/active_portfolio.txt      — active portfolio name

Backward compat:
  ~/.portfolio_manager/portfolio.json is auto-migrated to
  portfolios/Default.json on first access if no portfolios dir exists yet.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE           = Path.home() / ".portfolio_manager"
_PORTFOLIOS_DIR = _BASE / "portfolios"
_ACTIVE_FILE    = _BASE / "active_portfolio.txt"
_LEGACY_FILE    = _BASE / "portfolio.json"      # pre-Phase 4 single-portfolio file

DEFAULT_NAME    = "Default"


# ── Migration ─────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    _BASE.mkdir(parents=True, exist_ok=True)
    _PORTFOLIOS_DIR.mkdir(parents=True, exist_ok=True)

    # Migrate legacy single-portfolio file → portfolios/Default.json
    if _LEGACY_FILE.exists() and not (_PORTFOLIOS_DIR / f"{DEFAULT_NAME}.json").exists():
        try:
            shutil.copy2(_LEGACY_FILE, _PORTFOLIOS_DIR / f"{DEFAULT_NAME}.json")
        except Exception:
            pass


def _portfolio_path(name: str) -> Path:
    _ensure_dirs()
    safe = "".join(c for c in name if c.isalnum() or c in " _-").strip() or "Portfolio"
    return _PORTFOLIOS_DIR / f"{safe}.json"


# ── Active portfolio ──────────────────────────────────────────────────────────

def get_active_portfolio() -> str:
    """Return the currently active portfolio name."""
    try:
        if _ACTIVE_FILE.exists():
            name = _ACTIVE_FILE.read_text().strip()
            if name:
                return name
    except Exception:
        pass
    return DEFAULT_NAME


def set_active_portfolio(name: str) -> None:
    """Set the active portfolio name."""
    _ensure_dirs()
    _ACTIVE_FILE.write_text(name)


# ── Portfolio list ────────────────────────────────────────────────────────────

def list_portfolios() -> list[dict]:
    """
    Return a list of all portfolios as dicts with keys:
      name, description, strategy, risk_level, created_at, updated_at, num_positions
    Sorted alphabetically, Default first.
    """
    _ensure_dirs()
    result = []
    for path in sorted(_PORTFOLIOS_DIR.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            name = path.stem
            positions = data.get("positions", [])
            result.append({
                "name":          name,
                "description":   data.get("description", ""),
                "strategy":      data.get("strategy", ""),
                "risk_level":    data.get("risk_level", ""),
                "created_at":    data.get("created_at", ""),
                "updated_at":    data.get("saved_at", ""),
                "num_positions": len(positions),
            })
        except Exception:
            pass

    # Put Default first
    result.sort(key=lambda x: (x["name"] != DEFAULT_NAME, x["name"]))
    return result


def portfolio_exists(name: str) -> bool:
    return _portfolio_path(name).exists()


# ── Save / load ───────────────────────────────────────────────────────────────

def save_portfolio(
    positions_df: pd.DataFrame,
    name: Optional[str] = None,
    description: str = "",
    strategy: str = "",
    risk_level: str = "",
) -> bool:
    """
    Save equity positions to a named portfolio file.
    name=None → uses the active portfolio name.
    Preserves existing metadata and options_positions.
    """
    if name is None:
        name = get_active_portfolio()
    path = _portfolio_path(name)

    try:
        existing: dict = {}
        if path.exists():
            with open(path) as f:
                existing = json.load(f)

        existing["saved_at"]  = datetime.now().isoformat()
        existing["positions"] = positions_df.to_dict(orient="records")

        if not existing.get("created_at"):
            existing["created_at"] = existing["saved_at"]
        if description:
            existing["description"] = description
        if strategy:
            existing["strategy"] = strategy
        if risk_level:
            existing["risk_level"] = risk_level

        with open(path, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving portfolio '{name}': {e}")
        return False


def load_portfolio(name: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load equity positions from a named portfolio.
    name=None → uses the active portfolio name.
    Falls back to legacy portfolio.json if named file not found.
    """
    if name is None:
        name = get_active_portfolio()
    path = _portfolio_path(name)

    # Fallback to legacy file for Default
    if not path.exists() and name == DEFAULT_NAME and _LEGACY_FILE.exists():
        path = _LEGACY_FILE

    try:
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        df = pd.DataFrame(data.get("positions", []))
        if df.empty:
            return None
        df["shares"]     = pd.to_numeric(df["shares"],     errors="coerce")
        df["cost_basis"] = pd.to_numeric(df["cost_basis"], errors="coerce")
        return df
    except Exception as e:
        print(f"Error loading portfolio '{name}': {e}")
        return None


def get_portfolio_metadata(name: Optional[str] = None) -> dict:
    """Return metadata (description, strategy, risk_level, etc.) for a portfolio."""
    if name is None:
        name = get_active_portfolio()
    path = _portfolio_path(name)
    try:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return {
                "name":        name,
                "description": data.get("description", ""),
                "strategy":    data.get("strategy", ""),
                "risk_level":  data.get("risk_level", ""),
                "created_at":  data.get("created_at", ""),
                "updated_at":  data.get("saved_at", ""),
            }
    except Exception:
        pass
    return {"name": name, "description": "", "strategy": "", "risk_level": "", "created_at": "", "updated_at": ""}


def update_portfolio_metadata(
    name: str,
    description: str = "",
    strategy: str = "",
    risk_level: str = "",
) -> bool:
    """Update only the metadata fields of an existing portfolio."""
    path = _portfolio_path(name)
    try:
        data: dict = {}
        if path.exists():
            with open(path) as f:
                data = json.load(f)
        if description is not None:
            data["description"] = description
        if strategy is not None:
            data["strategy"] = strategy
        if risk_level is not None:
            data["risk_level"] = risk_level
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception:
        return False


def create_portfolio(
    name: str,
    description: str = "",
    strategy: str = "",
    risk_level: str = "",
) -> bool:
    """Create an empty portfolio file.  Returns False if name already exists."""
    path = _portfolio_path(name)
    if path.exists():
        return False
    try:
        with open(path, "w") as f:
            json.dump({
                "created_at":  datetime.now().isoformat(),
                "saved_at":    datetime.now().isoformat(),
                "description": description,
                "strategy":    strategy,
                "risk_level":  risk_level,
                "positions":   [],
            }, f, indent=2)
        return True
    except Exception:
        return False


def delete_portfolio(name: str) -> bool:
    """Delete a portfolio file.  Cannot delete the active portfolio."""
    if name == DEFAULT_NAME:
        return False   # protect Default
    path = _portfolio_path(name)
    try:
        if path.exists():
            path.unlink()
        # If this was active, reset to Default
        if get_active_portfolio() == name:
            set_active_portfolio(DEFAULT_NAME)
        return True
    except Exception:
        return False


def rename_portfolio(old_name: str, new_name: str) -> bool:
    """Rename a portfolio (cannot rename Default)."""
    if old_name == DEFAULT_NAME:
        return False
    old_path = _portfolio_path(old_name)
    new_path = _portfolio_path(new_name)
    if not old_path.exists() or new_path.exists():
        return False
    try:
        old_path.rename(new_path)
        if get_active_portfolio() == old_name:
            set_active_portfolio(new_name)
        return True
    except Exception:
        return False


# ── Options positions ─────────────────────────────────────────────────────────

def save_options_positions(options: list, name: Optional[str] = None) -> bool:
    """Save options positions into the active (or named) portfolio file."""
    if name is None:
        name = get_active_portfolio()
    path = _portfolio_path(name)
    try:
        _ensure_dirs()
        data: dict = {}
        if path.exists():
            with open(path) as f:
                data = json.load(f)
        data["options_positions"] = options
        data["options_saved_at"]  = datetime.now().isoformat()
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving options: {e}")
        return False


def load_options_positions(name: Optional[str] = None) -> list:
    """Load options positions from the active (or named) portfolio."""
    if name is None:
        name = get_active_portfolio()
    path = _portfolio_path(name)

    if not path.exists() and name == DEFAULT_NAME and _LEGACY_FILE.exists():
        path = _LEGACY_FILE

    try:
        if not path.exists():
            return []
        with open(path) as f:
            data = json.load(f)
        return data.get("options_positions", [])
    except Exception:
        return []


# ── Legacy helpers (backward-compatible) ─────────────────────────────────────

def get_last_saved_time(name: Optional[str] = None) -> Optional[str]:
    if name is None:
        name = get_active_portfolio()
    path = _portfolio_path(name)
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f).get("saved_at")
    except Exception:
        pass
    return None


def portfolio_file_exists(name: Optional[str] = None) -> bool:
    if name is None:
        name = get_active_portfolio()
    return _portfolio_path(name).exists() or (
        name == DEFAULT_NAME and _LEGACY_FILE.exists()
    )


def restore_portfolio_to_session(name: Optional[str] = None) -> bool:
    """
    If st.session_state['positions'] is not set, load from the active
    (or named) portfolio file.  Called at the top of pages needing positions.
    Returns True if positions are now available.
    """
    import streamlit as st

    if st.session_state.get("positions") is not None:
        return True

    df = load_portfolio(name)
    if df is not None and not df.empty:
        st.session_state["positions"] = df
        if not st.session_state.get("options_positions"):
            st.session_state["options_positions"] = load_options_positions(name)
        return True
    return False

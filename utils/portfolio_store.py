"""
Portfolio Persistence Store
Save/load portfolio positions to/from a local JSON file
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd


_STORE_PATH = Path.home() / ".portfolio_manager" / "portfolio.json"


def save_portfolio(positions_df: pd.DataFrame) -> bool:
    """Save portfolio positions to JSON file. Returns True on success.
    Preserves any existing options_positions data in the file."""
    try:
        _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Preserve existing data (e.g. options_positions) before overwriting
        existing: dict = {}
        if _STORE_PATH.exists():
            try:
                with open(_STORE_PATH, "r") as f:
                    existing = json.load(f)
            except Exception:
                pass
        existing["saved_at"] = datetime.now().isoformat()
        existing["positions"] = positions_df.to_dict(orient="records")
        with open(_STORE_PATH, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving portfolio: {e}")
        return False


def load_portfolio() -> Optional[pd.DataFrame]:
    """Load portfolio positions from JSON file. Returns None if not found."""
    try:
        if not _STORE_PATH.exists():
            return None
        with open(_STORE_PATH, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data["positions"])
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
        df["cost_basis"] = pd.to_numeric(df["cost_basis"], errors="coerce")
        return df
    except Exception as e:
        print(f"Error loading portfolio: {e}")
        return None


def get_last_saved_time() -> Optional[str]:
    """Return ISO-format timestamp of last save, or None."""
    try:
        if not _STORE_PATH.exists():
            return None
        with open(_STORE_PATH, "r") as f:
            data = json.load(f)
        return data.get("saved_at")
    except Exception:
        return None


def portfolio_file_exists() -> bool:
    """Return True if a saved portfolio file exists."""
    return _STORE_PATH.exists()


def save_options_positions(options: list) -> bool:
    """Append/overwrite options positions into the portfolio JSON file."""
    try:
        _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {}
        if _STORE_PATH.exists():
            with open(_STORE_PATH, "r") as f:
                data = json.load(f)
        data["options_positions"] = options
        data["options_saved_at"] = datetime.now().isoformat()
        with open(_STORE_PATH, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving options: {e}")
        return False


def load_options_positions() -> list:
    """Load options positions from the portfolio JSON file."""
    try:
        if not _STORE_PATH.exists():
            return []
        with open(_STORE_PATH, "r") as f:
            data = json.load(f)
        return data.get("options_positions", [])
    except Exception as e:
        print(f"Error loading options: {e}")
        return []

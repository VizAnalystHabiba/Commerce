"""
utils/helpers.py — Commerce360 AI Helper Utilities
Formatting, download helpers, and shared UI utilities.
"""

import pandas as pd
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config.config import CURRENCY_SYMBOL


def fmt_currency(value: float) -> str:
    """Format number as Brazilian Real currency string."""
    if value >= 1_000_000:
        return f"{CURRENCY_SYMBOL} {value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{CURRENCY_SYMBOL} {value/1_000:.1f}K"
    else:
        return f"{CURRENCY_SYMBOL} {value:,.2f}"


def fmt_number(value: float) -> str:
    """Format large numbers with K/M suffix."""
    if value >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:,.0f}"


def fmt_pct(value: float) -> str:
    """Format percentage."""
    return f"{value:.1f}%"


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to UTF-8 CSV bytes for Streamlit download buttons."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    """Convert DataFrame to Excel bytes for Streamlit download buttons."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


def safe_divide(numerator: float, denominator: float,
                default: float = 0.0) -> float:
    """Safe division that returns default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def deduplicate_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per order_id (keep first).
    Use when computing order-level metrics from item-level tables.
    """
    return df.drop_duplicates(subset="order_id", keep="first")


def color_delta(value: float, positive_good: bool = True) -> str:
    """Return 'normal', 'inverse', or 'off' for Streamlit metric delta."""
    if positive_good:
        return "normal" if value >= 0 else "inverse"
    else:
        return "inverse" if value >= 0 else "normal"


def trend_label(values: pd.Series) -> str:
    """Return a simple trend description for a time series."""
    if len(values) < 2:
        return "Insufficient data"
    change = (values.iloc[-1] - values.iloc[-3]) / (abs(values.iloc[-3]) + 1e-9)
    if change > 0.05:
        return f"Trending UP (+{change*100:.1f}% last period)"
    elif change < -0.05:
        return f"Trending DOWN ({change*100:.1f}% last period)"
    else:
        return "Stable"

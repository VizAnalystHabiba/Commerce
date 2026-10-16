"""
utils/data_loader.py — Commerce360 AI Data Loading Layer (Flask Optimized)
Loads, caches, and validates all processed CSV files in memory.
Framework-agnostic caching using standard Python in-memory storage.
"""

import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config.config import FILES

# In-memory dataframe cache for high-speed Flask responses
_MEMORY_CACHE = {}


def _load(key: str) -> pd.DataFrame:
    if key in _MEMORY_CACHE:
        return _MEMORY_CACHE[key]

    path = FILES.get(key)
    if path is None:
        raise ValueError(f"Unknown data key: '{key}'")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    _MEMORY_CACHE[key] = df
    return df


def load_sales_features() -> pd.DataFrame:
    df = _load("sales_features").copy()
    if "order_purchase_timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["order_purchase_timestamp"]):
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    return df


def load_sales_table() -> pd.DataFrame:
    df = _load("sales_table").copy()
    if "order_purchase_timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["order_purchase_timestamp"]):
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    return df


def load_customer_features() -> pd.DataFrame:
    df = _load("customer_features").copy()
    if "last_purchase_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["last_purchase_date"]):
        df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"], errors="coerce")
    return df


def load_customer_segments() -> pd.DataFrame:
    return _load("customer_segments").copy()


def load_customer_table() -> pd.DataFrame:
    return _load("customer_table").copy()


def load_product_features() -> pd.DataFrame:
    return _load("product_features").copy()


def load_product_table() -> pd.DataFrame:
    return _load("product_table").copy()


def load_seller_features() -> pd.DataFrame:
    return _load("seller_features").copy()


def load_sellers_cleaned() -> pd.DataFrame:
    return _load("sellers_cleaned").copy()


def load_delivery_features() -> pd.DataFrame:
    df = _load("delivery_features").copy()
    if "order_purchase_timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["order_purchase_timestamp"]):
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    if "order_estimated_delivery_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["order_estimated_delivery_date"]):
        df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"], errors="coerce")
    return df


def load_review_table() -> pd.DataFrame:
    df = _load("review_table").copy()
    if "review_creation_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["review_creation_date"]):
        df["review_creation_date"] = pd.to_datetime(df["review_creation_date"], errors="coerce")
    if "order_purchase_timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["order_purchase_timestamp"]):
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    return df


def load_payment_table() -> pd.DataFrame:
    df = _load("payment_table").copy()
    if "order_purchase_timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["order_purchase_timestamp"]):
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    return df


def load_ml_results() -> pd.DataFrame:
    return _load("ml_model_results").copy()


def load_orders_cleaned() -> pd.DataFrame:
    df = _load("orders_cleaned").copy()
    if "order_purchase_timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["order_purchase_timestamp"]):
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    return df


def clear_cache():
    """Clear in-memory cache to force re-reading from disk."""
    global _MEMORY_CACHE
    _MEMORY_CACHE.clear()

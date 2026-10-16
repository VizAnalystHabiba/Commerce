"""
config/config.py — Commerce360 AI Configuration
All paths, constants, and theme settings in one place.
"""

from pathlib import Path
import os

# ── Project Paths ─────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent   # e:/Commerce360
DATA_DIR      = ROOT_DIR / "data" / "processed"
RAW_DIR       = ROOT_DIR / "data" / "raw"

# ── Processed Files ───────────────────────────────────────────────────────────
FILES = {
    "sales_features":     DATA_DIR / "sales_features.csv",
    "sales_table":        DATA_DIR / "sales_table.csv",
    "customer_features":  DATA_DIR / "customer_features.csv",
    "customer_segments":  DATA_DIR / "customer_segments.csv",
    "customer_table":     DATA_DIR / "customer_table.csv",
    "product_features":   DATA_DIR / "product_features.csv",
    "product_table":      DATA_DIR / "product_table.csv",
    "seller_features":    DATA_DIR / "seller_features.csv",
    "delivery_features":  DATA_DIR / "delivery_features.csv",
    "delivery_table":     DATA_DIR / "delivery_table.csv",
    "review_table":       DATA_DIR / "review_table.csv",
    "review_features":    DATA_DIR / "review_features.csv",
    "payment_table":      DATA_DIR / "payment_table.csv",
    "payment_features":   DATA_DIR / "payment_features.csv",
    "ml_model_results":   DATA_DIR / "ml_model_results.csv",
    "orders_cleaned":     DATA_DIR / "orders_cleaned.csv",
    "customers_cleaned":  DATA_DIR / "customers_cleaned.csv",
    "sellers_cleaned":    DATA_DIR / "sellers_cleaned.csv",
}

# ── GenAI Configuration ───────────────────────────────────────────────────────
# Set GEMINI_API_KEY in a .env file or as an environment variable.
# Get a free key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-1.5-flash"

# ── Dashboard Theme ───────────────────────────────────────────────────────────
BRAND_COLOR    = "#1E3A5F"      # Navy blue — primary
ACCENT_COLOR   = "#E8A838"      # Gold — accent
SUCCESS_COLOR  = "#27AE60"      # Green
DANGER_COLOR   = "#E74C3C"      # Red
NEUTRAL_COLOR  = "#95A5A6"      # Grey

PLOTLY_TEMPLATE = "plotly_white"

COLOR_SEQUENCE = [
    "#1E3A5F", "#E8A838", "#27AE60", "#E74C3C",
    "#8E44AD", "#2980B9", "#F39C12", "#1ABC9C",
]

# ── KPI Formatting ────────────────────────────────────────────────────────────
CURRENCY_SYMBOL = "R$"          # Brazilian Real (Olist dataset)

# ── Segment Labels ────────────────────────────────────────────────────────────
# Applied after inspecting actual cluster statistics in the data
SEGMENT_COLORS = {
    "High-Value Customers":    "#E8A838",
    "Frequent Customers":      "#27AE60",
    "Low-Engagement Customers":"#2980B9",
    "At-Risk Customers":       "#E74C3C",
}

# ── Page Config ───────────────────────────────────────────────────────────────
PAGE_TITLE = "Commerce360 AI"
PAGE_ICON  = "🛒"
LAYOUT     = "wide"

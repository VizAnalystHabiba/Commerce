"""
components/metrics.py — Commerce360 AI KPI Card Components
Reusable Streamlit metric cards with consistent styling.
"""

import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config.config import BRAND_COLOR, ACCENT_COLOR, SUCCESS_COLOR, DANGER_COLOR
from utils.helpers import fmt_currency, fmt_number, fmt_pct


def kpi_card(label: str, value: str, delta: str = None,
             delta_color: str = "normal", help_text: str = None):
    """Render a single Streamlit metric card."""
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text,
    )


def executive_kpi_row(kpis: dict):
    """
    Render the 6 executive KPI cards in a single row.
    kpis: dict from calculations.get_executive_kpis()
    """
    cols = st.columns(6)
    with cols[0]:
        kpi_card(
            "Total Revenue",
            fmt_currency(kpis["total_revenue"]),
            help_text="Sum of all item revenues from delivered orders",
        )
    with cols[1]:
        kpi_card(
            "Total Orders",
            fmt_number(kpis["total_orders"]),
            help_text="Unique delivered orders",
        )
    with cols[2]:
        kpi_card(
            "Total Customers",
            fmt_number(kpis["total_customers"]),
            help_text="Unique customers with at least one order",
        )
    with cols[3]:
        kpi_card(
            "Active Sellers",
            fmt_number(kpis["total_sellers"]),
            help_text="Sellers with at least one order",
        )
    with cols[4]:
        kpi_card(
            "Avg Delivery Days",
            f"{kpis['avg_delivery_days']} days",
            help_text="Average actual delivery time for delivered orders",
        )
    with cols[5]:
        delta_col = "normal" if kpis["on_time_pct"] >= 90 else "inverse"
        kpi_card(
            "On-Time Delivery",
            fmt_pct(kpis["on_time_pct"]),
            delta_color=delta_col,
            help_text="% of orders delivered on or before estimated date",
        )


def section_header(title: str, subtitle: str = ""):
    """Render a styled section header."""
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()


def info_callout(message: str, icon: str = "ℹ️"):
    """Render an info callout box."""
    st.info(f"{icon} {message}")


def success_callout(message: str):
    """Render a success callout."""
    st.success(message)


def warning_callout(message: str):
    """Render a warning callout."""
    st.warning(message)

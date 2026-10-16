"""
components/filters.py — Commerce360 AI Sidebar Filters
Reusable filter components that return filtered DataFrames.
"""

import streamlit as st
import pandas as pd
from datetime import datetime


def date_range_filter(df: pd.DataFrame,
                      date_col: str = "order_purchase_timestamp",
                      label: str = "Date Range") -> pd.DataFrame:
    """Filter DataFrame by date range. Returns filtered df."""
    if date_col not in df.columns:
        return df

    min_date = df[date_col].min()
    max_date = df[date_col].max()

    if pd.isna(min_date) or pd.isna(max_date):
        return df

    min_d = min_date.date()
    max_d = max_date.date()

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input(f"{label} — Start", value=min_d,
                              min_value=min_d, max_value=max_d, key=f"{label}_start")
    with col2:
        end = st.date_input(f"{label} — End", value=max_d,
                            min_value=min_d, max_value=max_d, key=f"{label}_end")

    mask = (df[date_col].dt.date >= start) & (df[date_col].dt.date <= end)
    return df[mask]


def category_filter(df: pd.DataFrame,
                    col: str = "product_category_name_english",
                    label: str = "Category") -> pd.DataFrame:
    """Multi-select filter for product category."""
    if col not in df.columns:
        return df

    options = sorted(df[col].dropna().unique().tolist())
    selected = st.multiselect(label, options=options, default=[],
                              key=f"cat_filter_{col}")
    if selected:
        return df[df[col].isin(selected)]
    return df


def state_filter(df: pd.DataFrame,
                 col: str = "customer_state",
                 label: str = "State") -> pd.DataFrame:
    """Multi-select filter for state."""
    if col not in df.columns:
        return df

    options = sorted(df[col].dropna().unique().tolist())
    selected = st.multiselect(label, options=options, default=[],
                              key=f"state_filter_{col}")
    if selected:
        return df[df[col].isin(selected)]
    return df


def delivery_status_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Radio filter for delivery status."""
    if "delivery_status" not in df.columns:
        return df

    options = ["All", "On Time", "Delayed"]
    choice = st.radio("Delivery Status", options=options, horizontal=True,
                      key="del_status_filter")
    if choice != "All":
        return df[df["delivery_status"] == choice]
    return df


def order_status_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Select filter for order status."""
    if "order_status" not in df.columns:
        return df

    options = ["All"] + sorted(df["order_status"].dropna().unique().tolist())
    choice = st.selectbox("Order Status", options=options, key="order_status_filter")
    if choice != "All":
        return df[df["order_status"] == choice]
    return df


def segment_filter(df: pd.DataFrame, col: str = "segment_name") -> pd.DataFrame:
    """Multi-select filter for customer segments."""
    if col not in df.columns:
        return df

    options = sorted(df[col].dropna().unique().tolist())
    selected = st.multiselect("Customer Segment", options=options, default=[],
                              key="segment_filter")
    if selected:
        return df[df[col].isin(selected)]
    return df


def top_n_slider(label: str = "Show Top N", min_val: int = 5,
                 max_val: int = 30, default: int = 15, key: str = "top_n") -> int:
    """Slider to select top N items."""
    return st.slider(label, min_value=min_val, max_value=max_val,
                     value=default, key=key)

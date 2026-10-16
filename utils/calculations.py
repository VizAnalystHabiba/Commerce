"""
utils/calculations.py — Commerce360 AI Analytics Calculations
All KPI computations and business metrics derived from actual data.
Column names verified against real CSV schemas.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Executive KPIs ────────────────────────────────────────────────────────────

def get_executive_kpis(sales: pd.DataFrame, delivery: pd.DataFrame,
                       customer_features: pd.DataFrame,
                       seller_features: pd.DataFrame) -> dict:
    """
    Compute top-level business KPIs for the executive overview page.
    Uses item_revenue from sales_features (already engineered).
    Delivery deduplication: one row per order_id for delivery metrics.
    """
    # Total revenue from item-level data (item_revenue = price of each item)
    total_revenue = sales["item_revenue"].sum()

    # Total delivered orders — deduplicate to avoid counting multi-item orders multiple times
    delivered_orders = (
        delivery[delivery["order_status"] == "delivered"]
        .drop_duplicates(subset="order_id")
    )
    total_orders = delivered_orders["order_id"].nunique()

    # Unique customers (by customer_id)
    total_customers = customer_features["customer_id"].nunique()

    # Active sellers
    total_sellers = seller_features["seller_id"].nunique()

    # Delivery metrics — delivered orders only, one row per order
    avg_delivery_days = delivered_orders["delivery_days"].mean()
    on_time_pct = (
        (delivered_orders["on_time_flag"] == 1).sum() / len(delivered_orders) * 100
        if len(delivered_orders) > 0 else 0
    )

    return {
        "total_revenue":    round(total_revenue, 2),
        "total_orders":     int(total_orders),
        "total_customers":  int(total_customers),
        "total_sellers":    int(total_sellers),
        "avg_delivery_days": round(avg_delivery_days, 1),
        "on_time_pct":      round(on_time_pct, 1),
    }


# ── Monthly Trends ─────────────────────────────────────────────────────────────

def get_monthly_revenue(sales: pd.DataFrame) -> pd.DataFrame:
    """Monthly revenue trend using order_year and order_month from sales_features."""
    monthly = (
        sales.groupby(["order_year", "order_month", "order_month_name"])
        ["item_revenue"].sum()
        .reset_index()
        .sort_values(["order_year", "order_month"])
    )
    monthly["period"] = (
        monthly["order_year"].astype(str) + "-"
        + monthly["order_month"].astype(str).str.zfill(2)
    )
    return monthly


def get_monthly_orders(sales: pd.DataFrame) -> pd.DataFrame:
    """Monthly distinct order count (deduplicate by order_id per month)."""
    monthly = (
        sales.drop_duplicates(subset=["order_id", "order_year", "order_month"])
        .groupby(["order_year", "order_month"])["order_id"].nunique()
        .reset_index()
        .rename(columns={"order_id": "order_count"})
        .sort_values(["order_year", "order_month"])
    )
    monthly["period"] = (
        monthly["order_year"].astype(str) + "-"
        + monthly["order_month"].astype(str).str.zfill(2)
    )
    return monthly


# ── Sales & Product Analytics ─────────────────────────────────────────────────

def get_revenue_by_category(sales: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Top N categories by total revenue."""
    return (
        sales.groupby("product_category_name_english")["item_revenue"].sum()
        .reset_index()
        .rename(columns={"product_category_name_english": "category",
                         "item_revenue": "revenue"})
        .sort_values("revenue", ascending=False)
        .head(top_n)
    )


def get_top_products(product_features: pd.DataFrame,
                     product_table: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Top N products by total revenue, with category label."""
    merged = product_features.merge(
        product_table[["product_id", "product_category_name_english"]],
        on="product_id", how="left"
    )
    return (
        merged.sort_values("total_revenue", ascending=False)
        .head(top_n)
        [["product_id", "product_category_name_english", "sales_count",
          "total_revenue", "average_price", "average_freight"]]
        .rename(columns={
            "product_category_name_english": "category",
            "sales_count": "units_sold",
        })
    )


def get_sales_by_state(sales: pd.DataFrame) -> pd.DataFrame:
    """Revenue by seller state."""
    return (
        sales.groupby("seller_state")["item_revenue"].sum()
        .reset_index()
        .rename(columns={"item_revenue": "revenue"})
        .sort_values("revenue", ascending=False)
    )


# ── Customer Analytics ────────────────────────────────────────────────────────

def label_segments(customer_segments: pd.DataFrame,
                   customer_features: pd.DataFrame) -> pd.DataFrame:
    """
    Assign human-readable segment names based on actual cluster statistics.
    Clusters are sorted by total_spending descending so cluster 0 = highest spenders.
    Returns merged dataframe with 'segment_name' column.
    """
    seg = customer_segments.copy()

    # Compute median spending per cluster to rank them
    cluster_stats = (
        seg.groupby("segment")["total_spending"]
        .median()
        .reset_index()
        .sort_values("total_spending", ascending=False)
        .reset_index(drop=True)
    )

    n_clusters = len(cluster_stats)
    label_map = {}

    # Assign labels based on ranked spending position
    labels = ["High-Value Customers", "Frequent Customers",
              "Low-Engagement Customers", "At-Risk Customers"]

    for i, row in cluster_stats.iterrows():
        label_map[int(row["segment"])] = labels[i] if i < len(labels) else f"Segment {i}"

    seg["segment_name"] = seg["segment"].map(label_map)
    return seg


def get_segment_summary(customer_segments: pd.DataFrame,
                        customer_features: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics per customer segment."""
    seg = label_segments(customer_segments, customer_features)
    return (
        seg.groupby("segment_name")
        .agg(
            customer_count=("customer_id", "count"),
            avg_spending=("total_spending", "mean"),
            avg_orders=("total_orders", "mean"),
            avg_order_value=("average_order_value", "mean"),
            avg_recency=("customer_recency_days", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("avg_spending", ascending=False)
    )


# ── Seller Analytics ──────────────────────────────────────────────────────────

def get_top_sellers(seller_features: pd.DataFrame,
                    sellers_cleaned: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Top N sellers by revenue with state info."""
    merged = seller_features.merge(
        sellers_cleaned[["seller_id", "seller_state", "seller_city"]],
        on="seller_id", how="left"
    )
    return (
        merged.sort_values("seller_revenue", ascending=False)
        .head(top_n)
        [["seller_id", "seller_state", "seller_city",
          "seller_orders", "seller_revenue", "seller_items",
          "average_item_price", "average_freight"]]
    )


def get_revenue_by_seller_state(seller_features: pd.DataFrame,
                                 sellers_cleaned: pd.DataFrame) -> pd.DataFrame:
    """Total seller revenue by state."""
    merged = seller_features.merge(
        sellers_cleaned[["seller_id", "seller_state"]],
        on="seller_id", how="left"
    )
    return (
        merged.groupby("seller_state")["seller_revenue"].sum()
        .reset_index()
        .rename(columns={"seller_revenue": "revenue"})
        .sort_values("revenue", ascending=False)
    )


# ── Delivery Analytics ────────────────────────────────────────────────────────

def get_delivery_summary(delivery: pd.DataFrame) -> dict:
    """Core delivery KPIs from delivered orders."""
    delivered = (
        delivery[delivery["order_status"] == "delivered"]
        .drop_duplicates(subset="order_id")
    )
    total      = len(delivered)
    on_time    = (delivered["on_time_flag"] == 1).sum()
    delayed    = total - on_time
    avg_days   = delivered["delivery_days"].mean()
    avg_delay  = delivered.loc[delivered["delivery_delay_days"] > 0,
                               "delivery_delay_days"].mean()
    return {
        "total_delivered": int(total),
        "on_time":         int(on_time),
        "delayed":         int(delayed),
        "on_time_pct":     round(on_time / total * 100, 1) if total > 0 else 0,
        "avg_delivery_days": round(avg_days, 1),
        "avg_delay_days":  round(avg_delay, 1) if not np.isnan(avg_delay) else 0,
    }


def get_delivery_by_state(delivery: pd.DataFrame) -> pd.DataFrame:
    """Average delivery days and on-time % by customer state."""
    delivered = (
        delivery[delivery["order_status"] == "delivered"]
        .drop_duplicates(subset="order_id")
    )
    return (
        delivered.groupby("customer_state")
        .agg(
            avg_delivery_days=("delivery_days", "mean"),
            on_time_pct=("on_time_flag", lambda x: round(x.mean() * 100, 1)),
            order_count=("order_id", "count"),
        )
        .round(2)
        .reset_index()
        .sort_values("avg_delivery_days", ascending=False)
    )


def get_delivery_trend(delivery: pd.DataFrame) -> pd.DataFrame:
    """Monthly delivery performance trend."""
    df = delivery[delivery["order_status"] == "delivered"].copy()
    df["year"]  = df["order_purchase_timestamp"].dt.year
    df["month"] = df["order_purchase_timestamp"].dt.month
    df = df.drop_duplicates(subset="order_id")
    return (
        df.groupby(["year", "month"])
        .agg(
            avg_delivery_days=("delivery_days", "mean"),
            on_time_pct=("on_time_flag", lambda x: round(x.mean() * 100, 1)),
            order_count=("order_id", "count"),
        )
        .round(2)
        .reset_index()
        .sort_values(["year", "month"])
    )


# ── Review Analytics ──────────────────────────────────────────────────────────

def get_review_summary(reviews: pd.DataFrame) -> dict:
    """Core review KPIs."""
    return {
        "avg_score":    round(reviews["review_score"].mean(), 2),
        "total_reviews": len(reviews),
        "pct_positive": round(
            (reviews["review_score"] >= 4).mean() * 100, 1
        ),
        "pct_negative": round(
            (reviews["review_score"] <= 2).mean() * 100, 1
        ),
    }


def get_score_by_delivery_status(reviews: pd.DataFrame) -> pd.DataFrame:
    """Average review score grouped by delivery status."""
    return (
        reviews.groupby("delivery_status")["review_score"]
        .agg(["mean", "count"])
        .round(2)
        .reset_index()
        .rename(columns={"mean": "avg_score", "count": "review_count"})
    )


def get_monthly_review_scores(reviews: pd.DataFrame) -> pd.DataFrame:
    """Monthly average review score trend."""
    df = reviews.copy()
    df["year"]  = df["review_creation_date"].dt.year
    df["month"] = df["review_creation_date"].dt.month
    return (
        df.groupby(["year", "month"])["review_score"].mean()
        .round(2)
        .reset_index()
        .sort_values(["year", "month"])
    )


# ── Payment Analytics ─────────────────────────────────────────────────────────

def get_payment_summary(payments: pd.DataFrame) -> dict:
    """Core payment KPIs."""
    total_rev  = payments["payment_value"].sum()
    avg_val    = payments["payment_value"].mean()
    avg_inst   = payments["payment_installments"].mean()
    top_method = payments["payment_type"].value_counts().index[0]
    return {
        "total_payment_value": round(total_rev, 2),
        "avg_payment_value":   round(avg_val, 2),
        "avg_installments":    round(avg_inst, 1),
        "top_payment_method":  top_method,
    }


def get_payment_by_type(payments: pd.DataFrame) -> pd.DataFrame:
    """Payment value and count by payment type."""
    return (
        payments.groupby("payment_type")
        .agg(
            total_value=("payment_value", "sum"),
            transaction_count=("payment_type", "count"),
            avg_value=("payment_value", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("total_value", ascending=False)
    )


def get_installment_distribution(payments: pd.DataFrame) -> pd.DataFrame:
    """Distribution of installment counts."""
    return (
        payments.groupby("payment_installments")["payment_value"]
        .agg(["count", "mean"])
        .round(2)
        .reset_index()
        .rename(columns={"count": "transactions", "mean": "avg_value"})
        .sort_values("payment_installments")
    )

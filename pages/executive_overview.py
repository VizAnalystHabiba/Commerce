"""
pages/executive_overview.py — Commerce360 AI Executive Overview Page
Page 1: High-level business health dashboard.
"""

import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_loader import (
    load_sales_features, load_customer_features,
    load_seller_features, load_delivery_features
)
from utils.calculations import (
    get_executive_kpis, get_monthly_revenue, get_monthly_orders,
    get_revenue_by_category, get_delivery_summary
)
from components.metrics import executive_kpi_row, section_header
from components.charts import (
    revenue_trend_chart, orders_trend_chart,
    revenue_by_category_chart, delivery_status_pie, delivery_days_histogram
)
from components.filters import date_range_filter
from utils.helpers import df_to_csv_bytes


def show():
    st.title("Executive Overview")
    st.caption("Top-level business performance at a glance.")

    # Load data
    with st.spinner("Loading data..."):
        sales    = load_sales_features()
        cust_ft  = load_customer_features()
        sell_ft  = load_seller_features()
        delivery = load_delivery_features()

    # ── Sidebar Filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Filters")
        sales    = date_range_filter(sales,    label="Sales Date Range")
        delivery = date_range_filter(delivery, label="Delivery Date Range")

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    section_header("Key Performance Indicators",
                   "Aggregated business metrics for the selected period")
    kpis = get_executive_kpis(sales, delivery, cust_ft, sell_ft)
    executive_kpi_row(kpis)

    # ── Revenue & Orders Trend ────────────────────────────────────────────────
    st.divider()
    section_header("Revenue & Orders Trend", "Monthly business volume")

    col1, col2 = st.columns(2)

    with col1:
        monthly_rev = get_monthly_revenue(sales)
        st.plotly_chart(revenue_trend_chart(monthly_rev),
                        use_container_width=True)

    with col2:
        monthly_ord = get_monthly_orders(sales)
        st.plotly_chart(orders_trend_chart(monthly_ord),
                        use_container_width=True)

    # ── Category & Delivery ───────────────────────────────────────────────────
    st.divider()
    section_header("Category Performance & Delivery Health")

    col3, col4 = st.columns(2)

    with col3:
        top_cats = get_revenue_by_category(sales, top_n=10)
        st.plotly_chart(revenue_by_category_chart(top_cats),
                        use_container_width=True)

    with col4:
        del_summary = get_delivery_summary(delivery)
        st.plotly_chart(delivery_status_pie(del_summary),
                        use_container_width=True)

        # Delivery KPI cards under the pie
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Delivered", f"{del_summary['total_delivered']:,}")
        c2.metric("Avg Delay Days",  f"{del_summary['avg_delay_days']} days")
        c3.metric("Delayed Orders",  f"{del_summary['delayed']:,}")

    # ── Delivery Time Distribution ────────────────────────────────────────────
    st.divider()
    section_header("Delivery Time Distribution")
    st.plotly_chart(delivery_days_histogram(delivery), use_container_width=True)

    # ── Business Insight Callout ──────────────────────────────────────────────
    st.divider()
    st.info(
        f"**Business Snapshot** | Revenue: R${kpis['total_revenue']:,.2f} | "
        f"Orders: {kpis['total_orders']:,} | On-Time Delivery: {kpis['on_time_pct']}% | "
        f"Avg Delivery: {kpis['avg_delivery_days']} days"
    )

    # ── Data Download ─────────────────────────────────────────────────────────
    with st.expander("Download Summary Data"):
        monthly_rev["revenue"] = monthly_rev["item_revenue"]
        st.download_button(
            label="Download Monthly Revenue CSV",
            data=df_to_csv_bytes(monthly_rev[["period", "revenue"]]),
            file_name="monthly_revenue.csv",
            mime="text/csv",
        )

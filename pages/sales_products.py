"""
pages/sales_products.py — Commerce360 AI Sales & Product Analytics Page
Page 2: Revenue, category, product, and sales trend analysis.
"""

import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_sales_features, load_product_features, load_product_table
from utils.calculations import (
    get_revenue_by_category, get_top_products,
    get_monthly_revenue, get_monthly_orders, get_sales_by_state
)
from components.metrics import section_header
from components.charts import (
    revenue_by_category_chart, top_products_chart,
    revenue_trend_chart, orders_trend_chart
)
from components.filters import date_range_filter, category_filter, top_n_slider
from utils.helpers import df_to_csv_bytes, fmt_currency
import plotly.express as px


def show():
    st.title("Sales & Product Analytics")
    st.caption("Revenue trends, category performance, and top product analysis.")

    with st.spinner("Loading data..."):
        sales       = load_sales_features()
        prod_feat   = load_product_features()
        prod_table  = load_product_table()

    # ── Sidebar Filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Filters")
        sales = date_range_filter(sales, label="Date Range")
        sales = category_filter(sales, col="product_category_name_english",
                                label="Product Category")
        top_n = top_n_slider("Show Top N Products", default=15)

    # ── Revenue KPIs ──────────────────────────────────────────────────────────
    section_header("Sales Overview")
    total_revenue = sales["item_revenue"].sum()
    total_orders  = sales["order_id"].nunique()
    avg_order_val = total_revenue / total_orders if total_orders > 0 else 0
    total_items   = len(sales)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue",      fmt_currency(total_revenue))
    c2.metric("Total Orders",       f"{total_orders:,}")
    c3.metric("Avg Order Value",    fmt_currency(avg_order_val))
    c4.metric("Total Items Sold",   f"{total_items:,}")

    # ── Trends ────────────────────────────────────────────────────────────────
    st.divider()
    section_header("Sales Trends", "Monthly revenue and order volume")

    col1, col2 = st.columns(2)
    with col1:
        monthly_rev = get_monthly_revenue(sales)
        st.plotly_chart(revenue_trend_chart(monthly_rev), use_container_width=True)
    with col2:
        monthly_ord = get_monthly_orders(sales)
        st.plotly_chart(orders_trend_chart(monthly_ord), use_container_width=True)

    # ── Category Performance ──────────────────────────────────────────────────
    st.divider()
    section_header("Category Performance", "Revenue contribution by product category")

    top_cats = get_revenue_by_category(sales, top_n=top_n)

    col3, col4 = st.columns([2, 1])
    with col3:
        st.plotly_chart(revenue_by_category_chart(top_cats), use_container_width=True)
    with col4:
        st.markdown("**Top Categories**")
        display_cats = top_cats.copy()
        display_cats["revenue"] = display_cats["revenue"].apply(fmt_currency)
        st.dataframe(display_cats[["category", "revenue"]].rename(
            columns={"category": "Category", "revenue": "Revenue"}
        ), use_container_width=True, hide_index=True)

    # ── Top Products ──────────────────────────────────────────────────────────
    st.divider()
    section_header("Top Products", "Best performing products by total revenue")

    top_prods = get_top_products(prod_feat, prod_table, top_n=top_n)
    col5, col6 = st.columns([2, 1])
    with col5:
        st.plotly_chart(top_products_chart(top_prods), use_container_width=True)
    with col6:
        display = top_prods.copy()
        display["total_revenue"] = display["total_revenue"].apply(fmt_currency)
        display["average_price"] = display["average_price"].apply(fmt_currency)
        st.dataframe(
            display[["product_id", "category", "units_sold", "total_revenue"]].rename(
                columns={"product_id": "Product ID", "category": "Category",
                         "units_sold": "Units", "total_revenue": "Revenue"}
            ),
            use_container_width=True, hide_index=True
        )

    # ── Revenue by State ──────────────────────────────────────────────────────
    st.divider()
    section_header("Revenue by Seller State")
    state_rev = get_sales_by_state(sales)
    fig_state = px.bar(
        state_rev.head(15),
        x="seller_state", y="revenue",
        title="Revenue by Seller State (Top 15)",
        labels={"seller_state": "State", "revenue": "Revenue (R$)"},
        template="plotly_white",
        color="revenue",
        color_continuous_scale=[[0, "#BDC3C7"], [1, "#1E3A5F"]],
    )
    fig_state.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_state, use_container_width=True)

    # ── Price Distribution ─────────────────────────────────────────────────────
    st.divider()
    section_header("Product Price Distribution")
    fig_price = px.histogram(
        sales[sales["price"] < sales["price"].quantile(0.99)],
        x="price", nbins=50,
        title="Product Price Distribution (Excluding Top 1% Outliers)",
        labels={"price": "Price (R$)"},
        template="plotly_white",
        color_discrete_sequence=["#1E3A5F"],
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # ── Downloads ─────────────────────────────────────────────────────────────
    with st.expander("Download Data"):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "Download Category Revenue CSV",
                data=df_to_csv_bytes(top_cats),
                file_name="category_revenue.csv",
                mime="text/csv",
            )
        with col_d2:
            st.download_button(
                "Download Top Products CSV",
                data=df_to_csv_bytes(top_prods),
                file_name="top_products.csv",
                mime="text/csv",
            )

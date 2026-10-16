"""
pages/seller_analytics.py — Commerce360 AI Seller Analytics Page
Page 4: Seller performance, revenue, and geographic distribution.
"""

import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_seller_features, load_sellers_cleaned
from utils.calculations import get_top_sellers, get_revenue_by_seller_state
from components.metrics import section_header
from components.charts import (
    top_sellers_revenue_chart, seller_revenue_distribution_chart
)
from components.filters import state_filter, top_n_slider
from utils.helpers import df_to_csv_bytes, fmt_currency
import plotly.express as px


def show():
    st.title("Seller Analytics")
    st.caption("Seller performance, revenue contribution, and geographic distribution.")

    with st.spinner("Loading data..."):
        sell_feat  = load_seller_features()
        sell_clean = load_sellers_cleaned()

    merged = sell_feat.merge(
        sell_clean[["seller_id", "seller_state", "seller_city"]],
        on="seller_id", how="left"
    )

    # ── Sidebar Filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Filters")
        merged = state_filter(merged, col="seller_state", label="Seller State")
        top_n  = top_n_slider("Show Top N Sellers", default=15)

    # ── Seller KPIs ───────────────────────────────────────────────────────────
    section_header("Seller Overview")

    total_sellers  = merged["seller_id"].nunique()
    total_revenue  = merged["seller_revenue"].sum()
    avg_revenue    = merged["seller_revenue"].mean()
    avg_item_price = merged["average_item_price"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sellers",      f"{total_sellers:,}")
    c2.metric("Total Revenue",      fmt_currency(total_revenue))
    c3.metric("Avg Revenue / Seller", fmt_currency(avg_revenue))
    c4.metric("Avg Item Price",     fmt_currency(avg_item_price))

    # ── Top Sellers ───────────────────────────────────────────────────────────
    st.divider()
    section_header("Top Sellers by Revenue")

    top_sellers = get_top_sellers(sell_feat, sell_clean, top_n=top_n)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(top_sellers_revenue_chart(top_sellers), use_container_width=True)
    with col2:
        display = top_sellers.copy()
        display["seller_revenue"]    = display["seller_revenue"].apply(fmt_currency)
        display["average_item_price"] = display["average_item_price"].apply(fmt_currency)
        st.dataframe(
            display[["seller_id", "seller_state", "seller_orders",
                      "seller_revenue"]].rename(
                columns={
                    "seller_id":      "Seller ID",
                    "seller_state":   "State",
                    "seller_orders":  "Orders",
                    "seller_revenue": "Revenue",
                }
            ).assign(**{"Seller ID": lambda x: x["Seller ID"].str[:10] + "..."}),
            use_container_width=True,
            hide_index=True,
        )

    # ── Top Sellers by Orders ─────────────────────────────────────────────────
    st.divider()
    section_header("Top Sellers by Order Count")

    top_by_orders = merged.sort_values("seller_orders", ascending=False).head(top_n).copy()
    top_by_orders["short_id"] = top_by_orders["seller_id"].str[:8] + "..."

    fig_orders = px.bar(
        top_by_orders.sort_values("seller_orders"),
        x="seller_orders", y="short_id", orientation="h",
        title=f"Top {top_n} Sellers by Order Count",
        labels={"seller_orders": "Orders", "short_id": "Seller"},
        color="seller_state",
        template="plotly_white",
        color_discrete_sequence=["#1E3A5F", "#E8A838", "#27AE60", "#E74C3C",
                                  "#8E44AD", "#2980B9", "#F39C12", "#1ABC9C"],
    )
    st.plotly_chart(fig_orders, use_container_width=True)

    # ── Revenue by State ──────────────────────────────────────────────────────
    st.divider()
    section_header("Revenue by Seller State")

    state_rev = get_revenue_by_seller_state(sell_feat, sell_clean)

    col3, col4 = st.columns(2)
    with col3:
        fig_state = px.bar(
            state_rev.head(15).sort_values("revenue", ascending=False),
            x="seller_state", y="revenue",
            title="Revenue by Seller State (Top 15)",
            labels={"seller_state": "State", "revenue": "Revenue (R$)"},
            template="plotly_white",
            color_discrete_sequence=["#1E3A5F"],
        )
        st.plotly_chart(fig_state, use_container_width=True)

    with col4:
        fig_pie = px.pie(
            state_rev.head(8),
            names="seller_state", values="revenue",
            title="Revenue Share by Top 8 Seller States",
            template="plotly_white",
            hole=0.4,
            color_discrete_sequence=["#1E3A5F", "#E8A838", "#27AE60", "#E74C3C",
                                      "#8E44AD", "#2980B9", "#F39C12", "#1ABC9C"],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Revenue Distribution ──────────────────────────────────────────────────
    st.divider()
    section_header("Seller Revenue Distribution",
                   "How revenue is distributed across all sellers")

    st.plotly_chart(seller_revenue_distribution_chart(sell_feat),
                    use_container_width=True)

    # ── Freight Analysis ──────────────────────────────────────────────────────
    st.divider()
    section_header("Freight vs Item Price by Seller State")

    fig_scatter = px.scatter(
        merged,
        x="average_item_price", y="average_freight",
        color="seller_state",
        title="Average Item Price vs Average Freight by Seller",
        labels={
            "average_item_price": "Avg Item Price (R$)",
            "average_freight":    "Avg Freight (R$)",
            "seller_state":       "State",
        },
        template="plotly_white",
        opacity=0.7,
        hover_data=["seller_id", "seller_orders"],
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Downloads ─────────────────────────────────────────────────────────────
    with st.expander("Download Data"):
        st.download_button(
            "Download Seller Performance CSV",
            data=df_to_csv_bytes(merged),
            file_name="seller_performance.csv",
            mime="text/csv",
        )

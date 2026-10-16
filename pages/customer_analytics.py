"""
pages/customer_analytics.py — Commerce360 AI Customer Analytics Page
Page 3: Customer segmentation, RFM, spending, and recency analysis.
"""

import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_customer_features, load_customer_segments
from utils.calculations import label_segments, get_segment_summary
from components.metrics import section_header
from components.charts import (
    segment_size_chart, segment_spending_chart,
    segment_scatter_chart, rfm_bar_chart
)
from components.filters import segment_filter
from utils.helpers import df_to_csv_bytes, fmt_currency
import plotly.express as px


def show():
    st.title("Customer Analytics")
    st.caption("Customer segmentation based on K-Means clustering and RFM behavior analysis.")

    with st.spinner("Loading data..."):
        cust_feat = load_customer_features()
        cust_seg  = load_customer_segments()

    # Label segments from actual cluster statistics
    segs_labeled = label_segments(cust_seg, cust_feat)

    # ── Sidebar Filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Filters")
        segs_labeled = segment_filter(segs_labeled)

    # ── Segment Overview KPIs ─────────────────────────────────────────────────
    section_header("Customer Segmentation Overview",
                   "K-Means clustering based on spending, orders, and recency")

    summary = get_segment_summary(cust_seg, cust_feat)

    cols = st.columns(len(summary))
    for i, row in summary.iterrows():
        with cols[i % len(cols)]:
            st.metric(
                label=row["segment_name"],
                value=f"{row['customer_count']:,} customers",
                help=f"Avg spending: R${row['avg_spending']:,.2f} | Avg orders: {row['avg_orders']:.1f}"
            )

    # ── Segment Distribution ──────────────────────────────────────────────────
    st.divider()
    section_header("Segment Distribution")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(segment_size_chart(summary), use_container_width=True)
    with col2:
        st.plotly_chart(segment_spending_chart(summary), use_container_width=True)

    # ── Segment Map (Scatter) ─────────────────────────────────────────────────
    st.divider()
    section_header("Customer Segmentation Map",
                   "Recency vs Spending — bubble size = number of orders")

    st.plotly_chart(segment_scatter_chart(segs_labeled), use_container_width=True)

    # ── RFM Comparison ────────────────────────────────────────────────────────
    st.divider()
    section_header("RFM Behavior by Segment",
                   "Orders and average order value comparison across segments")

    st.plotly_chart(rfm_bar_chart(summary), use_container_width=True)

    # ── Segment Detail Table ──────────────────────────────────────────────────
    st.divider()
    section_header("Segment Detailed Statistics")

    display = summary.copy()
    display["avg_spending"]     = display["avg_spending"].apply(fmt_currency)
    display["avg_order_value"]  = display["avg_order_value"].apply(fmt_currency)
    display["avg_recency"]      = display["avg_recency"].apply(lambda x: f"{x:.0f} days")

    st.dataframe(
        display.rename(columns={
            "segment_name":   "Segment",
            "customer_count": "Customers",
            "avg_spending":   "Avg Spending",
            "avg_orders":     "Avg Orders",
            "avg_order_value":"Avg Order Value",
            "avg_recency":    "Avg Recency",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Spending Distribution by Segment ─────────────────────────────────────
    st.divider()
    section_header("Spending Distribution by Segment")

    # Merge segment name into features
    feat_with_seg = cust_feat.merge(
        segs_labeled[["customer_id", "segment_name"]], on="customer_id", how="left"
    )
    fig_box = px.box(
        feat_with_seg[feat_with_seg["total_spending"] < feat_with_seg["total_spending"].quantile(0.95)],
        x="segment_name", y="total_spending",
        title="Spending Distribution by Segment (Excluding Top 5% Outliers)",
        labels={"segment_name": "Segment", "total_spending": "Total Spending (R$)"},
        template="plotly_white",
        color="segment_name",
        color_discrete_sequence=["#1E3A5F", "#E8A838", "#27AE60", "#E74C3C"],
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Recency Distribution ──────────────────────────────────────────────────
    st.divider()
    section_header("Recency Distribution",
                   "How recently did customers make their last purchase?")

    fig_rec = px.histogram(
        cust_feat, x="customer_recency_days", nbins=40,
        title="Customer Recency Distribution (Days Since Last Purchase)",
        labels={"customer_recency_days": "Recency (days)"},
        template="plotly_white",
        color_discrete_sequence=["#1E3A5F"],
    )
    st.plotly_chart(fig_rec, use_container_width=True)

    # ── Business Interpretation ───────────────────────────────────────────────
    st.divider()
    st.info(
        "**Segmentation Logic**: Customers are grouped by K-Means (k=4) on spending, "
        "orders, average order value, items, and recency. "
        "Segment labels are assigned based on actual median spending per cluster — "
        "not arbitrary labels. High-Value = highest median spend."
    )

    # ── Downloads ─────────────────────────────────────────────────────────────
    with st.expander("Download Data"):
        st.download_button(
            "Download Customer Segments CSV",
            data=df_to_csv_bytes(segs_labeled),
            file_name="customer_segments.csv",
            mime="text/csv",
        )

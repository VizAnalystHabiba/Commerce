"""
pages/delivery_intelligence.py — Commerce360 AI Delivery Intelligence Page
Page 5: Delivery performance analytics and ML delay prediction results.
"""

import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_delivery_features, load_ml_results, load_sales_features
from utils.calculations import (
    get_delivery_summary, get_delivery_by_state,
    get_delivery_trend, get_revenue_by_category
)
from components.metrics import section_header
from components.charts import (
    delivery_status_pie, delivery_days_histogram,
    delivery_delay_distribution, delivery_by_state_chart,
    delivery_trend_chart, ml_metrics_bar, feature_importance_chart
)
from components.filters import date_range_filter, delivery_status_filter
from utils.helpers import df_to_csv_bytes
import plotly.express as px


# Feature importances from the trained Random Forest (verified output from delivery_model.py)
RF_FEATURE_IMPORTANCES = {
    "total_freight":          0.137978,
    "avg_volume_cm3":         0.135172,
    "freight_ratio":          0.133983,
    "estimated_delivery_days": 0.129400,
    "purchase_month":         0.124781,
    "total_price":            0.123273,
    "avg_weight_g":           0.121140,
    "purchase_dayofweek":     0.067178,
    "same_state":             0.018007,
    "num_items":              0.009088,
}


def show():
    st.title("Delivery Intelligence")
    st.caption(
        "Delivery performance analytics (descriptive) and ML-based delay prediction (predictive)."
    )

    with st.spinner("Loading data..."):
        delivery   = load_delivery_features()
        ml_results = load_ml_results()
        sales      = load_sales_features()

    # ── Sidebar Filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Filters")
        delivery = date_range_filter(delivery, label="Date Range")

    # ── Delivery KPIs ─────────────────────────────────────────────────────────
    section_header("Delivery Performance Overview")

    del_summary = get_delivery_summary(delivery)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Delivered",   f"{del_summary['total_delivered']:,}")
    c2.metric("On-Time Orders",    f"{del_summary['on_time']:,}")
    c3.metric("Delayed Orders",    f"{del_summary['delayed']:,}")
    c4.metric("On-Time Rate",      f"{del_summary['on_time_pct']}%")
    c5.metric("Avg Delivery Days", f"{del_summary['avg_delivery_days']} days")

    # ── Performance Overview ──────────────────────────────────────────────────
    st.divider()
    section_header("On-Time vs Delayed")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(delivery_status_pie(del_summary), use_container_width=True)
    with col2:
        st.plotly_chart(delivery_days_histogram(delivery), use_container_width=True)

    # ── Delay Distribution ────────────────────────────────────────────────────
    st.divider()
    section_header("Delay Analysis")

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(delivery_delay_distribution(delivery), use_container_width=True)

    with col4:
        # Delivery days by delivery status box plot
        delivered_df = delivery[delivery["order_status"] == "delivered"].copy()
        delivered_df = delivered_df.drop_duplicates(subset="order_id")
        fig_box = px.box(
            delivered_df, x="delivery_status", y="delivery_days",
            title="Delivery Days by Status",
            labels={"delivery_status": "Status", "delivery_days": "Delivery Days"},
            template="plotly_white",
            color="delivery_status",
            color_discrete_map={"On Time": "#27AE60", "Delayed": "#E74C3C"},
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # ── State-Level Analysis ──────────────────────────────────────────────────
    st.divider()
    section_header("Delivery Performance by Customer State",
                   "Color = On-Time %, Height = Avg Delivery Days")

    state_del = get_delivery_by_state(delivery)
    st.plotly_chart(delivery_by_state_chart(state_del), use_container_width=True)

    # ── Delivery Trend ────────────────────────────────────────────────────────
    st.divider()
    section_header("Delivery Performance Over Time")

    trend_df = get_delivery_trend(delivery)
    st.plotly_chart(delivery_trend_chart(trend_df), use_container_width=True)

    # ── Delivery by Category ──────────────────────────────────────────────────
    st.divider()
    section_header("Delivery Performance by Product Category")

    # Average delivery days by category — join delivery with sales
    cat_del = (
        sales[sales["order_status"] == "delivered"]
        .drop_duplicates(subset="order_id")
        .groupby("product_category_name_english")["delivery_days"]
        .agg(["mean", "count"])
        .round(1)
        .reset_index()
        .rename(columns={
            "product_category_name_english": "category",
            "mean": "avg_delivery_days",
            "count": "order_count",
        })
        .sort_values("avg_delivery_days", ascending=False)
        .head(20)
    )

    fig_cat_del = px.bar(
        cat_del,
        x="avg_delivery_days", y="category", orientation="h",
        title="Avg Delivery Days by Product Category (Top 20 Slowest)",
        labels={"avg_delivery_days": "Avg Delivery Days", "category": "Category"},
        template="plotly_white",
        color="avg_delivery_days",
        color_continuous_scale=[[0, "#27AE60"], [0.5, "#E8A838"], [1, "#E74C3C"]],
    )
    fig_cat_del.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_cat_del, use_container_width=True)

    # ── ML Prediction Section ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Machine Learning: Pre-Delivery Delay Prediction")
    st.info(
        "**What this model does**: At the time an order is placed, predict whether it "
        "will be delayed — using ONLY information available before delivery "
        "(price, freight, items, estimated delivery window, purchase timing, location). "
        "No post-delivery data is used."
    )

    # Model comparison
    section_header("Model Performance Comparison")
    st.plotly_chart(ml_metrics_bar(ml_results), use_container_width=True)

    # Metrics table
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**Detailed Metrics**")
        display_ml = ml_results.copy()
        display_ml.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        st.dataframe(display_ml, use_container_width=True, hide_index=True)

    with col6:
        st.markdown("**Feature Importances (Random Forest)**")
        st.plotly_chart(
            feature_importance_chart(RF_FEATURE_IMPORTANCES),
            use_container_width=True
        )

    # Interpretation
    st.markdown("""
    **How to interpret these results:**
    - **ROC-AUC = 0.72** (Random Forest) — The model has meaningful discriminative ability.
      A score of 0.5 = random guessing, 1.0 = perfect.
    - **High Recall (Logistic Regression)** — Catches ~65% of delayed orders but generates
      false alarms (conservative for risk management).
    - **Higher Precision (Random Forest)** — When it predicts delay, it is more often correct,
      but misses more actual delays.
    - **Class imbalance**: ~93% on-time, ~7% delayed. Both models use `class_weight='balanced'`.
    - **Business use**: Flag high-risk orders for proactive customer communication.
    """)

    # ── Downloads ─────────────────────────────────────────────────────────────
    with st.expander("Download Data"):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "Download Delivery Summary CSV",
                data=df_to_csv_bytes(state_del),
                file_name="delivery_by_state.csv",
                mime="text/csv",
            )
        with col_d2:
            st.download_button(
                "Download ML Results CSV",
                data=df_to_csv_bytes(ml_results),
                file_name="ml_model_results.csv",
                mime="text/csv",
            )

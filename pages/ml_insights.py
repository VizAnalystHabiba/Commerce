"""
pages/ml_insights.py — Commerce360 AI ML Insights Page
Detailed view of both ML models with business context.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_ml_results, load_customer_segments, load_customer_features
from utils.calculations import label_segments, get_segment_summary
from components.metrics import section_header
from components.charts import ml_metrics_bar, feature_importance_chart, segment_scatter_chart
from utils.helpers import df_to_csv_bytes
import plotly.express as px
import plotly.graph_objects as go


RF_FEATURE_IMPORTANCES = {
    "total_freight":           0.137978,
    "avg_volume_cm3":          0.135172,
    "freight_ratio":           0.133983,
    "estimated_delivery_days": 0.129400,
    "purchase_month":          0.124781,
    "total_price":             0.123273,
    "avg_weight_g":            0.121140,
    "purchase_dayofweek":      0.067178,
    "same_state":              0.018007,
    "num_items":               0.009088,
}


def show():
    st.title("Machine Learning Insights")
    st.caption(
        "Two ML models: (1) Delivery Delay Prediction — pre-delivery, leakage-free. "
        "(2) Customer Segmentation — K-Means behavioral clustering."
    )

    with st.spinner("Loading data..."):
        ml_results  = load_ml_results()
        cust_seg    = load_customer_segments()
        cust_feat   = load_customer_features()

    # ════════════════════════════════════════════════════════════════════════════
    # DELIVERY DELAY PREDICTION
    # ════════════════════════════════════════════════════════════════════════════
    section_header(
        "Model 1 — Delivery Delay Prediction",
        "Predicts whether an order will be delayed using ONLY pre-delivery information"
    )

    st.markdown("""
    **Problem**: At the moment an order is placed, can we predict if it will arrive late?

    **Approach**: Binary classification — *Delayed (1)* vs *On Time (0)*

    **Leakage Prevention**: The following post-delivery columns are strictly excluded:
    `delivery_days`, `delivery_delay_days`, `delivery_status`, `order_delivered_customer_date`,
    `order_delivered_carrier_date`

    **Class Imbalance**: ~93% on-time vs ~7% delayed — handled with `class_weight='balanced'`
    """)

    # Model performance
    st.divider()
    st.plotly_chart(ml_metrics_bar(ml_results), use_container_width=True)

    # Metric explanation table
    col1, col2 = st.columns(2)
    with col1:
        display_ml = ml_results.copy()
        display_ml.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        st.dataframe(display_ml, use_container_width=True, hide_index=True)

    with col2:
        metric_meanings = pd.DataFrame({
            "Metric":    ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            "Meaning":   [
                "% of total predictions correct (misleading with imbalanced data)",
                "Of orders flagged as delayed, how many truly were?",
                "Of truly delayed orders, how many did the model catch?",
                "Harmonic mean of Precision and Recall",
                "Overall discriminative ability (0.5=random, 1.0=perfect)",
            ],
            "Key?": ["Low priority", "Business critical", "Risk management", "Balanced view", "PRIMARY"]
        })
        st.dataframe(metric_meanings, use_container_width=True, hide_index=True)

    # Feature importance
    st.divider()
    section_header("Feature Importances",
                   "Which order characteristics most influence delay prediction?")

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            feature_importance_chart(RF_FEATURE_IMPORTANCES),
            use_container_width=True
        )
    with col4:
        feat_df = pd.DataFrame(
            list(RF_FEATURE_IMPORTANCES.items()),
            columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=False)
        feat_df["Business Meaning"] = [
            "Higher freight = longer/complex shipping route",
            "Larger, heavier products harder to ship",
            "High freight relative to price = difficult delivery",
            "Longer estimated window = longer supply chain",
            "Seasonal demand spikes affect delays",
            "Higher-value orders may have priority handling",
            "Heavier products are harder to ship quickly",
            "Purchases on certain days affect dispatch timing",
            "Same-state orders arrive faster",
            "More items = more complex fulfillment",
        ]
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # Business interpretation
    st.info(
        "**ROC-AUC = 0.72 (Random Forest)** — This is a realistic and academically "
        "defensible baseline using only purchase-time metadata. Without real-time GPS or "
        "carrier tracking data, 0.72 AUC demonstrates genuine signal extraction. "
        "It is far superior to a naive majority-class baseline (AUC = 0.50)."
    )

    # ════════════════════════════════════════════════════════════════════════════
    # CUSTOMER SEGMENTATION
    # ════════════════════════════════════════════════════════════════════════════
    st.divider()
    section_header(
        "Model 2 — Customer Segmentation (K-Means)",
        "Behavioral clustering based on spending, orders, and recency"
    )

    st.markdown("""
    **Algorithm**: K-Means Clustering  
    **k = 4 clusters** (selected based on business interpretability)  
    **Features Used**: `total_spending`, `total_orders`, `average_order_value`,
    `total_items`, `customer_recency_days`  
    **Preprocessing**: StandardScaler applied before clustering  
    **Evaluation**: Silhouette Score = **0.3958** (on 5,000 sample)
    """)

    segs_labeled = label_segments(cust_seg, cust_feat)
    summary      = get_segment_summary(cust_seg, cust_feat)

    # Segment summary table
    st.divider()
    display_seg = summary.copy()
    for col in ["avg_spending", "avg_order_value"]:
        display_seg[col] = display_seg[col].apply(lambda x: f"R${x:,.2f}")
    display_seg["avg_recency"] = display_seg["avg_recency"].apply(lambda x: f"{x:.0f} days")
    st.dataframe(
        display_seg.rename(columns={
            "segment_name":   "Segment",
            "customer_count": "Customers",
            "avg_spending":   "Avg Spending",
            "avg_orders":     "Avg Orders",
            "avg_order_value":"Avg Order Value",
            "avg_recency":    "Avg Recency",
        }),
        use_container_width=True, hide_index=True
    )

    # Scatter plot
    st.plotly_chart(segment_scatter_chart(segs_labeled), use_container_width=True)

    # Silhouette interpretation
    st.info(
        "**Silhouette Score = 0.3958** — A score between 0.25 and 0.50 indicates "
        "meaningful but overlapping clusters, which is expected for customer behavioral data "
        "where segments blend naturally. The clusters produce actionable business segments "
        "even with overlapping boundaries."
    )

    # Segment strategy table
    strategy = pd.DataFrame({
        "Segment": [
            "High-Value Customers",
            "Frequent Customers",
            "Low-Engagement Customers",
            "At-Risk Customers",
        ],
        "Characteristic": [
            "Highest spending, moderate recency",
            "Many orders, good recency",
            "Few orders, moderate spending",
            "Low spending, high recency (inactive)",
        ],
        "Recommended Action": [
            "VIP program, early access, exclusive offers",
            "Loyalty rewards, cross-sell premium products",
            "Re-engagement campaigns, targeted discounts",
            "Win-back email, limited-time offers",
        ],
    })
    st.markdown("**Business Strategy by Segment**")
    st.dataframe(strategy, use_container_width=True, hide_index=True)

    # ── Downloads ─────────────────────────────────────────────────────────────
    with st.expander("Download ML Data"):
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download Model Results CSV",
                data=df_to_csv_bytes(ml_results),
                file_name="ml_model_results.csv",
                mime="text/csv",
            )
        with c2:
            st.download_button(
                "Download Customer Segments CSV",
                data=df_to_csv_bytes(segs_labeled),
                file_name="customer_segments_labeled.csv",
                mime="text/csv",
            )

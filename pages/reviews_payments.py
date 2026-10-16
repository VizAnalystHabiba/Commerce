"""
pages/reviews_payments.py — Commerce360 AI Reviews & Payments Page
Page 6: Customer review analysis and payment behavior intelligence.
"""

import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_loader import load_review_table, load_payment_table
from utils.calculations import (
    get_review_summary, get_score_by_delivery_status,
    get_monthly_review_scores, get_payment_summary,
    get_payment_by_type, get_installment_distribution
)
from components.metrics import section_header
from components.charts import (
    review_score_distribution, review_by_delivery_chart,
    monthly_review_trend_chart, payment_type_pie,
    payment_value_by_type_chart, installment_chart
)
from components.filters import date_range_filter
from utils.helpers import df_to_csv_bytes, fmt_currency
import plotly.express as px


def show():
    st.title("Reviews & Payments")
    st.caption("Customer satisfaction analysis and payment behavior intelligence.")

    with st.spinner("Loading data..."):
        reviews  = load_review_table()
        payments = load_payment_table()

    # ── Sidebar Filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Filters")
        reviews  = date_range_filter(reviews,  date_col="review_creation_date",
                                     label="Review Date Range")
        payments = date_range_filter(payments, label="Payment Date Range")

    # ════════════════════════════════════════════════════════════════════════════
    # REVIEWS SECTION
    # ════════════════════════════════════════════════════════════════════════════
    section_header("Customer Reviews", "Satisfaction analysis based on review scores")

    rev_summary = get_review_summary(reviews)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Review Score",  f"{rev_summary['avg_score']} / 5.0")
    c2.metric("Total Reviews",     f"{rev_summary['total_reviews']:,}")
    c3.metric("Positive Reviews",  f"{rev_summary['pct_positive']}%",
              help="Reviews with score >= 4")
    c4.metric("Negative Reviews",  f"{rev_summary['pct_negative']}%",
              help="Reviews with score <= 2")

    # Score distribution & delivery comparison
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(review_score_distribution(reviews), use_container_width=True)
    with col2:
        score_by_del = get_score_by_delivery_status(reviews)
        st.plotly_chart(review_by_delivery_chart(score_by_del), use_container_width=True)

    # Monthly trend
    st.divider()
    section_header("Review Score Trend", "Monthly average satisfaction over time")
    monthly_scores = get_monthly_review_scores(reviews)
    st.plotly_chart(monthly_review_trend_chart(monthly_scores), use_container_width=True)

    # Review category breakdown
    st.divider()
    section_header("Review Category Distribution")
    if "review_category" in reviews.columns:
        cat_counts = reviews["review_category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        col3, col4 = st.columns(2)
        with col3:
            fig_cat = px.pie(
                cat_counts, names="category", values="count",
                title="Reviews by Category",
                template="plotly_white",
                color_discrete_sequence=["#27AE60", "#E74C3C", "#E8A838"],
                hole=0.4,
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        with col4:
            st.dataframe(
                cat_counts.rename(columns={"category": "Category", "count": "Reviews"}),
                use_container_width=True, hide_index=True
            )

    # Delivery days vs review score
    st.divider()
    section_header("Delivery Time vs Customer Satisfaction",
                   "Does faster delivery lead to better reviews?")

    reviews_clean = reviews[
        reviews["delivery_days"].notna() &
        reviews["review_score"].notna() &
        (reviews["delivery_days"] > 0) &
        (reviews["delivery_days"] < 60)
    ].copy()

    # Bin delivery days
    reviews_clean["delivery_bucket"] = reviews_clean["delivery_days"].apply(
        lambda x: "1-7 days" if x <= 7 else
                  "8-14 days" if x <= 14 else
                  "15-21 days" if x <= 21 else
                  "22-30 days" if x <= 30 else "30+ days"
    )

    bucket_order = ["1-7 days", "8-14 days", "15-21 days", "22-30 days", "30+ days"]
    bucket_stats = (
        reviews_clean.groupby("delivery_bucket")["review_score"]
        .agg(["mean", "count"])
        .round(2)
        .reset_index()
        .rename(columns={"mean": "avg_score", "count": "review_count"})
    )
    bucket_stats["delivery_bucket"] = pd.Categorical(
        bucket_stats["delivery_bucket"], categories=bucket_order, ordered=True
    )
    bucket_stats = bucket_stats.sort_values("delivery_bucket")

    fig_bucket = px.bar(
        bucket_stats, x="delivery_bucket", y="avg_score",
        title="Average Review Score by Delivery Time Bucket",
        labels={"delivery_bucket": "Delivery Time", "avg_score": "Avg Review Score"},
        template="plotly_white",
        color="avg_score",
        color_continuous_scale=[[0, "#E74C3C"], [0.5, "#E8A838"], [1, "#27AE60"]],
        text="avg_score",
    )
    fig_bucket.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_bucket.update_layout(yaxis_range=[0, 5.5], coloraxis_showscale=False)
    st.plotly_chart(fig_bucket, use_container_width=True)

    st.info(
        "**Business Insight**: Customers who receive orders within 7 days consistently "
        "give higher review scores. Delivery speed is a key driver of customer satisfaction."
    )

    # ════════════════════════════════════════════════════════════════════════════
    # PAYMENTS SECTION
    # ════════════════════════════════════════════════════════════════════════════
    st.divider()
    section_header("Payment Analytics", "Payment methods, values, and installment behavior")

    pay_summary = get_payment_summary(payments)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Payment Value", fmt_currency(pay_summary["total_payment_value"]))
    c6.metric("Avg Payment Value",   fmt_currency(pay_summary["avg_payment_value"]))
    c7.metric("Avg Installments",    f"{pay_summary['avg_installments']:.1f}x")
    c8.metric("Top Payment Method",  pay_summary["top_payment_method"].replace("_", " ").title())

    # Method distribution & value
    st.divider()
    pay_by_type = get_payment_by_type(payments)

    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(payment_type_pie(pay_by_type), use_container_width=True)
    with col6:
        st.plotly_chart(payment_value_by_type_chart(pay_by_type), use_container_width=True)

    # Installment analysis
    st.divider()
    section_header("Installment Behavior",
                   "How customers split payments across installments")

    inst_df = get_installment_distribution(payments)

    col7, col8 = st.columns(2)
    with col7:
        st.plotly_chart(installment_chart(inst_df), use_container_width=True)
    with col8:
        # Average payment value by installment count
        credit_only = payments[payments["payment_type"] == "credit_card"].copy()
        avg_by_inst = (
            credit_only[credit_only["payment_installments"] <= 12]
            .groupby("payment_installments")["payment_value"]
            .mean()
            .round(2)
            .reset_index()
        )
        fig_inst_val = px.line(
            avg_by_inst,
            x="payment_installments", y="payment_value",
            title="Avg Payment Value by Installments (Credit Card Only)",
            labels={"payment_installments": "Installments",
                    "payment_value": "Avg Value (R$)"},
            template="plotly_white",
            markers=True,
            color_discrete_sequence=["#E8A838"],
        )
        st.plotly_chart(fig_inst_val, use_container_width=True)

    # ── Downloads ─────────────────────────────────────────────────────────────
    with st.expander("Download Data"):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "Download Reviews CSV",
                data=df_to_csv_bytes(
                    reviews[["order_id", "review_score", "delivery_status",
                              "review_category", "review_creation_date"]]
                ),
                file_name="reviews_data.csv",
                mime="text/csv",
            )
        with col_d2:
            st.download_button(
                "Download Payment Summary CSV",
                data=df_to_csv_bytes(pay_by_type),
                file_name="payment_by_type.csv",
                mime="text/csv",
            )


import pandas as pd  # needed for pd.Categorical

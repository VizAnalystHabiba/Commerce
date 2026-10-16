"""
components/charts.py — Commerce360 AI Plotly Chart Library
All chart functions return Plotly figures.
Column names verified against actual CSV schemas.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import json
import plotly.utils
from config.config import (
    COLOR_SEQUENCE, PLOTLY_TEMPLATE, BRAND_COLOR,
    ACCENT_COLOR, SUCCESS_COLOR, DANGER_COLOR
)


def figure_to_json(fig) -> str:
    """Serialize Plotly figure to JSON string for HTML templates."""
    if fig is None:
        return "{}"
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# ── Revenue / Sales Charts ────────────────────────────────────────────────────

def revenue_trend_chart(monthly: pd.DataFrame) -> go.Figure:
    """Line chart: Monthly revenue trend. Needs columns: period, item_revenue."""
    fig = px.line(
        monthly, x="period", y="item_revenue",
        title="Monthly Revenue Trend",
        labels={"period": "Month", "item_revenue": "Revenue (R$)"},
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[BRAND_COLOR],
        markers=True,
    )
    fig.update_layout(xaxis_tickangle=-45, hovermode="x unified")
    return fig


def orders_trend_chart(monthly: pd.DataFrame) -> go.Figure:
    """Bar chart: Monthly order count trend. Needs columns: period, order_count."""
    fig = px.bar(
        monthly, x="period", y="order_count",
        title="Monthly Orders Trend",
        labels={"period": "Month", "order_count": "Orders"},
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[ACCENT_COLOR],
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def revenue_by_category_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: Revenue by category. Needs: category, revenue."""
    fig = px.bar(
        df.sort_values("revenue"), x="revenue", y="category",
        orientation="h",
        title="Revenue by Product Category",
        labels={"revenue": "Revenue (R$)", "category": "Category"},
        template=PLOTLY_TEMPLATE,
        color="revenue",
        color_continuous_scale=[[0, "#BDC3C7"], [1, BRAND_COLOR]],
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def top_products_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: Top products by revenue. Needs: product_id, total_revenue, category."""
    df = df.copy()
    df["short_id"] = df["product_id"].str[:8] + "..."
    fig = px.bar(
        df.sort_values("total_revenue"), x="total_revenue", y="short_id",
        orientation="h",
        title="Top Products by Revenue",
        labels={"total_revenue": "Revenue (R$)", "short_id": "Product"},
        color="category",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    return fig


# ── Customer Segment Charts ───────────────────────────────────────────────────

def segment_size_chart(summary: pd.DataFrame) -> go.Figure:
    """Pie chart: Customer segment distribution. Needs: segment_name, customer_count."""
    fig = px.pie(
        summary, names="segment_name", values="customer_count",
        title="Customer Segment Distribution",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def segment_spending_chart(summary: pd.DataFrame) -> go.Figure:
    """Bar: Avg spending per segment. Needs: segment_name, avg_spending."""
    fig = px.bar(
        summary.sort_values("avg_spending", ascending=False),
        x="segment_name", y="avg_spending",
        title="Average Spending by Customer Segment",
        labels={"segment_name": "Segment", "avg_spending": "Avg Spending (R$)"},
        template=PLOTLY_TEMPLATE,
        color="segment_name",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_layout(showlegend=False)
    return fig


def segment_scatter_chart(segs: pd.DataFrame) -> go.Figure:
    """
    Scatter: Customer segmentation map.
    Needs: total_spending, customer_recency_days, total_orders, segment_name.
    """
    fig = px.scatter(
        segs.sample(min(5000, len(segs)), random_state=42),
        x="customer_recency_days", y="total_spending",
        color="segment_name",
        size="total_orders",
        title="Customer Segmentation Map (Sample of 5,000)",
        labels={
            "customer_recency_days": "Days Since Last Purchase (Recency)",
            "total_spending": "Total Spending (R$)",
            "segment_name": "Segment",
        },
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        opacity=0.7,
    )
    return fig


def rfm_bar_chart(summary: pd.DataFrame) -> go.Figure:
    """Grouped bar: Orders and recency by segment."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Avg Orders",
        x=summary["segment_name"],
        y=summary["avg_orders"],
        marker_color=BRAND_COLOR,
    ))
    fig.add_trace(go.Bar(
        name="Avg Order Value (R$)",
        x=summary["segment_name"],
        y=summary["avg_order_value"],
        marker_color=ACCENT_COLOR,
    ))
    fig.update_layout(
        barmode="group",
        title="Orders & Order Value by Segment",
        template=PLOTLY_TEMPLATE,
        xaxis_title="Segment",
    )
    return fig


# ── Seller Charts ─────────────────────────────────────────────────────────────

def top_sellers_revenue_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: Top sellers by revenue."""
    df = df.copy()
    df["short_id"] = df["seller_id"].str[:8] + "..."
    fig = px.bar(
        df.sort_values("seller_revenue"), x="seller_revenue", y="short_id",
        orientation="h",
        title="Top Sellers by Revenue",
        labels={"seller_revenue": "Revenue (R$)", "short_id": "Seller"},
        color="seller_state",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    return fig


def seller_revenue_distribution_chart(seller_features: pd.DataFrame) -> go.Figure:
    """Histogram: Distribution of seller revenues."""
    fig = px.histogram(
        seller_features, x="seller_revenue",
        nbins=50,
        title="Seller Revenue Distribution",
        labels={"seller_revenue": "Revenue (R$)"},
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[BRAND_COLOR],
    )
    fig.update_layout(bargap=0.1)
    return fig


# ── Delivery Charts ───────────────────────────────────────────────────────────

def delivery_status_pie(summary: dict) -> go.Figure:
    """Pie: On-time vs delayed."""
    fig = go.Figure(data=[go.Pie(
        labels=["On Time", "Delayed"],
        values=[summary["on_time"], summary["delayed"]],
        hole=0.4,
        marker=dict(colors=[SUCCESS_COLOR, DANGER_COLOR]),
    )])
    fig.update_layout(title="Delivery Performance Overview", template=PLOTLY_TEMPLATE)
    return fig


def delivery_days_histogram(delivery: pd.DataFrame) -> go.Figure:
    """Histogram: Distribution of delivery days."""
    delivered = delivery[delivery["order_status"] == "delivered"].copy()
    fig = px.histogram(
        delivered, x="delivery_days", nbins=40,
        title="Distribution of Delivery Times",
        labels={"delivery_days": "Delivery Days"},
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[BRAND_COLOR],
    )
    fig.update_layout(bargap=0.05)
    return fig


def delivery_delay_distribution(delivery: pd.DataFrame) -> go.Figure:
    """Histogram of delay days for delayed orders only."""
    delayed = delivery[
        (delivery["order_status"] == "delivered") &
        (delivery["delivery_delay_days"] > 0)
    ].copy()
    fig = px.histogram(
        delayed, x="delivery_delay_days", nbins=30,
        title="Delay Distribution (Delayed Orders Only)",
        labels={"delivery_delay_days": "Days Delayed"},
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[DANGER_COLOR],
    )
    return fig


def delivery_by_state_chart(state_df: pd.DataFrame) -> go.Figure:
    """Bar: Avg delivery days by customer state."""
    fig = px.bar(
        state_df.head(20).sort_values("avg_delivery_days", ascending=False),
        x="customer_state", y="avg_delivery_days",
        title="Average Delivery Days by Customer State",
        labels={"customer_state": "State", "avg_delivery_days": "Avg Days"},
        color="on_time_pct",
        color_continuous_scale=[[0, DANGER_COLOR], [0.5, ACCENT_COLOR], [1, SUCCESS_COLOR]],
        template=PLOTLY_TEMPLATE,
    )
    fig.update_coloraxes(colorbar_title="On-Time %")
    return fig


def delivery_trend_chart(trend_df: pd.DataFrame) -> go.Figure:
    """Dual-axis: Monthly delivery days and on-time %."""
    trend_df = trend_df.copy()
    trend_df["period"] = (
        trend_df["year"].astype(str) + "-"
        + trend_df["month"].astype(str).str.zfill(2)
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Avg Delivery Days",
        x=trend_df["period"],
        y=trend_df["avg_delivery_days"],
        marker_color=BRAND_COLOR,
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        name="On-Time %",
        x=trend_df["period"],
        y=trend_df["on_time_pct"],
        mode="lines+markers",
        marker_color=SUCCESS_COLOR,
        yaxis="y2",
    ))
    fig.update_layout(
        title="Monthly Delivery Performance",
        template=PLOTLY_TEMPLATE,
        xaxis_tickangle=-45,
        yaxis=dict(title="Avg Delivery Days"),
        yaxis2=dict(title="On-Time %", overlaying="y", side="right"),
        hovermode="x unified",
    )
    return fig


# ── Review Charts ─────────────────────────────────────────────────────────────

def review_score_distribution(reviews: pd.DataFrame) -> go.Figure:
    """Bar: Distribution of review scores 1-5."""
    counts = reviews["review_score"].value_counts().sort_index().reset_index()
    counts.columns = ["score", "count"]
    colors = {1: DANGER_COLOR, 2: "#E67E22", 3: ACCENT_COLOR,
              4: "#2ECC71", 5: SUCCESS_COLOR}
    fig = px.bar(
        counts, x="score", y="count",
        title="Review Score Distribution",
        labels={"score": "Review Score", "count": "Number of Reviews"},
        template=PLOTLY_TEMPLATE,
        color="score",
        color_discrete_map=colors,
    )
    fig.update_layout(showlegend=False)
    return fig


def review_by_delivery_chart(df: pd.DataFrame) -> go.Figure:
    """Bar: Avg review score by delivery status."""
    fig = px.bar(
        df, x="delivery_status", y="avg_score",
        title="Review Score by Delivery Status",
        labels={"delivery_status": "Delivery Status", "avg_score": "Avg Review Score"},
        template=PLOTLY_TEMPLATE,
        color="delivery_status",
        color_discrete_map={"On Time": SUCCESS_COLOR, "Delayed": DANGER_COLOR},
        text="avg_score",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, 5.5])
    return fig


def monthly_review_trend_chart(monthly: pd.DataFrame) -> go.Figure:
    """Line: Monthly avg review score trend."""
    monthly = monthly.copy()
    monthly["period"] = (
        monthly["year"].astype(str) + "-"
        + monthly["month"].astype(str).str.zfill(2)
    )
    fig = px.line(
        monthly, x="period", y="review_score",
        title="Monthly Average Review Score",
        labels={"period": "Month", "review_score": "Avg Score"},
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[ACCENT_COLOR],
        markers=True,
    )
    fig.update_layout(yaxis_range=[1, 5.5], xaxis_tickangle=-45)
    return fig


# ── Payment Charts ────────────────────────────────────────────────────────────

def payment_type_pie(payment_by_type: pd.DataFrame) -> go.Figure:
    """Pie: Payment method distribution by transaction count."""
    fig = px.pie(
        payment_by_type, names="payment_type", values="transaction_count",
        title="Payment Method Distribution",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        hole=0.4,
    )
    return fig


def payment_value_by_type_chart(payment_by_type: pd.DataFrame) -> go.Figure:
    """Bar: Total payment value by payment type."""
    fig = px.bar(
        payment_by_type.sort_values("total_value", ascending=False),
        x="payment_type", y="total_value",
        title="Total Payment Value by Payment Method",
        labels={"payment_type": "Payment Method", "total_value": "Total Value (R$)"},
        template=PLOTLY_TEMPLATE,
        color="payment_type",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_layout(showlegend=False)
    return fig


def installment_chart(installment_df: pd.DataFrame) -> go.Figure:
    """Bar: Transactions by installment count."""
    df = installment_df[installment_df["payment_installments"] <= 12].copy()
    fig = px.bar(
        df, x="payment_installments", y="transactions",
        title="Purchase Count by Number of Installments",
        labels={"payment_installments": "Installments", "transactions": "Transactions"},
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[BRAND_COLOR],
    )
    return fig


# ── ML Charts ─────────────────────────────────────────────────────────────────

def ml_metrics_bar(ml_results: pd.DataFrame) -> go.Figure:
    """Grouped bar: ML model comparison across all metrics."""
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    fig = go.Figure()
    colors = [BRAND_COLOR, ACCENT_COLOR]
    for i, row in ml_results.iterrows():
        fig.add_trace(go.Bar(
            name=row["model"].replace(" (Pre-Delivery)", ""),
            x=metrics,
            y=[row[m] for m in metrics],
            marker_color=colors[i % len(colors)],
        ))
    fig.update_layout(
        barmode="group",
        title="ML Model Performance Comparison (Pre-Delivery Prediction)",
        template=PLOTLY_TEMPLATE,
        yaxis=dict(title="Score", range=[0, 1]),
        xaxis_title="Metric",
    )
    return fig


def feature_importance_chart(importances: dict) -> go.Figure:
    """Horizontal bar: Feature importances."""
    df = pd.DataFrame(list(importances.items()),
                      columns=["feature", "importance"])
    df = df.sort_values("importance", ascending=True)
    fig = px.bar(
        df, x="importance", y="feature", orientation="h",
        title="Feature Importance (Random Forest)",
        labels={"importance": "Importance Score", "feature": "Feature"},
        template=PLOTLY_TEMPLATE,
        color="importance",
        color_continuous_scale=[[0, "#BDC3C7"], [1, BRAND_COLOR]],
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig

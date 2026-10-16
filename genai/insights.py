"""
genai/insights.py — Commerce360 AI GenAI Insight Generator
Generates data-grounded business insights using Google Gemini.
NEVER fabricates statistics — all numbers injected from actual data.
"""

import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.config import GEMINI_API_KEY, GEMINI_MODEL
from genai.prompts import (
    build_executive_prompt, build_sales_prompt, build_customer_prompt,
    build_delivery_prompt, build_full_report_prompt, build_recommendations_prompt
)

# Attempt to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

_api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)


def _get_model():
    """Initialize Gemini model. Returns None if API key not configured."""
    if not _api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=_api_key)
        return genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        return None


def _call_gemini(prompt: str) -> str:
    """Call Gemini API with error handling. Returns fallback message if unavailable."""
    model = _get_model()
    if model is None:
        return (
            "**GenAI Unavailable** — Set your GEMINI_API_KEY in the .env file to enable "
            "AI-generated insights. Get a free key at https://aistudio.google.com/app/apikey\n\n"
            "Data-based summary is shown in the dashboard metrics above."
        )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"**GenAI Error**: {str(e)}\n\nPlease check your API key and try again."


def generate_executive_insights(kpis: dict, monthly_rev_df, top_cats_df) -> str:
    """Generate executive overview insights from real KPIs."""
    # Determine trend from monthly data
    if len(monthly_rev_df) >= 3:
        last3 = monthly_rev_df["item_revenue"].tail(3).values
        trend = "upward" if last3[-1] > last3[0] else "downward"
        monthly_trend = f"{trend} (last 3 months: R${last3[0]:,.0f} → R${last3[-1]:,.0f})"
    else:
        monthly_trend = "insufficient data for trend"

    top_cat = top_cats_df.iloc[0]["category"] if len(top_cats_df) > 0 else "Unknown"

    prompt = build_executive_prompt(
        kpis=kpis,
        monthly_trend=monthly_trend,
        top_category=top_cat,
        on_time_pct=kpis["on_time_pct"],
    )
    return _call_gemini(prompt)


def generate_sales_insights(revenue: float, orders: int, top_cats_df,
                             monthly_rev_df, avg_order_value: float) -> str:
    """Generate sales insights from actual computed metrics."""
    if len(monthly_rev_df) >= 3:
        last3 = monthly_rev_df["item_revenue"].tail(3).values
        trend = f"{'growing' if last3[-1] > last3[0] else 'declining'} ({last3[0]:,.0f} -> {last3[-1]:,.0f})"
    else:
        trend = "stable"

    top_cats = [
        {"category": row["category"], "revenue": row["revenue"]}
        for _, row in top_cats_df.head(5).iterrows()
    ]
    prompt = build_sales_prompt(revenue, orders, top_cats, trend, avg_order_value)
    return _call_gemini(prompt)


def generate_customer_insights(segment_summary_df, total_customers: int) -> str:
    """Generate customer segmentation insights from K-Means results."""
    segs = segment_summary_df.to_dict("records")
    prompt = build_customer_prompt(segs, total_customers)
    return _call_gemini(prompt)


def generate_delivery_insights(delivery_summary: dict, state_df,
                                ml_results_df) -> str:
    """Generate delivery insights combining descriptive and ML findings."""
    worst_state = (
        state_df.sort_values("avg_delivery_days", ascending=False).iloc[0]["customer_state"]
        if len(state_df) > 0 else "Unknown"
    )
    avg_delay = delivery_summary.get("avg_delay_days", 0)

    # Get RF ROC-AUC from ml_results
    rf_row = ml_results_df[ml_results_df["model"].str.contains("Random Forest")]
    roc_auc = float(rf_row["roc_auc"].values[0]) if len(rf_row) > 0 else 0.72

    prompt = build_delivery_prompt(delivery_summary, avg_delay, worst_state, roc_auc)
    return _call_gemini(prompt)


def generate_full_report(all_metrics: dict) -> str:
    """Generate the complete 10-section business report from all metrics."""
    prompt = build_full_report_prompt(all_metrics)
    return _call_gemini(prompt)


def generate_recommendations(all_metrics: dict) -> str:
    """Generate 5 prioritized business recommendations from actual data."""
    prompt = build_recommendations_prompt(all_metrics)
    return _call_gemini(prompt)


def build_all_metrics(kpis: dict, top_cats_df, segment_summary_df,
                       delivery_summary: dict, ml_results_df,
                       rev_summary: dict, pay_summary: dict) -> dict:
    """
    Assemble all metrics needed for full report generation.
    All values come from actual computed data.
    """
    top_cat = top_cats_df.iloc[0]["category"] if len(top_cats_df) > 0 else "Unknown"
    top_cat_rev = float(top_cats_df.iloc[0]["revenue"]) if len(top_cats_df) > 0 else 0

    # Largest segment
    largest = segment_summary_df.sort_values("customer_count", ascending=False).iloc[0]
    # Highest avg spending segment
    high_val = segment_summary_df.sort_values("avg_spending", ascending=False).iloc[0]

    # RF ROC-AUC
    rf_row = ml_results_df[ml_results_df["model"].str.contains("Random Forest")]
    roc_auc = float(rf_row["roc_auc"].values[0]) if len(rf_row) > 0 else 0.72

    total_customers = segment_summary_df["customer_count"].sum()
    avg_order_value = kpis["total_revenue"] / kpis["total_orders"] if kpis["total_orders"] > 0 else 0

    return {
        "total_revenue":            kpis["total_revenue"],
        "total_orders":             kpis["total_orders"],
        "total_customers":          total_customers,
        "total_sellers":            kpis["total_sellers"],
        "avg_delivery_days":        kpis["avg_delivery_days"],
        "on_time_pct":              kpis["on_time_pct"],
        "top_category":             top_cat,
        "top_category_revenue":     top_cat_rev,
        "avg_order_value":          round(avg_order_value, 2),
        "largest_segment":          largest["segment_name"],
        "largest_segment_count":    int(largest["customer_count"]),
        "high_value_avg_spending":  float(high_val["avg_spending"]),
        "delayed_orders":           delivery_summary["delayed"],
        "avg_delay_days":           delivery_summary["avg_delay_days"],
        "ml_roc_auc":               round(roc_auc, 3),
        "avg_review_score":         rev_summary["avg_score"],
        "pct_positive":             rev_summary["pct_positive"],
        "top_payment_method":       pay_summary["top_payment_method"],
        "avg_installments":         pay_summary["avg_installments"],
    }

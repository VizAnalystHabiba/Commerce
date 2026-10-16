"""
genai/prompts.py — Commerce360 AI Prompt Templates
Data-grounded prompt builders. Never fabricate statistics.
All numbers are injected from actual computed data.
"""


def build_executive_prompt(kpis: dict, monthly_trend: str,
                            top_category: str, on_time_pct: float) -> str:
    return f"""
You are a senior business intelligence analyst for an e-commerce company.
Analyze the following ACTUAL business metrics and generate a concise executive summary.

VERIFIED BUSINESS DATA:
- Total Revenue: R${kpis['total_revenue']:,.2f}
- Total Orders: {kpis['total_orders']:,}
- Total Customers: {kpis['total_customers']:,}
- Active Sellers: {kpis['total_sellers']:,}
- Average Delivery Time: {kpis['avg_delivery_days']} days
- On-Time Delivery Rate: {kpis['on_time_pct']}%
- Revenue Trend: {monthly_trend}
- Top Revenue Category: {top_category}
- Delivery On-Time %: {on_time_pct}%

INSTRUCTIONS:
1. Summarize the overall business health in 3-4 sentences.
2. Identify the 2-3 most important findings from the data above.
3. Flag any concern (delivery rate below 90%, negative trend, etc.).
4. Keep language executive-level: clear, concise, business-focused.
5. Do NOT invent any numbers. Use ONLY the data provided above.
6. Do NOT use generic filler phrases like "it is important to note".
"""


def build_sales_prompt(revenue: float, orders: int, top_categories: list,
                       monthly_trend: str, avg_order_value: float) -> str:
    cats_str = "\n".join([f"  - {c['category']}: R${c['revenue']:,.2f}" for c in top_categories[:5]])
    return f"""
You are a sales analytics expert for an e-commerce company.
Analyze these ACTUAL sales figures and provide business insights.

VERIFIED SALES DATA:
- Total Revenue: R${revenue:,.2f}
- Total Orders: {orders:,}
- Average Order Value: R${avg_order_value:,.2f}
- Revenue Trend: {monthly_trend}
- Top 5 Revenue Categories:
{cats_str}

INSTRUCTIONS:
1. Identify which categories are driving revenue and why this matters.
2. Comment on the average order value — is it healthy for e-commerce?
3. Identify any sales concentration risk (too dependent on one category?).
4. Provide 2 concrete, data-backed sales recommendations.
5. Use ONLY the data above. Do not fabricate statistics.
"""


def build_customer_prompt(segment_summary: list, total_customers: int) -> str:
    segs_str = "\n".join([
        f"  - {s['segment_name']}: {s['customer_count']:,} customers, "
        f"avg spending R${s['avg_spending']:,.2f}, avg orders {s['avg_orders']:.1f}"
        for s in segment_summary
    ])
    return f"""
You are a customer intelligence expert for an e-commerce company.
Analyze these ACTUAL customer segmentation results.

VERIFIED CUSTOMER DATA:
- Total Customers: {total_customers:,}
- K-Means Segments (k=4, features: spending, orders, recency):
{segs_str}

INSTRUCTIONS:
1. Describe what each customer segment represents in business terms.
2. Identify the most valuable segment and why.
3. Identify the highest-risk segment and what the business should do.
4. Provide 2-3 concrete retention/growth strategies for specific segments.
5. Comment on the customer base distribution — is it healthy?
6. Use ONLY the data above. Do not invent customer counts or percentages.
"""


def build_delivery_prompt(delivery_summary: dict, avg_delay: float,
                          worst_state: str, ml_roc_auc: float) -> str:
    return f"""
You are a logistics and operations analyst for an e-commerce company.
Analyze these ACTUAL delivery performance metrics.

VERIFIED DELIVERY DATA:
- Total Delivered Orders: {delivery_summary['total_delivered']:,}
- On-Time Rate: {delivery_summary['on_time_pct']}%
- Delayed Orders: {delivery_summary['delayed']:,}
- Average Delivery Time: {delivery_summary['avg_delivery_days']} days
- Average Delay (when delayed): {avg_delay} days
- Worst Performing State: {worst_state}
- ML Delay Prediction ROC-AUC: {ml_roc_auc:.3f} (Random Forest, pre-delivery features only)

INSTRUCTIONS:
1. Assess whether the on-time delivery rate is acceptable for e-commerce (benchmark: >90%).
2. Quantify the business risk of the delay rate.
3. Explain what a ROC-AUC of {ml_roc_auc:.2f} means practically for delay prediction.
4. Provide 2-3 specific operational recommendations to improve delivery performance.
5. Mention the geographic concentration of delays if relevant.
6. Use ONLY the data above. Do not fabricate metrics.
"""


def build_full_report_prompt(all_metrics: dict) -> str:
    """Build a comprehensive report prompt from all computed metrics."""
    return f"""
You are the Chief Analytics Officer of an e-commerce company.
Generate a professional business intelligence report using ONLY the verified data below.

VERIFIED COMMERCE360 AI METRICS:

EXECUTIVE:
- Revenue: R${all_metrics['total_revenue']:,.2f}
- Orders: {all_metrics['total_orders']:,}
- Customers: {all_metrics['total_customers']:,}
- Sellers: {all_metrics['total_sellers']:,}
- Avg Delivery: {all_metrics['avg_delivery_days']} days
- On-Time %: {all_metrics['on_time_pct']}%

SALES:
- Top Category: {all_metrics['top_category']}
- Top Category Revenue: R${all_metrics['top_category_revenue']:,.2f}
- Avg Order Value: R${all_metrics['avg_order_value']:,.2f}

CUSTOMERS:
- Largest Segment: {all_metrics['largest_segment']} ({all_metrics['largest_segment_count']:,} customers)
- Highest Value Segment Avg Spending: R${all_metrics['high_value_avg_spending']:,.2f}

DELIVERY:
- Delayed Orders: {all_metrics['delayed_orders']:,}
- Avg Delay: {all_metrics['avg_delay_days']} days
- ML ROC-AUC: {all_metrics['ml_roc_auc']}

REVIEWS:
- Avg Score: {all_metrics['avg_review_score']} / 5.0
- Positive Reviews: {all_metrics['pct_positive']}%

PAYMENTS:
- Top Method: {all_metrics['top_payment_method']}
- Avg Installments: {all_metrics['avg_installments']}x

REPORT STRUCTURE (generate all 10 sections):

1. EXECUTIVE SUMMARY (3-4 sentences, business health overview)
2. SALES INSIGHTS (key revenue findings, trends, category analysis)
3. PRODUCT INSIGHTS (top performers, concentration risk)
4. CUSTOMER INSIGHTS (segmentation findings, retention priorities)
5. SELLER INSIGHTS (performance distribution, geographic patterns)
6. DELIVERY INSIGHTS (on-time rate assessment, improvement areas)
7. REVIEW INSIGHTS (satisfaction level, delivery-review correlation)
8. PAYMENT INSIGHTS (payment behavior, installment patterns)
9. MACHINE LEARNING INSIGHTS (delivery prediction, segmentation quality)
10. BUSINESS RECOMMENDATIONS (5 prioritized, actionable recommendations)

RULES:
- Use only the data provided. Never invent statistics.
- Each section should be 3-5 sentences.
- Recommendations must reference specific data points.
- Use professional business language.
- Be direct — no filler phrases.
"""


def build_recommendations_prompt(all_metrics: dict) -> str:
    """Generate targeted business recommendations from actual metrics."""
    return f"""
You are a management consultant analyzing an e-commerce business.
Generate exactly 5 specific, prioritized business recommendations.

ACTUAL DATA:
- On-Time Delivery: {all_metrics['on_time_pct']}% (industry benchmark: >90%)
- Delayed Orders: {all_metrics['delayed_orders']:,}
- Avg Review Score: {all_metrics['avg_review_score']} / 5.0
- Top Revenue Category: {all_metrics['top_category']}
- Customer Segments: Low-Engagement = majority of customer base
- ML Delay Prediction ROC-AUC: {all_metrics['ml_roc_auc']} (Random Forest)

For each recommendation:
1. State the specific problem or opportunity
2. Reference the actual data point that supports it
3. Describe the concrete action to take
4. Estimate the expected business impact

Use ONLY the data provided. Number each recommendation 1-5 by priority.
"""

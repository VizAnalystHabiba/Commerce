"""
genai/insight_generator.py — Commerce360 AI Insight Generator Module Wrapper
Provides unified interface to the GenAI insights layer.
"""

from genai.insights import (
    generate_executive_insights,
    generate_sales_insights,
    generate_customer_insights,
    generate_delivery_insights,
    generate_full_report,
    generate_recommendations,
    build_all_metrics,
)

__all__ = [
    "generate_executive_insights",
    "generate_sales_insights",
    "generate_customer_insights",
    "generate_delivery_insights",
    "generate_full_report",
    "generate_recommendations",
    "build_all_metrics",
]

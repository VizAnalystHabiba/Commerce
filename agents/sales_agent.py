"""
agents/sales_agent.py — Commerce360 AI Sales Analytics Agent
============================================================
Specialized analytical agent responsible for:
  - Monitoring total revenue, order count, and AOV
  - Tracking category concentration and revenue contributions
  - Identifying seasonal peaks and trends
  - Generating sales performance summaries
"""

import pandas as pd
from typing import Dict, Any


class SalesAgent:
    """Specialized analytical agent for e-commerce sales and product intelligence."""

    def __init__(self, sales_df: pd.DataFrame):
        self.sales = sales_df

    def analyze(self) -> Dict[str, Any]:
        """Runs comprehensive sales analysis and returns structured insights."""
        total_revenue = float(self.sales["item_revenue"].sum())
        total_orders = int(self.sales["order_id"].nunique())
        total_items = int(len(self.sales))
        aov = total_revenue / total_orders if total_orders > 0 else 0.0

        # Category revenue aggregation
        cat_rev = (
            self.sales.groupby("product_category_name_english")["item_revenue"]
            .sum()
            .sort_values(ascending=False)
        )
        top_category = cat_rev.index[0] if len(cat_rev) > 0 else "Unknown"
        top_cat_revenue = float(cat_rev.iloc[0]) if len(cat_rev) > 0 else 0.0
        top_cat_share = (top_cat_revenue / total_revenue * 100) if total_revenue > 0 else 0.0

        top_5_share = float(cat_rev.head(5).sum() / total_revenue * 100) if total_revenue > 0 else 0.0

        # Monthly trajectory
        monthly = (
            self.sales.groupby(["order_year", "order_month"])["item_revenue"]
            .sum()
            .reset_index()
            .sort_values(["order_year", "order_month"])
        )

        trend_status = "Stable"
        if len(monthly) >= 3:
            recent_growth = (
                (monthly.iloc[-1]["item_revenue"] - monthly.iloc[-3]["item_revenue"])
                / (monthly.iloc[-3]["item_revenue"] + 1e-5)
            )
            if recent_growth > 0.05:
                trend_status = f"Expanding (+{recent_growth*100:.1f}% over last quarter)"
            elif recent_growth < -0.05:
                trend_status = f"Contracting ({recent_growth*100:.1f}% over last quarter)"

        return {
            "agent_name": "Sales Analytics Agent",
            "total_revenue": total_revenue,
            "total_orders": total_orders,
            "total_items_sold": total_items,
            "average_order_value": round(aov, 2),
            "top_category": top_category,
            "top_category_revenue": round(top_cat_revenue, 2),
            "top_category_share_pct": round(top_cat_share, 2),
            "top_5_category_concentration_pct": round(top_5_share, 2),
            "revenue_trend": trend_status,
            "insights": [
                f"Generated R$ {total_revenue:,.2f} across {total_orders:,} orders with an AOV of R$ {aov:,.2f}.",
                f"Top category '{top_category}' accounts for {top_cat_share:.1f}% of total sales.",
                f"Top 5 categories represent {top_5_share:.1f}% of gross revenue.",
                f"Overall sales volume trend is currently {trend_status}."
            ]
        }

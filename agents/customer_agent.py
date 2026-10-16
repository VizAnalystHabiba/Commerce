"""
agents/customer_agent.py — Commerce360 AI Customer Intelligence Agent
=====================================================================
Specialized analytical agent responsible for:
  - Customer segmentation behavioral profiling
  - Customer lifetime spending and order frequency
  - Recency distribution and dormancy risk
  - Segment-specific growth and retention recommendations
"""

import pandas as pd
from typing import Dict, Any


class CustomerAgent:
    """Specialized analytical agent for customer segmentation and behavior analysis."""

    def __init__(self, customer_features_df: pd.DataFrame, customer_segments_df: pd.DataFrame):
        self.customers = customer_features_df
        self.segments = customer_segments_df

    def analyze(self) -> Dict[str, Any]:
        """Runs customer behavior and segmentation analysis."""
        total_customers = int(self.customers["customer_id"].nunique())
        avg_spending = float(self.customers["total_spending"].mean())
        avg_recency = float(self.customers["customer_recency_days"].mean())

        # Match segment summary
        merged = self.segments.copy()
        if "segment_name" not in merged.columns and "segment" in merged.columns:
            merged["segment_name"] = "Segment " + merged["segment"].astype(str)

        seg_counts = merged["segment_name"].value_counts().to_dict()
        seg_avg_spend = merged.groupby("segment_name")["total_spending"].mean().to_dict()

        # Identification of key segments
        high_value_seg = max(seg_avg_spend, key=seg_avg_spend.get) if seg_avg_spend else "None"
        largest_seg = max(seg_counts, key=seg_counts.get) if seg_counts else "None"

        # Repeat customer rate
        multi_order_customers = (self.customers["total_orders"] > 1).sum()
        repeat_rate = (multi_order_customers / total_customers * 100) if total_customers > 0 else 0.0

        return {
            "agent_name": "Customer Intelligence Agent",
            "total_customers": total_customers,
            "average_spending": round(avg_spending, 2),
            "average_recency_days": round(avg_recency, 1),
            "repeat_customer_rate_pct": round(repeat_rate, 2),
            "largest_segment": largest_seg,
            "largest_segment_size": seg_counts.get(largest_seg, 0),
            "high_value_segment": high_value_seg,
            "high_value_avg_spend": round(seg_avg_spend.get(high_value_seg, 0.0), 2),
            "segment_breakdown": {
                name: {
                    "count": int(seg_counts.get(name, 0)),
                    "avg_spending": round(seg_avg_spend.get(name, 0.0), 2)
                }
                for name in seg_counts
            },
            "insights": [
                f"Customer base includes {total_customers:,} buyers with an average spend of R$ {avg_spending:,.2f}.",
                f"Repeat customer rate is {repeat_rate:.1f}%, indicating a predominantly single-purchase acquisition profile.",
                f"Largest cluster is '{largest_seg}' with {seg_counts.get(largest_seg, 0):,} customers.",
                f"Top tier '{high_value_seg}' spends an average of R$ {seg_avg_spend.get(high_value_seg, 0.0):,.2f} per customer."
            ]
        }

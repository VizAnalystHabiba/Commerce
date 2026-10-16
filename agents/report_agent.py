"""
agents/report_agent.py — Commerce360 AI Report Coordinator Agent
================================================================
Specialized coordinator agent that:
  - Invokes Sales, Customer, and Delivery analytics agents
  - Synthesizes multi-domain findings into cohesive executive intelligence
  - Cross-correlates metrics (e.g. sales vs delivery, reviews vs delivery)
  - Formulates prioritized cross-functional business recommendations
"""

from typing import Dict, Any, List
from agents.sales_agent import SalesAgent
from agents.customer_agent import CustomerAgent
from agents.delivery_agent import DeliveryAgent


class ReportAgent:
    """Master coordinator agent synthesizing analytics across all domains."""

    def __init__(self, sales_agent: SalesAgent, customer_agent: CustomerAgent, delivery_agent: DeliveryAgent):
        self.sales_agent = sales_agent
        self.customer_agent = customer_agent
        self.delivery_agent = delivery_agent

    def coordinate_and_report(self) -> Dict[str, Any]:
        """Runs full multi-agent analysis and synthesizes comprehensive insights."""
        sales_analysis = self.sales_agent.analyze()
        customer_analysis = self.customer_agent.analyze()
        delivery_analysis = self.delivery_agent.analyze()

        # Cross-functional synthesis
        recommendations = [
            f"Protect {sales_analysis['top_category']} category market share while reducing freight friction for cross-state buyers.",
            f"Deploy targeted VIP retention incentives for {customer_analysis['high_value_segment']} buyers who contribute highest lifetime value.",
            f"Activate pre-delivery risk alerts (ML AUC: {delivery_analysis['ml_delay_model_roc_auc']}) on orders shipping to {delivery_analysis['slowest_state']}.",
            f"Implement carrier service level agreements (SLAs) to reduce the {delivery_analysis['delayed_orders']:,} delayed orders backlog.",
            f"Offer flexible multi-installment incentives to elevate the current average order value of R$ {sales_analysis['average_order_value']:.2f}."
        ]

        executive_summary = (
            f"Commerce360 AI synthesis: The business achieved R$ {sales_analysis['total_revenue']:,.2f} in revenue "
            f"across {sales_analysis['total_orders']:,} orders. Fulfillment health records an on-time rate of "
            f"{delivery_analysis['on_time_rate_pct']}%. Customer segmentation identifies {customer_analysis['total_customers']:,} "
            f"unique buyers, anchored by the '{customer_analysis['high_value_segment']}' tier."
        )

        return {
            "agent_name": "Report Coordinator Agent",
            "executive_summary": executive_summary,
            "domain_findings": {
                "sales": sales_analysis,
                "customer": customer_analysis,
                "delivery": delivery_analysis,
            },
            "cross_functional_recommendations": recommendations
        }

"""
pages/ai_business_insights.py — Commerce360 AI GenAI & Specialized Agents Page
Page 7: Data-grounded GenAI business insights, specialized analytics agents,
and automated executive report generation with instant download.
"""

import streamlit as st
import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_loader import (
    load_sales_features, load_customer_features, load_customer_segments,
    load_delivery_features, load_review_table, load_payment_table,
    load_ml_results, load_product_features, load_product_table,
    load_seller_features
)
from utils.calculations import (
    get_executive_kpis, get_monthly_revenue, get_revenue_by_category,
    get_top_products, get_segment_summary, get_delivery_summary,
    get_delivery_by_state, get_review_summary, get_payment_summary
)
from genai.insights import (
    generate_executive_insights, generate_sales_insights,
    generate_customer_insights, generate_delivery_insights,
    generate_recommendations, build_all_metrics, _get_model
)
from genai.report_generator import generate_automated_report
from agents.sales_agent import SalesAgent
from agents.customer_agent import CustomerAgent
from agents.delivery_agent import DeliveryAgent
from agents.report_agent import ReportAgent
from components.metrics import section_header


def show():
    st.title("GenAI Business Insights & Analytics Agents")
    st.caption(
        "Factual, data-grounded AI intelligence engine and specialized analytical agents. "
        "Every insight is derived directly from computed metrics."
    )

    with st.spinner("Loading analytical data for AI engine..."):
        sales = load_sales_features()
        cust_feat = load_customer_features()
        cust_seg = load_customer_segments()
        sell_feat = load_seller_features()
        delivery = load_delivery_features()
        reviews = load_review_table()
        payments = load_payment_table()
        ml_results = load_ml_results()
        prod_feat = load_product_features()
        prod_table = load_product_table()

    # Pre-calculate business metrics
    kpis = get_executive_kpis(sales, delivery, cust_feat, sell_feat)
    monthly_rev = get_monthly_revenue(sales)
    top_cats = get_revenue_by_category(sales, top_n=10)
    top_prods = get_top_products(prod_feat, prod_table, top_n=10)
    seg_summary = get_segment_summary(cust_seg, cust_feat)
    del_summary = get_delivery_summary(delivery)
    state_del = get_delivery_by_state(delivery)
    rev_summary = get_review_summary(reviews)
    pay_summary = get_payment_summary(payments)

    all_metrics = build_all_metrics(
        kpis=kpis,
        top_cats_df=top_cats,
        segment_summary_df=seg_summary,
        delivery_summary=del_summary,
        ml_results_df=ml_results,
        rev_summary=rev_summary,
        pay_summary=pay_summary
    )

    # Status badge for GenAI connection
    has_api_key = _get_model() is not None
    with st.sidebar:
        st.subheader("GenAI Settings")
        if has_api_key:
            st.success("Gemini API Connected")
        else:
            st.info("Operating in Factual Analytical Mode (Set GEMINI_API_KEY in .env for live LLM synthesis)")

    # Tabs for organization
    tab1, tab2, tab3, tab4 = st.tabs([
        "Automated Executive Report",
        "Domain AI Insights",
        "Specialized Analytics Agents",
        "Strategic Recommendations"
    ])

    # ── TAB 1: AUTOMATED EXECUTIVE REPORT ─────────────────────────────────────
    with tab1:
        section_header(
            "Executive Analytical Report Generation",
            "Generates a complete 10-section e-commerce decision intelligence report"
        )

        col_rep1, col_rep2 = st.columns([3, 1])
        with col_rep1:
            st.write(
                "Click below to generate the consolidated executive intelligence report. "
                "The report synthesizes sales, customers, logistics, machine learning models, and recommendations."
            )
        with col_rep2:
            gen_btn = st.button("Generate Full Report", type="primary", use_container_width=True)

        if gen_btn or "full_report_content" in st.session_state:
            if gen_btn:
                with st.spinner("Synthesizing multi-domain intelligence..."):
                    report_text = generate_automated_report(
                        metrics=all_metrics,
                        top_cats=top_cats,
                        top_prods=top_prods,
                        seg_summary=seg_summary,
                        ml_results=ml_results,
                        use_llm=has_api_key
                    )
                    st.session_state["full_report_content"] = report_text

            st.markdown(st.session_state["full_report_content"])

            st.download_button(
                label="Download Full Report (.md)",
                data=st.session_state["full_report_content"].encode("utf-8"),
                file_name="Commerce360_Executive_Report.md",
                mime="text/markdown",
                use_container_width=True
            )

    # ── TAB 2: DOMAIN AI INSIGHTS ─────────────────────────────────────────────
    with tab2:
        section_header("Domain-Specific Business Intelligence")

        sub_tab = st.selectbox(
            "Select Intelligence Domain",
            ["Executive Briefing", "Sales & Category Insights", "Customer Intelligence", "Logistics & Risk Insights"]
        )

        if sub_tab == "Executive Briefing":
            st.markdown("#### Executive Briefing")
            with st.spinner("Generating briefing..."):
                exec_text = generate_executive_insights(kpis, monthly_rev, top_cats)
                st.markdown(exec_text)

        elif sub_tab == "Sales & Category Insights":
            st.markdown("#### Sales & Category Intelligence")
            with st.spinner("Analyzing sales trajectory..."):
                sales_text = generate_sales_insights(
                    revenue=kpis["total_revenue"],
                    orders=kpis["total_orders"],
                    top_cats_df=top_cats,
                    monthly_rev_df=monthly_rev,
                    avg_order_value=all_metrics["avg_order_value"]
                )
                st.markdown(sales_text)

        elif sub_tab == "Customer Intelligence":
            st.markdown("#### Customer Segmentation Intelligence")
            with st.spinner("Evaluating customer segments..."):
                cust_text = generate_customer_insights(seg_summary, all_metrics["total_customers"])
                st.markdown(cust_text)

        elif sub_tab == "Logistics & Risk Insights":
            st.markdown("#### Logistics & Delay Risk Intelligence")
            with st.spinner("Assessing delivery networks..."):
                del_text = generate_delivery_insights(del_summary, state_del, ml_results)
                st.markdown(del_text)

    # ── TAB 3: SPECIALIZED ANALYTICS AGENTS ────────────────────────────────────
    with tab3:
        section_header(
            "Specialized Analytics Agents",
            "Autonomous domain agents performing dedicated analytical workloads"
        )

        agent_select = st.radio(
            "Select Analytics Agent",
            ["All Agents (Coordinated)", "Sales Agent", "Customer Agent", "Delivery Agent"],
            horizontal=True
        )

        sales_ag = SalesAgent(sales)
        cust_ag = CustomerAgent(cust_feat, cust_seg)
        del_ag = DeliveryAgent(delivery, ml_results)
        coord_ag = ReportAgent(sales_ag, cust_ag, del_ag)

        if agent_select == "All Agents (Coordinated)":
            coord_res = coord_ag.coordinate_and_report()
            st.markdown("### Master Coordinator Agent Output")
            st.info(coord_res["executive_summary"])

            st.markdown("#### Cross-Functional Strategic Directives")
            for idx, rec in enumerate(coord_res["cross_functional_recommendations"], 1):
                st.markdown(f"**{idx}.** {rec}")

            with st.expander("Inspect Individual Agent Telemetry"):
                st.json(coord_res["domain_findings"])

        elif agent_select == "Sales Agent":
            s_res = sales_ag.analyze()
            st.markdown(f"### {s_res['agent_name']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Revenue Analyzed", f"R$ {s_res['total_revenue']:,.2f}")
            c2.metric("Top Category Share", f"{s_res['top_category_share_pct']}%")
            c3.metric("Revenue Trend", s_res["revenue_trend"])

            st.markdown("#### Agent Findings")
            for ins in s_res["insights"]:
                st.write(f"- {ins}")

        elif agent_select == "Customer Agent":
            c_res = cust_ag.analyze()
            st.markdown(f"### {c_res['agent_name']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Customers Profiled", f"{c_res['total_customers']:,}")
            c2.metric("Repeat Customer Rate", f"{c_res['repeat_customer_rate_pct']}%")
            c3.metric("Top Spending Tier", c_res["high_value_segment"])

            st.markdown("#### Agent Findings")
            for ins in c_res["insights"]:
                st.write(f"- {ins}")

        elif agent_select == "Delivery Agent":
            d_res = del_ag.analyze()
            st.markdown(f"### {d_res['agent_name']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("On-Time Rate", f"{d_res['on_time_rate_pct']}%")
            c2.metric("Delayed Volume", f"{d_res['delayed_orders']:,} orders")
            c3.metric("ML Delay AUC", f"{d_res['ml_delay_model_roc_auc']}")

            st.markdown("#### Agent Findings")
            for ins in d_res["insights"]:
                st.write(f"- {ins}")

    # ── TAB 4: STRATEGIC RECOMMENDATIONS ──────────────────────────────────────
    with tab4:
        section_header("Prioritized Strategic Recommendations")
        with st.spinner("Generating strategic recommendations..."):
            recs_text = generate_recommendations(all_metrics)
            st.markdown(recs_text)

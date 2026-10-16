"""
app.py — Commerce360 AI Flask Web Application Entry Point
=========================================================
Enterprise E-Commerce Analytics & Decision Intelligence System
Built on the Olist Brazilian E-Commerce Public Dataset.

Run with:
    python app.py
"""

from flask import Flask, render_template, request, Response, jsonify
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import utils.data_loader as dl
import utils.calculations as calc
from utils.helpers import fmt_currency, fmt_number, fmt_pct
from components.charts import (
    revenue_trend_chart, orders_trend_chart,
    revenue_by_category_chart, top_products_chart,
    segment_size_chart, segment_spending_chart, segment_scatter_chart, rfm_bar_chart,
    top_sellers_revenue_chart, seller_revenue_distribution_chart,
    delivery_status_pie, delivery_days_histogram, delivery_delay_distribution,
    delivery_by_state_chart, delivery_trend_chart,
    review_score_distribution, review_by_delivery_chart, monthly_review_trend_chart,
    payment_type_pie, payment_value_by_type_chart, installment_chart,
    ml_metrics_bar, feature_importance_chart, figure_to_json
)
import plotly.express as px
from genai.insights import (
    generate_executive_insights, generate_sales_insights,
    generate_customer_insights, generate_delivery_insights,
    generate_recommendations, build_all_metrics, _get_model
)
from genai.report_generator import generate_automated_report, build_deterministic_report
from agents.sales_agent import SalesAgent
from agents.customer_agent import CustomerAgent
from agents.delivery_agent import DeliveryAgent
from agents.report_agent import ReportAgent

app = Flask(__name__)

# Feature importances from verified leakage-free Random Forest
RF_FEATURE_IMPORTANCES = {
    "total_freight": 0.137978,
    "avg_volume_cm3": 0.135172,
    "freight_ratio": 0.133983,
    "estimated_delivery_days": 0.129400,
    "purchase_month": 0.124781,
    "total_price": 0.123273,
    "avg_weight_g": 0.121140,
    "purchase_dayofweek": 0.067178,
    "same_state": 0.018007,
    "num_items": 0.009088,
}


# ── Custom Jinja Template Filters ─────────────────────────────────────────────

@app.template_filter('format_currency')
def filter_currency(val):
    if val is None or pd.isna(val):
        return "0.00"
    return f"{float(val):,.2f}"


@app.template_filter('format_int')
def filter_int(val):
    if val is None or pd.isna(val):
        return "0"
    return f"{int(val):,}"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
@app.route('/overview')
def index():
    sales = dl.load_sales_features()
    delivery = dl.load_delivery_features()
    cust_ft = dl.load_customer_features()
    sell_ft = dl.load_seller_features()

    kpis = calc.get_executive_kpis(sales, delivery, cust_ft, sell_ft)
    monthly_rev = calc.get_monthly_revenue(sales)
    monthly_orders = calc.get_monthly_orders(sales)
    top_cats = calc.get_revenue_by_category(sales, top_n=10)
    del_summary = calc.get_delivery_summary(delivery)

    chart_rev_trend = figure_to_json(revenue_trend_chart(monthly_rev))
    chart_orders_trend = figure_to_json(orders_trend_chart(monthly_orders))
    chart_cats = figure_to_json(revenue_by_category_chart(top_cats))
    chart_del_pie = figure_to_json(delivery_status_pie(del_summary))
    chart_del_hist = figure_to_json(delivery_days_histogram(delivery))

    return render_template(
        'index.html',
        active_page='overview',
        kpis=kpis,
        chart_rev_trend=chart_rev_trend,
        chart_orders_trend=chart_orders_trend,
        chart_cats=chart_cats,
        chart_del_pie=chart_del_pie,
        chart_del_hist=chart_del_hist,
    )


@app.route('/sales')
def sales():
    sales_df = dl.load_sales_features()
    prod_ft = dl.load_product_features()
    prod_tbl = dl.load_product_table()

    all_categories = sorted(sales_df["product_category_name_english"].dropna().unique().tolist())
    selected_cat = request.args.get('category', '').strip()
    top_n = int(request.args.get('top_n', 15))

    filtered_sales = sales_df
    if selected_cat and selected_cat in all_categories:
        filtered_sales = sales_df[sales_df["product_category_name_english"] == selected_cat]

    total_revenue = float(filtered_sales["item_revenue"].sum())
    total_orders = int(filtered_sales["order_id"].nunique())
    aov = total_revenue / total_orders if total_orders > 0 else 0.0
    total_items = int(len(filtered_sales))

    top_cats = calc.get_revenue_by_category(filtered_sales, top_n=top_n)
    top_prods = calc.get_top_products(prod_ft, prod_tbl, top_n=top_n)
    state_rev = calc.get_sales_by_state(filtered_sales)

    # State chart
    fig_state = px.bar(
        state_rev.head(15), x="seller_state", y="revenue",
        title="Revenue by Seller State (Top 15)",
        labels={"seller_state": "State", "revenue": "Revenue (R$)"},
        template="plotly_white", color="revenue",
        color_continuous_scale=[[0, "#BDC3C7"], [1, "#1E3A5F"]],
    )
    fig_state.update_layout(coloraxis_showscale=False)

    # Price dist chart
    price_sample = filtered_sales[filtered_sales["price"] < filtered_sales["price"].quantile(0.99)]
    fig_price = px.histogram(
        price_sample, x="price", nbins=40,
        title="Product Price Distribution", labels={"price": "Price (R$)"},
        template="plotly_white", color_discrete_sequence=["#1E3A5F"]
    )

    chart_cats = figure_to_json(revenue_by_category_chart(top_cats))
    chart_prods = figure_to_json(top_products_chart(top_prods))
    chart_state = figure_to_json(fig_state)
    chart_price_dist = figure_to_json(fig_price)

    return render_template(
        'sales.html',
        active_page='sales',
        total_revenue=total_revenue,
        total_orders=total_orders,
        aov=aov,
        total_items=total_items,
        all_categories=all_categories,
        selected_category=selected_cat,
        top_n=top_n,
        chart_cats=chart_cats,
        chart_prods=chart_prods,
        chart_state=chart_state,
        chart_price_dist=chart_price_dist,
        top_products_table=top_prods.to_dict('records')
    )


@app.route('/customers')
def customers():
    cust_ft = dl.load_customer_features()
    cust_seg = dl.load_customer_segments()

    segs_labeled = calc.label_segments(cust_seg, cust_ft)
    summary = calc.get_segment_summary(cust_seg, cust_ft)

    chart_seg_size = figure_to_json(segment_size_chart(summary))
    chart_seg_spend = figure_to_json(segment_spending_chart(summary))
    chart_seg_scatter = figure_to_json(segment_scatter_chart(segs_labeled))
    chart_rfm = figure_to_json(rfm_bar_chart(summary))

    return render_template(
        'customers.html',
        active_page='customers',
        summary_list=summary.to_dict('records'),
        chart_seg_size=chart_seg_size,
        chart_seg_spend=chart_seg_spend,
        chart_seg_scatter=chart_seg_scatter,
        chart_rfm=chart_rfm
    )


@app.route('/sellers')
def sellers():
    sell_ft = dl.load_seller_features()
    sell_clean = dl.load_sellers_cleaned()

    total_sellers = int(sell_ft["seller_id"].nunique())
    total_revenue = float(sell_ft["seller_revenue"].sum())
    avg_revenue = float(sell_ft["seller_revenue"].mean())
    avg_item_price = float(sell_ft["average_item_price"].mean())

    top_sellers = calc.get_top_sellers(sell_ft, sell_clean, top_n=15)
    state_rev = calc.get_revenue_by_seller_state(sell_ft, sell_clean)

    # State chart
    fig_state = px.bar(
        state_rev.head(15).sort_values("revenue", ascending=False),
        x="seller_state", y="revenue",
        title="Revenue by Seller State (Top 15)",
        labels={"seller_state": "State", "revenue": "Revenue (R$)"},
        template="plotly_white", color_discrete_sequence=["#1E3A5F"],
    )

    # Top sellers by orders
    merged_sellers = sell_ft.merge(sell_clean[["seller_id", "seller_state"]], on="seller_id", how="left")
    top_by_orders = merged_sellers.sort_values("seller_orders", ascending=False).head(15).copy()
    top_by_orders["short_id"] = top_by_orders["seller_id"].str[:8] + "..."

    fig_orders = px.bar(
        top_by_orders.sort_values("seller_orders"),
        x="seller_orders", y="short_id", orientation="h",
        title="Top 15 Sellers by Order Count",
        labels={"seller_orders": "Orders", "short_id": "Seller"},
        color="seller_state", template="plotly_white",
    )

    chart_sellers_rev = figure_to_json(top_sellers_revenue_chart(top_sellers))
    chart_sellers_orders = figure_to_json(fig_orders)
    chart_state_rev = figure_to_json(fig_state)
    chart_rev_dist = figure_to_json(seller_revenue_distribution_chart(sell_ft))

    return render_template(
        'sellers.html',
        active_page='sellers',
        total_sellers=total_sellers,
        total_revenue=total_revenue,
        avg_revenue=avg_revenue,
        avg_item_price=avg_item_price,
        chart_sellers_rev=chart_sellers_rev,
        chart_sellers_orders=chart_sellers_orders,
        chart_state_rev=chart_state_rev,
        chart_rev_dist=chart_rev_dist,
        top_sellers_table=top_sellers.to_dict('records')
    )


@app.route('/delivery')
def delivery():
    del_df = dl.load_delivery_features()
    ml_res = dl.load_ml_results()

    del_summary = calc.get_delivery_summary(del_df)
    state_del = calc.get_delivery_by_state(del_df)
    trend_df = calc.get_delivery_trend(del_df)

    chart_del_pie = figure_to_json(delivery_status_pie(del_summary))
    chart_del_delay = figure_to_json(delivery_delay_distribution(del_df))
    chart_del_state = figure_to_json(delivery_by_state_chart(state_del))
    chart_del_trend = figure_to_json(delivery_trend_chart(trend_df))
    chart_ml_metrics = figure_to_json(ml_metrics_bar(ml_res))
    chart_feat_imp = figure_to_json(feature_importance_chart(RF_FEATURE_IMPORTANCES))

    return render_template(
        'delivery.html',
        active_page='delivery',
        del_summary=del_summary,
        chart_del_pie=chart_del_pie,
        chart_del_delay=chart_del_delay,
        chart_del_state=chart_del_state,
        chart_del_trend=chart_del_trend,
        chart_ml_metrics=chart_ml_metrics,
        chart_feat_imp=chart_feat_imp,
        ml_table=ml_res.to_dict('records')
    )


@app.route('/reviews-payments')
def reviews_payments():
    rev_df = dl.load_review_table()
    pay_df = dl.load_payment_table()

    rev_summary = calc.get_review_summary(rev_df)
    score_by_del = calc.get_score_by_delivery_status(rev_df)
    monthly_rev = calc.get_monthly_review_scores(rev_df)

    pay_summary = calc.get_payment_summary(pay_df)
    pay_by_type = calc.get_payment_by_type(pay_df)
    inst_df = calc.get_installment_distribution(pay_df)

    chart_rev_dist = figure_to_json(review_score_distribution(rev_df))
    chart_rev_del = figure_to_json(review_by_delivery_chart(score_by_del))
    chart_rev_trend = figure_to_json(monthly_review_trend_chart(monthly_rev))

    chart_pay_type = figure_to_json(payment_type_pie(pay_by_type))
    chart_pay_val = figure_to_json(payment_value_by_type_chart(pay_by_type))
    chart_pay_inst = figure_to_json(installment_chart(inst_df))

    return render_template(
        'reviews_payments.html',
        active_page='reviews_payments',
        rev_summary=rev_summary,
        pay_summary=pay_summary,
        chart_rev_dist=chart_rev_dist,
        chart_rev_del=chart_rev_del,
        chart_rev_trend=chart_rev_trend,
        chart_pay_type=chart_pay_type,
        chart_pay_val=chart_pay_val,
        chart_pay_inst=chart_pay_inst
    )


@app.route('/ml-insights')
def ml_insights():
    ml_res = dl.load_ml_results()
    cust_seg = dl.load_customer_segments()
    cust_ft = dl.load_customer_features()

    segs_labeled = calc.label_segments(cust_seg, cust_ft)
    summary = calc.get_segment_summary(cust_seg, cust_ft)

    chart_ml_metrics = figure_to_json(ml_metrics_bar(ml_res))
    chart_feat_imp = figure_to_json(feature_importance_chart(RF_FEATURE_IMPORTANCES))
    chart_seg_scatter = figure_to_json(segment_scatter_chart(segs_labeled))

    return render_template(
        'ml_insights.html',
        active_page='ml_insights',
        ml_results=ml_res.to_dict('records'),
        summary_list=summary.to_dict('records'),
        chart_ml_metrics=chart_ml_metrics,
        chart_feat_imp=chart_feat_imp,
        chart_seg_scatter=chart_seg_scatter
    )


@app.route('/ai-insights')
def ai_insights():
    sales = dl.load_sales_features()
    cust_ft = dl.load_customer_features()
    cust_seg = dl.load_customer_segments()
    sell_ft = dl.load_seller_features()
    del_df = dl.load_delivery_features()
    rev_df = dl.load_review_table()
    pay_df = dl.load_payment_table()
    ml_res = dl.load_ml_results()

    kpis = calc.get_executive_kpis(sales, del_df, cust_ft, sell_ft)
    monthly_rev = calc.get_monthly_revenue(sales)
    top_cats = calc.get_revenue_by_category(sales, top_n=10)
    seg_summary = calc.get_segment_summary(cust_seg, cust_ft)
    del_summary = calc.get_delivery_summary(del_df)
    state_del = calc.get_delivery_by_state(del_df)
    rev_summary = calc.get_review_summary(rev_df)
    pay_summary = calc.get_payment_summary(pay_df)

    all_metrics = build_all_metrics(
        kpis=kpis, top_cats_df=top_cats, segment_summary_df=seg_summary,
        delivery_summary=del_summary, ml_results_df=ml_res,
        rev_summary=rev_summary, pay_summary=pay_summary
    )

    # Agents instantiation & analysis
    sales_ag = SalesAgent(sales)
    cust_ag = CustomerAgent(cust_ft, cust_seg)
    del_ag = DeliveryAgent(del_df, ml_res)
    coord_ag = ReportAgent(sales_ag, cust_ag, del_ag)

    sales_res = sales_ag.analyze()
    cust_res = cust_ag.analyze()
    del_res = del_ag.analyze()
    coord_res = coord_ag.coordinate_and_report()

    # Domain Briefings
    exec_briefing = generate_executive_insights(kpis, monthly_rev, top_cats)
    sales_briefing = generate_sales_insights(
        kpis["total_revenue"], kpis["total_orders"], top_cats, monthly_rev, all_metrics["avg_order_value"]
    )
    del_briefing = generate_delivery_insights(del_summary, state_del, ml_res)
    recs_text = generate_recommendations(all_metrics)

    has_api_key = _get_model() is not None

    return render_template(
        'ai_insights.html',
        active_page='ai_insights',
        has_api_key=has_api_key,
        sales_res=sales_res,
        cust_res=cust_res,
        del_res=del_res,
        coord_res=coord_res,
        exec_briefing=exec_briefing,
        sales_briefing=sales_briefing,
        del_briefing=del_briefing,
        recs_text=recs_text
    )


@app.route('/report')
def report():
    sales = dl.load_sales_features()
    cust_ft = dl.load_customer_features()
    cust_seg = dl.load_customer_segments()
    sell_ft = dl.load_seller_features()
    del_df = dl.load_delivery_features()
    rev_df = dl.load_review_table()
    pay_df = dl.load_payment_table()
    ml_res = dl.load_ml_results()
    prod_ft = dl.load_product_features()
    prod_tbl = dl.load_product_table()

    kpis = calc.get_executive_kpis(sales, del_df, cust_ft, sell_ft)
    top_cats = calc.get_revenue_by_category(sales, top_n=10)
    top_prods = calc.get_top_products(prod_ft, prod_tbl, top_n=10)
    seg_summary = calc.get_segment_summary(cust_seg, cust_ft)
    del_summary = calc.get_delivery_summary(del_df)
    rev_summary = calc.get_review_summary(rev_df)
    pay_summary = calc.get_payment_summary(pay_df)

    all_metrics = build_all_metrics(
        kpis=kpis, top_cats_df=top_cats, segment_summary_df=seg_summary,
        delivery_summary=del_summary, ml_results_df=ml_res,
        rev_summary=rev_summary, pay_summary=pay_summary
    )

    report_md = build_deterministic_report(
        metrics=all_metrics, top_cats=top_cats, top_prods=top_prods,
        seg_summary=seg_summary, ml_results=ml_res
    )

    # Convert markdown to basic HTML paragraphs/headings for viewing
    formatted_html = report_md.replace("\n# ", "\n<h1>").replace("\n## ", "\n<h3 class='mt-4 mb-2 text-primary'>").replace("\n### ", "\n<h5>")
    formatted_html = formatted_html.replace("\n- ", "\n<li>").replace("\n\n", "<br><br>")

    return render_template(
        'report.html',
        active_page='report',
        report_html=formatted_html
    )


@app.route('/download-report')
def download_report():
    sales = dl.load_sales_features()
    cust_ft = dl.load_customer_features()
    cust_seg = dl.load_customer_segments()
    sell_ft = dl.load_seller_features()
    del_df = dl.load_delivery_features()
    rev_df = dl.load_review_table()
    pay_df = dl.load_payment_table()
    ml_res = dl.load_ml_results()
    prod_ft = dl.load_product_features()
    prod_tbl = dl.load_product_table()

    kpis = calc.get_executive_kpis(sales, del_df, cust_ft, sell_ft)
    top_cats = calc.get_revenue_by_category(sales, top_n=10)
    top_prods = calc.get_top_products(prod_ft, prod_tbl, top_n=10)
    seg_summary = calc.get_segment_summary(cust_seg, cust_ft)
    del_summary = calc.get_delivery_summary(del_df)
    rev_summary = calc.get_review_summary(rev_df)
    pay_summary = calc.get_payment_summary(pay_df)

    all_metrics = build_all_metrics(
        kpis=kpis, top_cats_df=top_cats, segment_summary_df=seg_summary,
        delivery_summary=del_summary, ml_results_df=ml_res,
        rev_summary=rev_summary, pay_summary=pay_summary
    )

    report_content = build_deterministic_report(
        metrics=all_metrics, top_cats=top_cats, top_prods=top_prods,
        seg_summary=seg_summary, ml_results=ml_res
    )

    return Response(
        report_content,
        mimetype="text/markdown",
        headers={"Content-Disposition": "attachment;filename=Commerce360_Executive_Report.md"}
    )


if __name__ == '__main__':
    print("=" * 60)
    print("COMMERCE360 AI — Python Flask Web Application")
    print("Serving on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)

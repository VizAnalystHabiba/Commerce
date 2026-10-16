# Commerce360 AI — E-Commerce Analytics & Intelligence System

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly.js-2.35-blue.svg)](https://plotly.com/javascript/)
[![Gemini](https://img.shields.io/badge/GenAI-Google%20Gemini-8E44AD.svg)](https://ai.google.dev/)

An end-to-end enterprise e-commerce decision intelligence system built on the **Olist Brazilian E-Commerce Public Dataset**.

Commerce360 AI transforms raw transactional data into actionable commercial insights by unifying **Data Engineering, Exploratory Analytics, Leakage-Free Machine Learning, Python Flask Web Architecture, Autonomous Analytics Agents, and GenAI Executive Reporting**.

---

## 1. Executive Summary & Architecture

```
Raw Olist Data (9 Relational CSVs)
              ↓
Data Cleaning & Entity Integration (Avoiding Row Multiplication)
              ↓
Domain Feature Engineering (RFM, Fulfillment Margins, Seasonality)
              ↓
Exploratory Data Analysis (EDA)
              ↓
Machine Learning Layer
  ├── Model 1: Customer Segmentation (K-Means, k=4, Silhouette: 0.3958)
  └── Model 2: Delivery Delay Prediction (Leakage-Free, ROC-AUC: 0.7214)
              ↓
★ PYTHON FLASK WEB APPLICATION (Primary BI Platform) ★
  ├── /overview          - Executive KPI Scorecards & Velocity
  ├── /sales             - Revenue Streams & Category Concentration
  ├── /customers         - Behavioral Segments & RFM Profiles
  ├── /sellers           - Merchant Volume & Freight Dynamics
  ├── /delivery          - Fulfillment Bottlenecks & ML Delays
  ├── /reviews-payments  - CSAT Drivers & Payment Rails
  ├── /ml-insights       - In-Depth Model Evaluation & Feature Importances
  └── /ai-insights       - GenAI Engine & Autonomous Agents
              ↓
Specialized Analytics Agents (Sales, Customer, Delivery, Coordinator)
              ↓
Data-Grounded GenAI Insights & Automated Executive Reporting (/report)
```

---

## 2. Business Objectives & Analytical Questions

Commerce360 AI addresses fundamental e-commerce business questions:
1. **Sales & Trajectory:** What is the total business scale, monthly growth trajectory, and average order value?
2. **Product Intelligence:** Which product categories drive the highest revenue versus unit volume?
3. **Customer Segmentation:** How can buyers be grouped based on purchase behavior, frequency, and monetary value?
4. **Seller Performance:** What is the geographic distribution of top merchants, and how do freight charges vary?
5. **Logistics Bottlenecks:** Where are shipping delays occurring, and how do they correlate with customer review scores?
6. **Machine Learning:** Can we predict whether an order will be delayed *at checkout* using only pre-delivery features?
7. **Executive Decision Support:** What concrete, data-backed operational actions should management take?

---

## 3. Dataset Architecture & Entity Modeling

Based on the 9 relational tables from Olist:
1. `olist_orders_dataset.csv`
2. `olist_order_items_dataset.csv`
3. `olist_customers_dataset.csv`
4. `olist_products_dataset.csv`
5. `olist_sellers_dataset.csv`
6. `olist_order_payments_dataset.csv`
7. `olist_order_reviews_dataset.csv`
8. `olist_geolocation_dataset.csv`
9. `product_category_name_translation.csv`

### Architectural Rule: Preventing Data Leakage & Row Inflation
Blindly joining all tables causes **multi-item order multiplication**, leading to artificially inflated order counts and duplicated customer revenue. Commerce360 AI creates decoupled analytical tables:
- `sales_features.csv` (Item-level transactional details)
- `customer_features.csv` (Customer-level aggregated RFM metrics)
- `delivery_features.csv` (Order-level fulfillment metrics)
- `seller_features.csv` (Merchant-level revenue and freight metrics)
- `review_features.csv` & `payment_features.csv`

---

## 4. Machine Learning Implementation

### 1. Customer Segmentation (K-Means Clustering)
- **Features Used:** `total_orders`, `total_spending`, `average_order_value`, `total_items`, `customer_recency_days`
- **Normalization:** `StandardScaler`
- **Optimal Clusters:** `k = 4` (Silhouette Score = `0.3958` on 5,000 customer validation sample)
- **Data-Grounded Cluster Profiles:**
  - **High-Value Customers:** Highest median expenditure, priority retention target.
  - **Frequent Customers:** Repeat purchasers with steady volume.
  - **Low-Engagement Customers:** Moderate-to-low single purchase buyers (majority volume).
  - **At-Risk Customers:** Dormant accounts requiring win-back promotions.

### 2. Delivery Delay Prediction (Strict Leakage Prevention)
- **Target:** `is_delayed` (1 = Delivered after estimated delivery date, 0 = On Time)
- **Data Leakage Fix:** Previous academic models incorrectly used post-delivery features (`delivery_days`, `delivery_delay_days`, `order_delivered_customer_date`). Commerce360 AI uses **only pre-delivery features** known at the time of purchase:
  - `total_price`
  - `total_freight`
  - `num_items`
  - `estimated_delivery_days`
  - `purchase_month` & `purchase_dayofweek`
  - `same_state` (Customer state == Seller state)
  - `freight_ratio` (Freight / Price)
  - `avg_weight_g` & `avg_volume_cm3`
- **Class Balancing:** `class_weight='balanced'` handles ~93% / 7% class imbalance.
- **Evaluation Results:**
  - **Random Forest:** `ROC-AUC: 0.7214` | `Accuracy: 0.9188` | `Precision: 0.2913`
  - **Logistic Regression:** `ROC-AUC: 0.6430` | `Recall: 0.6453` (High sensitivity for risk flagging)

---

## 5. Web Application Architecture (Flask + Plotly.js)

The user-facing portal is built using **Python Flask**, with server-side rendered Jinja2 templates, Bootstrap 5 UI, and interactive Plotly.js visualizations:

- **`/overview` (Executive Overview):** 6 Top-level KPI scorecards, monthly revenue trajectories, category revenue bar charts, and delivery duration distributions.
- **`/sales` (Sales & Product Analytics):** Interactive category slice-and-dice, top catalog products, seller state origin breakdown, and unit price distribution.
- **`/customers` (Customer Analytics):** Cluster breakdown, expenditure comparisons, interactive recency vs. spending scatter map, and actionable retention directives.
- **`/sellers` (Seller Analytics):** Top merchants by revenue and order count, regional concentration, and freight impact.
- **`/delivery` (Delivery Intelligence):** Separate historical performance from pre-delivery ML delay classification and feature importances.
- **`/reviews-payments` (Reviews & Payments):** Customer satisfaction ratings, proof of delivery speed impact on CSAT, and installment payment adoption.
- **`/ml-insights` (ML Insights):** In-depth technical evaluation of both models, confusion matrices, and silhouette metrics.
- **`/ai-insights` (AI Business Insights & Agents):** Live execution of Sales, Customer, Delivery, and Coordinator agents, plus domain executive briefings.
- **`/report` & `/download-report`:** 10-section structured executive report with one-click Markdown download.

---

## 6. Specialized Analytics Agents

Commerce360 AI introduces 4 purpose-built analytical agents in `agents/`:
- **Sales Analytics Agent:** Monitors revenue trends, AOV, category concentration risk, and seasonal trajectory.
- **Customer Intelligence Agent:** Profiles customer lifetime spending, recency distributions, and segment transitions.
- **Delivery Operations Agent:** Analyzes transit durations, regional bottlenecks, and interfaces with the ML delay classifier.
- **Report Coordinator Agent:** Synthesizes findings across agents to formulate executive strategic directives.

---

## 7. Project Structure

```text
Commerce360/
│
├── app.py                         # Master Flask Web Application
├── requirements.txt               # Pinned Python Dependencies
├── README.md                      # Complete Project Documentation
│
├── config/
│   └── config.py                  # Central Paths, Themes & Constants
│
├── data/
│   ├── raw/                       # 9 Original Olist CSV Datasets
│   └── processed/                 # Cleaned & Engineered Analytical Datasets
│
├── templates/                     # Jinja2 HTML Templates
│   ├── base.html                  # Master Layout (Sidebar, Navbar, CDNs)
│   ├── index.html                 # Page 1: Executive Overview
│   ├── sales.html                 # Page 2: Sales & Product Analytics
│   ├── customers.html             # Page 3: Customer Analytics & Segmentation
│   ├── sellers.html               # Page 4: Seller Marketplace Analytics
│   ├── delivery.html              # Page 5: Delivery Intelligence & ML
│   ├── reviews_payments.html      # Page 6: CSAT & Payment Behavior
│   ├── ml_insights.html           # Page 7: In-Depth ML Evaluation
│   ├── ai_insights.html           # Page 8: AI Insights & Analytics Agents
│   └── report.html                # Page 9: 10-Section Consolidated Report
│
├── static/
│   ├── css/
│   │   └── style.css              # Custom Modern Dashboard CSS
│   └── js/
│       └── dashboard.js           # Client-Side Interactivity & Chart Resizing
│
├── components/
│   └── charts.py                  # Plotly Visualizations Library (25+ charts)
│
├── utils/
│   ├── data_loader.py             # High-Speed In-Memory Cached Data Loader
│   ├── calculations.py            # Business Logic & Metric Computations
│   └── helpers.py                 # Formatting, Exporters & Math Helpers
│
├── ml/
│   ├── delivery_model.py          # Leakage-Free Delay Classification
│   └── customer_model.py          # K-Means Segmentation Pipeline
│
├── genai/
│   ├── insights.py                # Gemini Prompt Execution Engine
│   ├── prompts.py                 # Data-Grounded Prompt Builders
│   ├── insight_generator.py       # Insight Wrapper Interface
│   └── report_generator.py        # 10-Section Automated Report Generator
│
├── agents/
│   ├── sales_agent.py             # Dedicated Sales Agent
│   ├── customer_agent.py          # Dedicated Customer Agent
│   ├── delivery_agent.py          # Dedicated Logistics Agent
│   └── report_agent.py            # Master Coordinator Agent
│
├── notebooks/                     # Exploratory & Preparation Notebooks
│   ├── orders1_prep.ipynb ...
│   ├── integration10.ipynb
│   ├── feature_engineering11.ipynb
│   ├── eda12.ipynb
│   └── ml13.ipynb
│
└── src/                           # Core Data Pipeline Modules
    ├── preprocessing.py
    ├── integration.py
    └── feature_engineering.py
```

---

## 8. Installation & Setup

### 1. Activate Environment
In Windows Command Prompt or PowerShell:
```powershell
cd E:\Commerce360
.\venv\Scripts\activate
```

Or in Git Bash:
```bash
cd /e/Commerce360
source venv/Scripts/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Optional: Configure Google Gemini API Key
To enable live GenAI synthesis, create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```
*(Note: If no API key is provided, the application runs in deterministic analytical mode with full reports and metrics fully accessible).*

---

## 9. Running the Application

Launch the Flask web application:
```powershell
python app.py
```

The application will start on:
**`http://127.0.0.1:5000`**

Open your browser and navigate to:
- `http://127.0.0.1:5000/` — Executive Overview
- `http://127.0.0.1:5000/sales` — Sales & Products
- `http://127.0.0.1:5000/customers` — Customer Analytics
- `http://127.0.0.1:5000/sellers` — Seller Analytics
- `http://127.0.0.1:5000/delivery` — Delivery Intelligence
- `http://127.0.0.1:5000/reviews-payments` — Reviews & Payments
- `http://127.0.0.1:5000/ml-insights` — Machine Learning Insights
- `http://127.0.0.1:5000/ai-insights` — AI Insights & Analytics Agents
- `http://127.0.0.1:5000/report` — Executive Report View
- `http://127.0.0.1:5000/download-report` — Download Report (.md)

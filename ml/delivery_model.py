"""
Commerce360 AI — Delivery Delay Prediction Model
=================================================
Purpose:
    Predict whether a delivered order was delayed, using ONLY features
    that would have been known at the time of purchase (pre-delivery).

Target:
    is_delayed  (1 = delayed, 0 = on time)
    Derived from: delivery_delay_days > 0

Leakage-free features (all known at purchase/dispatch time):
    - total_price             : sum of item prices in the order
    - total_freight           : sum of freight values
    - num_items               : number of items ordered
    - estimated_delivery_days : days between purchase date and estimated delivery date
    - purchase_month          : month when the order was placed (seasonality)
    - purchase_dayofweek      : day of week (0=Monday … 6=Sunday)
    - same_state              : 1 if customer and seller are in the same state
    - freight_ratio           : freight / (price + epsilon)  — shipping cost burden
    - product_weight_g        : average product weight (proxy for shipping difficulty)
    - product_volume_cm3      : average product volume in cm³

EXPLICITLY EXCLUDED (post-delivery / leakage):
    - delivery_days
    - delivery_delay_days
    - delivery_status
    - order_delivered_customer_date
    - order_delivered_carrier_date

Source files (all verified):
    data/processed/delivery_features.csv   — order + customer + seller + item data
    data/processed/sales_features.csv      — includes product weight/dimensions
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------------------

print("=" * 60)
print("COMMERCE360 AI — Delivery Delay Prediction (Leakage-Free)")
print("=" * 60)

# Resolve the data directory relative to THIS script's location
# Works regardless of which directory you run the script from
SCRIPT_DIR = Path(__file__).resolve().parent        # e:/Commerce360/ml
PROJECT_DIR = SCRIPT_DIR.parent                     # e:/Commerce360
PROCESSED   = PROJECT_DIR / "data" / "processed"

print(f"\nProject root : {PROJECT_DIR}")
print(f"Data folder  : {PROCESSED}")

delivery = pd.read_csv(PROCESSED / "delivery_features.csv")
sales    = pd.read_csv(PROCESSED / "sales_features.csv")

print(f"\ndelivery_features loaded : {delivery.shape}")
print(f"sales_features loaded    : {sales.shape}")

# ------------------------------------------------------------------------------
# 2. FILTER TO DELIVERED ORDERS ONLY
#    Cancelled / unavailable orders never reached delivery — exclude them
#    so the target is meaningful.
# ------------------------------------------------------------------------------

delivery = delivery[delivery["order_status"] == "delivered"].copy()
print(f"\nAfter filtering to 'delivered' orders: {delivery.shape}")

# ------------------------------------------------------------------------------
# 3. BUILD ORDER-LEVEL AGGREGATES FROM SALES FEATURES
#    sales_features is at item level (one row per order item).
#    We aggregate to order level to avoid row multiplication.
# ------------------------------------------------------------------------------

order_agg = (
    sales
    .groupby("order_id")
    .agg(
        total_price       = ("price",              "sum"),
        total_freight     = ("freight_value",      "sum"),
        num_items         = ("order_item_id",      "count"),
        avg_weight_g      = ("product_weight_g",   "mean"),
        avg_length_cm     = ("product_length_cm",  "mean"),
        avg_height_cm     = ("product_height_cm",  "mean"),
        avg_width_cm      = ("product_width_cm",   "mean"),
    )
    .reset_index()
)

# Derived: approximate volume per order
order_agg["avg_volume_cm3"] = (
    order_agg["avg_length_cm"]
    * order_agg["avg_height_cm"]
    * order_agg["avg_width_cm"]
)

print(f"\nOrder-level aggregation shape: {order_agg.shape}")

# ------------------------------------------------------------------------------
# 4. MERGE — use delivery_features as the base (already deduplicated per order)
#    delivery_features has one row per order-item combination.
#    We need ONE row per order_id. Take first occurrence for location fields,
#    then bring in the aggregated price / freight / weight.
# ------------------------------------------------------------------------------

# Deduplicate delivery_features to one row per order (keep first)
delivery_orders = (
    delivery
    .drop_duplicates(subset="order_id", keep="first")
    .copy()
)
print(f"Unique orders in delivery_features: {delivery_orders.shape[0]}")

# Merge aggregated item stats onto order-level delivery data
df = delivery_orders.merge(order_agg, on="order_id", how="inner")
print(f"After merge with order aggregates: {df.shape}")

# ------------------------------------------------------------------------------
# 5. DEFINE TARGET VARIABLE
#    is_delayed = 1 if the order was delayed, 0 if on time
#    Source: on_time_flag == 0 means delayed
# ------------------------------------------------------------------------------

df["is_delayed"] = (df["on_time_flag"] == 0).astype(int)

print(f"\nTarget distribution (is_delayed):")
vc = df["is_delayed"].value_counts()
print(f"  On Time  (0): {vc.get(0, 0):,}  ({vc.get(0,0)/len(df)*100:.1f}%)")
print(f"  Delayed  (1): {vc.get(1, 0):,}  ({vc.get(1,0)/len(df)*100:.1f}%)")

# ------------------------------------------------------------------------------
# 6. PRE-DELIVERY FEATURE ENGINEERING
#    ALL features below are known at the time the order is placed or
#    at the time of dispatch — no post-delivery information used.
# ------------------------------------------------------------------------------

df["order_purchase_timestamp"]    = pd.to_datetime(df["order_purchase_timestamp"])
df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])

# Days between purchase and estimated delivery (window size)
df["estimated_delivery_days"] = (
    df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
).dt.days

# Purchase seasonality signals
df["purchase_month"]      = df["order_purchase_timestamp"].dt.month
df["purchase_dayofweek"]  = df["order_purchase_timestamp"].dt.dayofweek

# Same-state indicator (shorter distance = lower delay risk hypothesis)
df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)

# Freight burden ratio
df["freight_ratio"] = df["total_freight"] / (df["total_price"] + 1e-6)

# ------------------------------------------------------------------------------
# 7. SELECT FINAL FEATURE SET
# ------------------------------------------------------------------------------

FEATURES = [
    "total_price",
    "total_freight",
    "num_items",
    "estimated_delivery_days",
    "purchase_month",
    "purchase_dayofweek",
    "same_state",
    "freight_ratio",
    "avg_weight_g",
    "avg_volume_cm3",
]

TARGET = "is_delayed"

# Confirm NO post-delivery columns are in features
LEAKAGE_COLS = [
    "delivery_days", "delivery_delay_days", "delivery_status",
    "order_delivered_customer_date", "order_delivered_carrier_date",
    "on_time_flag",
]
for col in LEAKAGE_COLS:
    assert col not in FEATURES, f"LEAKAGE DETECTED: {col} found in feature set!"

print("\nLeakage check PASSED — no post-delivery columns in feature set.")

# Build modelling dataset
model_df = df[FEATURES + [TARGET]].dropna()
print(f"Modelling dataset shape (after dropping NaN): {model_df.shape}")

X = model_df[FEATURES]
y = model_df[TARGET]

# ------------------------------------------------------------------------------
# 8. TRAIN / TEST SPLIT — stratified to preserve class ratio
# ------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")
print(f"Train delayed %: {y_train.mean():.2%}  |  Test delayed %: {y_test.mean():.2%}")

# ------------------------------------------------------------------------------
# 9. SCALING — required for Logistic Regression
# ------------------------------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ------------------------------------------------------------------------------
# 10. MODEL TRAINING & EVALUATION
# ------------------------------------------------------------------------------

def evaluate_model(name, y_test, y_pred, y_proba):
    """Compute and return all evaluation metrics for one model."""
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'-'*50}")
    print(f"Model : {name}")
    print(f"{'-'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (of orders predicted delayed, how many truly were)")
    print(f"  Recall    : {rec:.4f}  (of truly delayed orders, how many we caught)")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    True Neg (On Time predicted On Time) : {cm[0][0]:,}")
    print(f"    False Pos (On Time predicted Delayed): {cm[0][1]:,}")
    print(f"    False Neg (Delayed predicted On Time): {cm[1][0]:,}")
    print(f"    True Pos  (Delayed predicted Delayed): {cm[1][1]:,}")
    print(f"\n  Full Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["On Time", "Delayed"]))

    return {
        "model":     name,
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1_score":  round(f1,   4),
        "roc_auc":   round(auc,  4),
    }


results = []

# -- MODEL A : Logistic Regression ---------------------------------------------
# class_weight='balanced' compensates for the ~9:1 class imbalance
print("\n\n[MODEL A] Training Logistic Regression …")
lr = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)
lr.fit(X_train_scaled, y_train)
y_pred_lr   = lr.predict(X_test_scaled)
y_proba_lr  = lr.predict_proba(X_test_scaled)[:, 1]
results.append(evaluate_model(
    "Logistic Regression (Pre-Delivery)", y_test, y_pred_lr, y_proba_lr
))

# -- MODEL B : Random Forest ---------------------------------------------------
print("\n[MODEL B] Training Random Forest …")
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)          # Tree-based: raw (unscaled) features
y_pred_rf   = rf.predict(X_test)
y_proba_rf  = rf.predict_proba(X_test)[:, 1]
results.append(evaluate_model(
    "Random Forest (Pre-Delivery)", y_test, y_pred_rf, y_proba_rf
))

# -- FEATURE IMPORTANCE (Random Forest) ----------------------------------------
importance_df = (
    pd.DataFrame({
        "feature":   FEATURES,
        "importance": rf.feature_importances_,
    })
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
print("\n\nTop Feature Importances (Random Forest):")
print(importance_df.to_string(index=False))

# ------------------------------------------------------------------------------
# 11. SAVE RESULTS
# ------------------------------------------------------------------------------

results_df = pd.DataFrame(results)
out_path = PROCESSED / "ml_model_results.csv"
results_df.to_csv(out_path, index=False)
print(f"\n\n[OK] Results saved -> {out_path}")
print(results_df.to_string(index=False))

# ------------------------------------------------------------------------------
# 12. INTERPRETATION NOTE
# ------------------------------------------------------------------------------

print("""
-----------------------------------------------------------------------------
INTERPRETATION NOTE
-----------------------------------------------------------------------------
This model predicts BEFORE delivery whether an order is at risk of delay.

The dataset is imbalanced (~9:1 on-time vs delayed).
We use class_weight='balanced' so the model does not simply predict
"on time" for every order to achieve high accuracy.

KEY METRIC for this problem: ROC-AUC (overall discrimination ability)
SECONDARY METRIC: Recall for the Delayed class (do we catch delayed orders?)

A ROC-AUC of 0.65–0.75 is a realistic and academically honest baseline
for this type of tabular prediction without GPS or real-time logistics data.

IMPORTANT DISTINCTION:
  Descriptive Analytics (EDA/Power BI): What happened to past deliveries?
  Predictive ML (this model)          : At order time, is this order at risk?
-----------------------------------------------------------------------------
""")

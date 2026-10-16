"""
ml/customer_model.py — Commerce360 AI Customer Segmentation Model (K-Means)
=============================================================================
Performs behavioral clustering on customer RFM + engagement features:
  - total_orders
  - total_spending
  - average_order_value
  - total_items
  - customer_recency_days

Outputs:
  - data/processed/customer_segments.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PROCESSED = PROJECT_DIR / "data" / "processed"


def train_customer_segmentation():
    print("=" * 60)
    print("COMMERCE360 AI — Customer Segmentation (K-Means)")
    print("=" * 60)

    input_file = PROCESSED / "customer_features.csv"
    if not input_file.exists():
        raise FileNotFoundError(f"Missing {input_file}")

    customers = pd.read_csv(input_file)
    print(f"Loaded customer features: {customers.shape}")

    feature_cols = [
        "total_orders",
        "total_spending",
        "average_order_value",
        "total_items",
        "customer_recency_days"
    ]

    customer_ml = customers[["customer_id"] + feature_cols].dropna().copy()
    print(f"Features for clustering: {feature_cols}")

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_ml[feature_cols])

    # K-Means clustering (k=4)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    customer_ml["segment"] = kmeans.fit_predict(scaled_features)

    # Compute Silhouette Score on sample
    sample_size = min(5000, len(scaled_features))
    sil_score = silhouette_score(
        scaled_features,
        customer_ml["segment"],
        sample_size=sample_size,
        random_state=42
    )
    print(f"\nSilhouette Score (sample={sample_size}): {sil_score:.4f}")

    # Summary by cluster
    summary = (
        customer_ml.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_spending=("total_spending", "mean"),
            avg_orders=("total_orders", "mean"),
            avg_order_value=("average_order_value", "mean"),
            avg_recency=("customer_recency_days", "mean")
        )
        .round(2)
        .reset_index()
    )

    # Rank clusters by median spending for data-grounded business labeling
    median_spend = (
        customer_ml.groupby("segment")["total_spending"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    business_labels = [
        "High-Value Customers",
        "Frequent Customers",
        "Low-Engagement Customers",
        "At-Risk Customers"
    ]

    label_mapping = {
        cluster_id: business_labels[idx]
        for idx, cluster_id in enumerate(median_spend)
    }

    customer_ml["segment_name"] = customer_ml["segment"].map(label_mapping)
    summary["segment_name"] = summary["segment"].map(label_mapping)

    print("\n--- Customer Segments Summary ---")
    print(summary.to_string(index=False))

    output_file = PROCESSED / "customer_segments.csv"
    customer_ml.to_csv(output_file, index=False)
    print(f"\n[OK] Saved customer segments to {output_file}")

    return customer_ml, summary, sil_score


if __name__ == "__main__":
    train_customer_segmentation()

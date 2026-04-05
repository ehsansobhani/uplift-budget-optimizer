"""
Feature engineering pipeline for retail media uplift modeling.

Transforms raw transaction logs + client demographics into customer-level
features suitable for uplift model training. Each feature group is an
independent, testable function that can be cached and computed incrementally.

Feature Groups:
  1. RFM (Recency, Frequency, Monetary)
  2. Category affinity
  3. Temporal behavior
  4. Basket composition
  5. Demographics
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RFM FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def build_rfm_features(purchases: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary features per customer.

    Args:
        purchases: Raw transaction data with columns:
            client_id, transaction_id, transaction_datetime,
            trn_sum_from_iss, trn_sum_from_red

    Returns:
        DataFrame indexed by client_id with RFM features.
    """
    ref_date = purchases["transaction_datetime"].max()

    # Transaction-level aggregation
    trn_level = purchases.groupby(["client_id", "transaction_id"]).agg(
        trn_date=("transaction_datetime", "min"),
        trn_amount=("trn_sum_from_iss", "sum"),
        trn_redeemed=("trn_sum_from_red", "sum"),
        trn_items=("product_quantity", "sum"),
    ).reset_index()

    # Client-level aggregation
    rfm = trn_level.groupby("client_id").agg(
        recency_days=("trn_date", lambda x: (ref_date - x.max()).days),
        frequency_total=("transaction_id", "nunique"),
        monetary_total=("trn_amount", "sum"),
        monetary_avg_basket=("trn_amount", "mean"),
        monetary_std_basket=("trn_amount", "std"),
        monetary_max_basket=("trn_amount", "max"),
        monetary_min_basket=("trn_amount", "min"),
        total_items=("trn_items", "sum"),
        total_redeemed=("trn_redeemed", "sum"),
    ).reset_index()

    # Derived
    rfm["monetary_std_basket"] = rfm["monetary_std_basket"].fillna(0)
    rfm["redemption_ratio"] = np.where(
        rfm["monetary_total"] > 0,
        rfm["total_redeemed"] / rfm["monetary_total"],
        0.0,
    )
    rfm["avg_items_per_basket"] = np.where(
        rfm["frequency_total"] > 0,
        rfm["total_items"] / rfm["frequency_total"],
        0.0,
    )

    # Recent activity windows
    for window_days in [7, 14, 30, 60, 90]:
        cutoff = ref_date - pd.Timedelta(days=window_days)
        window_trns = trn_level[trn_level["trn_date"] >= cutoff]
        window_agg = window_trns.groupby("client_id").agg(
            **{f"frequency_{window_days}d": ("transaction_id", "nunique"),
               f"monetary_{window_days}d": ("trn_amount", "sum")}
        ).reset_index()
        rfm = rfm.merge(window_agg, on="client_id", how="left")
        rfm[f"frequency_{window_days}d"] = rfm[f"frequency_{window_days}d"].fillna(0)
        rfm[f"monetary_{window_days}d"] = rfm[f"monetary_{window_days}d"].fillna(0)

    logger.info("RFM features: %d clients, %d features", len(rfm), len(rfm.columns) - 1)
    return rfm


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CATEGORY AFFINITY FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def build_category_features(
    purchases: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute category affinity features per customer.

    Captures breadth of shopping, concentration, and specific category
    preferences — critical for retail media audience targeting.
    """
    # Join purchases with product categories
    purch_cat = purchases.merge(
        products[["product_id", "department"]],
        on="product_id",
        how="left",
    )

    # Spend per client × department
    spend_by_cat = purch_cat.groupby(["client_id", "department"]).agg(
        cat_spend=("trn_sum_from_iss", "sum"),
        cat_count=("transaction_id", "nunique"),
    ).reset_index()

    # Client-level category metrics
    client_total = spend_by_cat.groupby("client_id")["cat_spend"].sum().rename("total_spend")
    spend_by_cat = spend_by_cat.merge(client_total, on="client_id")
    spend_by_cat["cat_share"] = spend_by_cat["cat_spend"] / spend_by_cat["total_spend"].clip(lower=1e-9)

    cat_features = spend_by_cat.groupby("client_id").agg(
        n_unique_categories=("department", "nunique"),
        top_category_share=("cat_share", "max"),
        category_entropy=("cat_share", lambda x: -np.sum(x * np.log(x.clip(lower=1e-9)))),
        n_cat_trips=("cat_count", "sum"),
    ).reset_index()

    # Top-N category binary flags (for brand targeting)
    top_categories = spend_by_cat.groupby("department")["cat_spend"].sum().nlargest(10).index

    for cat in top_categories:
        cat_clients = spend_by_cat[spend_by_cat["department"] == cat][["client_id", "cat_share"]]
        cat_clients = cat_clients.rename(columns={"cat_share": f"share_{cat.lower().replace(' ', '_')}"})
        cat_features = cat_features.merge(cat_clients, on="client_id", how="left")
        cat_features[f"share_{cat.lower().replace(' ', '_')}"] = (
            cat_features[f"share_{cat.lower().replace(' ', '_')}"].fillna(0)
        )

    logger.info("Category features: %d clients, %d features", len(cat_features), len(cat_features.columns) - 1)
    return cat_features


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TEMPORAL BEHAVIOR FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def build_temporal_features(purchases: pd.DataFrame) -> pd.DataFrame:
    """
    Compute temporal shopping behavior features.

    Captures day-of-week/hour preferences, shopping regularity,
    and spending trends — signals for campaign timing optimization.
    """
    purch = purchases.copy()
    purch["dow"] = purch["transaction_datetime"].dt.dayofweek
    purch["hour"] = purch["transaction_datetime"].dt.hour
    purch["is_weekend"] = purch["dow"].isin([5, 6]).astype(int)
    purch["week"] = purch["transaction_datetime"].dt.isocalendar().week.astype(int)

    # Transaction-level for inter-purchase intervals
    trn_dates = purch.groupby(["client_id", "transaction_id"])["transaction_datetime"].min()
    trn_dates = trn_dates.reset_index().sort_values(["client_id", "transaction_datetime"])

    # Inter-purchase intervals
    trn_dates["prev_date"] = trn_dates.groupby("client_id")["transaction_datetime"].shift(1)
    trn_dates["ipi_days"] = (trn_dates["transaction_datetime"] - trn_dates["prev_date"]).dt.days

    ipi_stats = trn_dates.groupby("client_id")["ipi_days"].agg(
        avg_ipi_days="mean",
        std_ipi_days="std",
        min_ipi_days="min",
        max_ipi_days="max",
    ).reset_index()
    ipi_stats["std_ipi_days"] = ipi_stats["std_ipi_days"].fillna(0)

    # Purchase regularity (CV of inter-purchase intervals; lower = more regular)
    ipi_stats["purchase_regularity"] = np.where(
        ipi_stats["avg_ipi_days"] > 0,
        ipi_stats["std_ipi_days"] / ipi_stats["avg_ipi_days"].clip(lower=1e-9),
        0.0,
    )

    # Day/hour preferences
    temporal = purch.groupby("client_id").agg(
        preferred_dow=("dow", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0),
        preferred_hour=("hour", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12),
        weekend_ratio=("is_weekend", "mean"),
        n_active_weeks=("week", "nunique"),
    ).reset_index()

    # Spending trend (slope of weekly spend)
    weekly_spend = purch.groupby(["client_id", "week"])["trn_sum_from_iss"].sum().reset_index()
    def _trend_slope(group):
        if len(group) < 3:
            return 0.0
        x = np.arange(len(group))
        y = group["trn_sum_from_iss"].values
        slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0.0
        return slope

    trends = weekly_spend.groupby("client_id").apply(_trend_slope, include_groups=False).rename("spend_trend").reset_index()

    # Merge all temporal features
    result = temporal.merge(ipi_stats, on="client_id", how="left")
    result = result.merge(trends, on="client_id", how="left")
    result["spend_trend"] = result["spend_trend"].fillna(0)

    logger.info("Temporal features: %d clients, %d features", len(result), len(result.columns) - 1)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BASKET COMPOSITION FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def build_basket_features(
    purchases: pd.DataFrame,
    products: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute basket-level shopping behavior features.

    Captures basket size, product diversity, store loyalty,
    and express-vs-stock-up shopping patterns.
    """
    # Basket-level aggregation
    basket = purchases.groupby(["client_id", "transaction_id"]).agg(
        n_items=("product_quantity", "sum"),
        n_unique_products=("product_id", "nunique"),
        basket_value=("trn_sum_from_iss", "sum"),
        store_id=("store_id", "first"),
    ).reset_index()

    # Client-level basket metrics
    basket_features = basket.groupby("client_id").agg(
        avg_basket_items=("n_items", "mean"),
        std_basket_items=("n_items", "std"),
        avg_unique_products=("n_unique_products", "mean"),
        avg_basket_value=("basket_value", "mean"),
        n_unique_stores=("store_id", "nunique"),
    ).reset_index()

    basket_features["std_basket_items"] = basket_features["std_basket_items"].fillna(0)

    # Store loyalty: fraction of trips to most-visited store
    store_counts = basket.groupby(["client_id", "store_id"]).size().reset_index(name="visits")
    max_store_visits = store_counts.groupby("client_id")["visits"].max().rename("max_store_visits")
    total_visits = store_counts.groupby("client_id")["visits"].sum().rename("total_visits")
    store_loyalty = (max_store_visits / total_visits).rename("store_loyalty").reset_index()
    basket_features = basket_features.merge(store_loyalty, on="client_id", how="left")

    # Express trip ratio (≤ 5 items = quick run)
    basket["is_express"] = (basket["n_items"] <= 5).astype(int)
    express_ratio = basket.groupby("client_id")["is_express"].mean().rename("express_trip_ratio").reset_index()
    basket_features = basket_features.merge(express_ratio, on="client_id", how="left")

    logger.info("Basket features: %d clients, %d features", len(basket_features), len(basket_features.columns) - 1)
    return basket_features


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DEMOGRAPHIC FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def build_demographic_features(clients: pd.DataFrame) -> pd.DataFrame:
    """
    Process client demographic features for model training.
    """
    demo = clients[["client_id"]].copy()

    # Age bins
    if "age" in clients.columns:
        demo["age"] = clients["age"].fillna(clients["age"].median())
        demo["age_bin"] = pd.cut(
            demo["age"],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=[0, 1, 2, 3, 4, 5],
        ).astype(int)
    else:
        demo["age"] = 40
        demo["age_bin"] = 2

    # Gender encoding
    if "gender" in clients.columns:
        demo["is_female"] = (clients["gender"] == "F").astype(int)
        demo["is_male"] = (clients["gender"] == "M").astype(int)
        demo["gender_unknown"] = (clients["gender"] == "U").astype(int)
    else:
        demo["is_female"] = 0
        demo["is_male"] = 0
        demo["gender_unknown"] = 1

    # Tenure
    if "first_issue_date" in clients.columns:
        ref = clients["first_issue_date"].max()
        demo["tenure_days"] = (ref - clients["first_issue_date"]).dt.days.fillna(0).astype(int)
    else:
        demo["tenure_days"] = 365

    # Location
    if "location_id" in clients.columns:
        demo["location_id"] = clients["location_id"].fillna(-1).astype(int)

    logger.info("Demographic features: %d clients, %d features", len(demo), len(demo.columns) - 1)
    return demo


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_table(
    clients: pd.DataFrame,
    purchases: pd.DataFrame,
    products: pd.DataFrame,
    train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Orchestrate all feature groups into a single training-ready feature table.

    Args:
        clients: Client demographics
        purchases: Transaction history (pre-campaign)
        products: Product catalog
        train: RCT labels (client_id, treatment_flg, target)

    Returns:
        Feature table with one row per client, including treatment and target columns.
    """
    logger.info("Building feature table from %d clients, %d purchases", len(clients), len(purchases))

    # Build each feature group
    rfm = build_rfm_features(purchases)
    category = build_category_features(purchases, products)
    temporal = build_temporal_features(purchases)
    basket = build_basket_features(purchases, products)
    demographics = build_demographic_features(clients)

    # Start from train (defines the population + labels)
    feature_table = train[["client_id", "treatment_flg", "target"]].copy()

    # Merge all feature groups
    for feat_df, name in [
        (rfm, "RFM"),
        (category, "Category"),
        (temporal, "Temporal"),
        (basket, "Basket"),
        (demographics, "Demographics"),
    ]:
        pre_cols = len(feature_table.columns)
        feature_table = feature_table.merge(feat_df, on="client_id", how="left")
        n_new = len(feature_table.columns) - pre_cols
        logger.info("  + %s: %d features", name, n_new)

    # Fill any remaining NaN (clients with no purchase history)
    numeric_cols = feature_table.select_dtypes(include=[np.number]).columns
    feature_table[numeric_cols] = feature_table[numeric_cols].fillna(0)

    logger.info(
        "Feature table ready: %d rows × %d columns "
        "(treatment rate: %.3f, conversion rate: %.3f)",
        len(feature_table),
        len(feature_table.columns),
        feature_table["treatment_flg"].mean(),
        feature_table["target"].mean(),
    )

    return feature_table


def get_feature_columns(feature_table: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excluding IDs, treatment, target)."""
    exclude = {"client_id", "treatment_flg", "target"}
    return [c for c in feature_table.columns if c not in exclude]

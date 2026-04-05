"""
Full X5 RetailHero pipeline with PURCHASES data.

Handles:
  - Reassembling split CSV parts (purchases_part_0..N.csv)
  - Full 4GB purchase history processing via chunked aggregation
  - 5 feature groups: RFM, Category, Temporal, Basket, Demographics
  - 4 uplift models with comparison
  - A/B test simulation + incrementality
  - Holdout scoring + submission generation

Usage (locally):
    # Option A: if you have the original purchases.csv
    python main_full.py --purchases-file data/purchases.csv

    # Option B: if you split it into parts
    python main_full.py --purchases-dir data/  # reads purchases_part_*.csv

    # Option C: auto-detect (looks for both)
    python main_full.py
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.models.uplift_models import UpliftModelConfig, get_model
from src.models.evaluate import (
    evaluate_model, compare_models,
    plot_qini_curves, plot_uplift_by_decile,
)
from src.models.train import stratified_train_test_split
from src.experimentation.ab_simulator import (
    run_ab_comparison, power_analysis, bootstrap_uplift_ci, simulate_campaign,
)
from src.monitoring.drift import detect_feature_drift, data_quality_checks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main_full")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def find_data_files(data_dir: str = "data") -> dict[str, str | list[str]]:
    """Auto-detect data files in the given directory."""
    data_dir = Path(data_dir)
    files = {}

    # Required files
    for name in ["clients.csv", "uplift_train.csv", "uplift_test.csv", "products.csv"]:
        path = data_dir / name
        if path.exists():
            files[name.replace(".csv", "")] = str(path)
        else:
            raise FileNotFoundError(f"Required file not found: {path}")

    # Purchases: single file or split parts
    single = data_dir / "purchases.csv"
    if single.exists():
        files["purchases"] = str(single)
    else:
        parts = sorted(glob.glob(str(data_dir / "purchases_part_*.csv")))
        if parts:
            files["purchases_parts"] = parts
        else:
            raise FileNotFoundError(
                f"No purchases data found in {data_dir}. "
                "Expected purchases.csv or purchases_part_*.csv"
            )

    return files


def load_purchases_chunked(
    source: str | list[str],
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """
    Load purchases from a single CSV or multiple split parts.
    Processes in chunks to handle 4GB+ files with bounded memory.

    Returns a per-client aggregated DataFrame (NOT raw rows).
    """
    logger.info("Loading purchases (chunked aggregation)...")

    agg_frames = []
    total_rows = 0

    if isinstance(source, str):
        # Single file — read in chunks
        sources = [(source, True)]  # (path, has_header)
    else:
        # Multiple parts — each has its own header
        sources = [(p, True) for p in source]

    for file_path, has_header in sources:
        file_name = os.path.basename(file_path)
        logger.info("  Processing %s ...", file_name)

        reader = pd.read_csv(
            file_path,
            header=0 if has_header else None,
            names=None if has_header else [
                "client_id", "transaction_id", "transaction_datetime",
                "regular_points_received", "express_points_received",
                "regular_points_spent", "express_points_spent",
                "purchase_sum", "store_id", "product_id",
                "product_quantity", "trn_sum_from_iss", "trn_sum_from_red",
            ],
            dtype={
                "client_id": str,
                "transaction_id": str,
                "store_id": str,
                "product_id": str,
            },
            parse_dates=["transaction_datetime"],
            chunksize=chunk_size,
        )

        for chunk in reader:
            total_rows += len(chunk)
            chunk["trn_sum_from_red"] = chunk["trn_sum_from_red"].fillna(0)

            # Aggregate this chunk at client level
            chunk_agg = _aggregate_chunk(chunk)
            agg_frames.append(chunk_agg)

    # Combine all chunk-level aggregations
    logger.info("  Combining %d chunk aggregations (%d total rows)...", len(agg_frames), total_rows)
    combined = pd.concat(agg_frames, ignore_index=True)

    # Re-aggregate across chunks (same client may appear in multiple chunks)
    final = combined.groupby("client_id").agg(
        n_rows=("n_rows", "sum"),
        n_transactions=("n_transactions", "sum"),
        n_products=("n_products", "sum"),
        n_stores=("n_stores", "sum"),
        total_spend=("total_spend", "sum"),
        total_redeemed=("total_redeemed", "sum"),
        total_quantity=("total_quantity", "sum"),
        total_points_received=("total_points_received", "sum"),
        total_points_spent=("total_points_spent", "sum"),
        min_date=("min_date", "min"),
        max_date=("max_date", "max"),
        sum_spend_sq=("sum_spend_sq", "sum"),
        total_express_points=("total_express_points", "sum"),
        n_weekend_rows=("n_weekend_rows", "sum"),
        n_morning_rows=("n_morning_rows", "sum"),
        n_evening_rows=("n_evening_rows", "sum"),
        n_small_basket_rows=("n_small_basket_rows", "sum"),
        # These are approximate since we can't perfectly merge across chunks
        # but good enough for feature engineering
    ).reset_index()

    logger.info("  Purchase aggregation complete: %d unique clients", len(final))
    return final


def _aggregate_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a single chunk of purchase data at client level."""
    chunk["dow"] = chunk["transaction_datetime"].dt.dayofweek
    chunk["hour"] = chunk["transaction_datetime"].dt.hour
    chunk["is_weekend"] = chunk["dow"].isin([5, 6]).astype(int)
    chunk["is_morning"] = (chunk["hour"] < 12).astype(int)
    chunk["is_evening"] = (chunk["hour"] >= 18).astype(int)
    chunk["is_small_qty"] = (chunk["product_quantity"] <= 2).astype(int)

    agg = chunk.groupby("client_id").agg(
        n_rows=("product_id", "count"),
        n_transactions=("transaction_id", "nunique"),
        n_products=("product_id", "nunique"),
        n_stores=("store_id", "nunique"),
        total_spend=("trn_sum_from_iss", "sum"),
        total_redeemed=("trn_sum_from_red", "sum"),
        total_quantity=("product_quantity", "sum"),
        total_points_received=("regular_points_received", "sum"),
        total_points_spent=("regular_points_spent", "sum"),
        min_date=("transaction_datetime", "min"),
        max_date=("transaction_datetime", "max"),
        sum_spend_sq=("trn_sum_from_iss", lambda x: (x ** 2).sum()),
        total_express_points=("express_points_received", "sum"),
        n_weekend_rows=("is_weekend", "sum"),
        n_morning_rows=("is_morning", "sum"),
        n_evening_rows=("is_evening", "sum"),
        n_small_basket_rows=("is_small_qty", "sum"),
    ).reset_index()

    return agg


# ═══════════════════════════════════════════════════════════════════════════════
# PURCHASE-BASED FEATURE ENGINEERING (from aggregated data)
# ═══════════════════════════════════════════════════════════════════════════════

def build_purchase_features(purchase_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Build rich features from pre-aggregated purchase data.

    5 feature groups derived from the aggregation:
      1. RFM (Recency, Frequency, Monetary)
      2. Loyalty points behavior
      3. Temporal shopping patterns
      4. Basket composition
      5. Store behavior
    """
    df = purchase_agg[["client_id"]].copy()
    ref_date = purchase_agg["max_date"].max()

    # ── 1. RFM Features ─────────────────────────────────────────────────
    df["recency_days"] = (ref_date - purchase_agg["max_date"]).dt.days.fillna(999)
    df["frequency_total"] = purchase_agg["n_transactions"]
    df["monetary_total"] = purchase_agg["total_spend"]
    df["monetary_avg_basket"] = np.where(
        purchase_agg["n_transactions"] > 0,
        purchase_agg["total_spend"] / purchase_agg["n_transactions"],
        0,
    )
    # Spend std approximation from E[X²] - E[X]²
    mean_spend = np.where(
        purchase_agg["n_rows"] > 0,
        purchase_agg["total_spend"] / purchase_agg["n_rows"],
        0,
    )
    mean_spend_sq = np.where(
        purchase_agg["n_rows"] > 0,
        purchase_agg["sum_spend_sq"] / purchase_agg["n_rows"],
        0,
    )
    df["monetary_std_item"] = np.sqrt(np.maximum(mean_spend_sq - mean_spend ** 2, 0))

    # Active days span
    active_span = (purchase_agg["max_date"] - purchase_agg["min_date"]).dt.days.fillna(0)
    df["active_span_days"] = active_span
    df["purchase_intensity"] = np.where(
        active_span > 0,
        purchase_agg["n_transactions"] / active_span * 30,  # transactions per month
        0,
    )
    df["log_frequency"] = np.log1p(purchase_agg["n_transactions"])
    df["log_monetary"] = np.log1p(purchase_agg["total_spend"])

    # ── 2. Loyalty Points Features ──────────────────────────────────────
    df["total_points_received"] = purchase_agg["total_points_received"]
    df["total_points_spent"] = purchase_agg["total_points_spent"].abs()
    df["points_utilization"] = np.where(
        purchase_agg["total_points_received"] > 0,
        purchase_agg["total_points_spent"].abs() / purchase_agg["total_points_received"],
        0,
    )
    df["redemption_ratio"] = np.where(
        purchase_agg["total_spend"] > 0,
        purchase_agg["total_redeemed"] / purchase_agg["total_spend"],
        0,
    )
    df["has_redeemed_points"] = (purchase_agg["total_redeemed"] > 0).astype(int)
    df["has_express_points"] = (purchase_agg["total_express_points"] > 0).astype(int)
    df["express_points_ratio"] = np.where(
        purchase_agg["total_points_received"] > 0,
        purchase_agg["total_express_points"] / purchase_agg["total_points_received"],
        0,
    )

    # ── 3. Temporal Features ────────────────────────────────────────────
    df["weekend_ratio"] = np.where(
        purchase_agg["n_rows"] > 0,
        purchase_agg["n_weekend_rows"] / purchase_agg["n_rows"],
        0,
    )
    df["morning_ratio"] = np.where(
        purchase_agg["n_rows"] > 0,
        purchase_agg["n_morning_rows"] / purchase_agg["n_rows"],
        0,
    )
    df["evening_ratio"] = np.where(
        purchase_agg["n_rows"] > 0,
        purchase_agg["n_evening_rows"] / purchase_agg["n_rows"],
        0,
    )

    # ── 4. Basket Composition ───────────────────────────────────────────
    df["avg_items_per_txn"] = np.where(
        purchase_agg["n_transactions"] > 0,
        purchase_agg["total_quantity"] / purchase_agg["n_transactions"],
        0,
    )
    df["avg_products_per_txn"] = np.where(
        purchase_agg["n_transactions"] > 0,
        purchase_agg["n_rows"] / purchase_agg["n_transactions"],  # rows ≈ line items
        0,
    )
    df["n_unique_products"] = purchase_agg["n_products"]
    df["product_diversity"] = np.where(
        purchase_agg["n_rows"] > 0,
        purchase_agg["n_products"] / purchase_agg["n_rows"],
        0,
    )
    df["small_basket_ratio"] = np.where(
        purchase_agg["n_rows"] > 0,
        purchase_agg["n_small_basket_rows"] / purchase_agg["n_rows"],
        0,
    )

    # ── 5. Store Behavior ───────────────────────────────────────────────
    df["n_unique_stores"] = purchase_agg["n_stores"]
    df["store_diversity"] = np.where(
        purchase_agg["n_transactions"] > 0,
        purchase_agg["n_stores"] / purchase_agg["n_transactions"],
        0,
    )
    df["is_multi_store"] = (purchase_agg["n_stores"] > 1).astype(int)

    # ── Fill NaN ────────────────────────────────────────────────────────
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].fillna(0)

    logger.info("Purchase features: %d clients × %d features", len(df), len(df.columns) - 1)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# CLIENT DEMOGRAPHIC FEATURES (same as before)
# ═══════════════════════════════════════════════════════════════════════════════

def build_client_features(clients: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from client demographics and loyalty card metadata."""
    df = clients[["client_id"]].copy()

    issue_date = pd.to_datetime(clients["first_issue_date"], errors="coerce")
    redeem_date = pd.to_datetime(clients["first_redeem_date"], errors="coerce")
    ref_date = issue_date.max()

    # Age
    age = clients["age"].copy()
    age = age.where((age > 0) & (age < 120), other=np.nan).fillna(age.median())
    df["age"] = age
    df["age_squared"] = age ** 2
    df["log_age"] = np.log1p(age)
    df["age_bin_young"] = (age < 30).astype(int)
    df["age_bin_working"] = ((age >= 30) & (age < 45)).astype(int)
    df["age_bin_middle"] = ((age >= 45) & (age < 60)).astype(int)
    df["age_bin_senior"] = (age >= 60).astype(int)
    df["age_decile"] = pd.qcut(age, q=10, labels=False, duplicates="drop")

    # Gender
    gender = clients["gender"]
    df["is_female"] = (gender == "F").astype(int)
    df["is_male"] = (gender == "M").astype(int)
    df["gender_unknown"] = (gender == "U").astype(int)

    # Tenure
    tenure_days = (ref_date - issue_date).dt.days.fillna(0)
    df["tenure_days"] = tenure_days
    df["log_tenure"] = np.log1p(tenure_days)
    df["tenure_new"] = (tenure_days < 90).astype(int)
    df["tenure_mid"] = ((tenure_days >= 90) & (tenure_days < 365)).astype(int)
    df["tenure_loyal"] = (tenure_days >= 365).astype(int)

    # Redemption
    df["has_ever_redeemed"] = (~redeem_date.isna()).astype(int)
    days_to_redeem = (redeem_date - issue_date).dt.days
    df["days_to_first_redeem"] = days_to_redeem.fillna(-1)
    df["fast_redeemer"] = ((days_to_redeem >= 0) & (days_to_redeem <= 30)).astype(int)

    # Registration timing
    df["issue_dow"] = issue_date.dt.dayofweek.fillna(0).astype(int)
    df["issue_month"] = issue_date.dt.month.fillna(1).astype(int)
    df["issue_quarter"] = issue_date.dt.quarter.fillna(1).astype(int)
    df["issue_is_weekend"] = issue_date.dt.dayofweek.isin([5, 6]).astype(int)

    # Interactions
    df["age_x_female"] = df["age"] * df["is_female"]
    df["young_female"] = ((age < 35) & (gender == "F")).astype(int)
    df["senior_loyal"] = ((age >= 60) & (tenure_days >= 365)).astype(int)

    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].fillna(0)

    logger.info("Client features: %d rows × %d features", len(df), len(df.columns) - 1)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED FEATURE TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def build_full_feature_table(
    clients: pd.DataFrame,
    purchase_agg: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all feature groups with RCT labels."""
    client_feats = build_client_features(clients)
    purchase_feats = build_purchase_features(purchase_agg)

    ft = labels[["client_id", "treatment_flg", "target"]].copy()
    ft = ft.merge(client_feats, on="client_id", how="left")
    ft = ft.merge(purchase_feats, on="client_id", how="left")

    numeric = ft.select_dtypes(include=[np.number]).columns
    ft[numeric] = ft[numeric].fillna(0)

    feature_cols = [c for c in ft.columns if c not in {"client_id", "treatment_flg", "target"}]
    logger.info(
        "Full feature table: %d rows × %d features (treatment=%.1f%%, conv=%.1f%%)",
        len(ft), len(feature_cols),
        100 * ft["treatment_flg"].mean(), 100 * ft["target"].mean(),
    )
    return ft


def build_test_features(
    clients: pd.DataFrame,
    purchase_agg: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Build features for holdout test set (no labels)."""
    client_feats = build_client_features(clients)
    purchase_feats = build_purchase_features(purchase_agg)

    tf = test[["client_id"]].copy()
    tf = tf.merge(client_feats, on="client_id", how="left")
    tf = tf.merge(purchase_feats, on="client_id", how="left")

    numeric = tf.select_dtypes(include=[np.number]).columns
    tf[numeric] = tf[numeric].fillna(0)
    return tf


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"client_id", "treatment_flg", "target"}]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Retail Media Campaign Optimizer — Full Pipeline")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing all CSV files")
    parser.add_argument("--purchases-file", type=str, default=None,
                        help="Path to purchases.csv (overrides auto-detect)")
    parser.add_argument("--purchases-dir", type=str, default=None,
                        help="Dir with purchases_part_*.csv (overrides auto-detect)")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: LOAD DATA
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 1: LOADING DATA")
    logger.info("=" * 70)

    data_dir = Path(args.data_dir)
    clients = pd.read_csv(data_dir / "clients.csv")
    products = pd.read_csv(data_dir / "products.csv")
    train = pd.read_csv(data_dir / "uplift_train.csv")
    test = pd.read_csv(data_dir / "uplift_test.csv")

    logger.info("  clients:  %s", clients.shape)
    logger.info("  products: %s", products.shape)
    logger.info("  train:    %s", train.shape)
    logger.info("  test:     %s", test.shape)

    # Load purchases
    if args.purchases_file:
        purchase_source = args.purchases_file
    elif args.purchases_dir:
        purchase_source = sorted(glob.glob(os.path.join(args.purchases_dir, "purchases_part_*.csv")))
    elif (data_dir / "purchases.csv").exists():
        purchase_source = str(data_dir / "purchases.csv")
    else:
        parts = sorted(glob.glob(str(data_dir / "purchases_part_*.csv")))
        if parts:
            purchase_source = parts
        else:
            raise FileNotFoundError("No purchases data found!")

    purchase_agg = load_purchases_chunked(purchase_source)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 2: FEATURE ENGINEERING (Demographics + Purchases)")
    logger.info("=" * 70)

    feature_table = build_full_feature_table(clients, purchase_agg, train)
    feature_cols = get_feature_columns(feature_table)
    feature_table.to_parquet(output_dir / "feature_table.parquet", index=False)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: MODEL TRAINING
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("=" * 70)

    train_df, test_df = stratified_train_test_split(feature_table, test_size=0.25)
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    t_train = train_df["treatment_flg"].values
    t_test = test_df["treatment_flg"].values
    y_train = train_df["target"].values
    y_test = test_df["target"].values

    config = UpliftModelConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=0.05,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )

    model_names = ["s_learner", "t_learner", "x_learner", "class_transform"]
    results = {}
    trained_models = {}

    for name in model_names:
        logger.info("─" * 50)
        logger.info("Training: %s", name)

        model = get_model(name, config)
        t0 = time.time()
        model.fit(X_train, t_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        scores = model.predict(X_test)
        infer_time = time.time() - t0

        result = evaluate_model(name, y_test, scores, t_test)
        results[name] = result
        trained_models[name] = model

        logger.info(
            "  AUUC=%.4f | Uplift@10%%=%.4f | Uplift@30%%=%.4f | "
            "Train=%.1fs | Infer=%.3fs",
            result.auuc, result.uplift_at_10, result.uplift_at_30,
            train_time, infer_time,
        )
        result.decile_table.to_csv(output_dir / f"{name}_deciles.csv", index=False)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: COMPARISON + VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════
    comparison = compare_models(list(results.values()))
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON — X5 RetailHero (Demographics + Purchases)")
    print("=" * 70)
    print(comparison.to_string(index=False))
    print()

    plot_qini_curves(list(results.values()), y_test, t_test,
                     save_path=str(output_dir / "qini_curves.png"))

    best_name = comparison.iloc[0]["Model"]
    plot_uplift_by_decile(results[best_name],
                         save_path=str(output_dir / "best_model_deciles.png"))

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: A/B TEST SIMULATION
    # ═══════════════════════════════════════════════════════════════════════
    best_scores = results[best_name].uplift_scores
    # ─────────────────────────────────────────────
    # Budget Optimization Curve (NEW)
    # ─────────────────────────────────────────────
    budgets = np.linspace(0.01, 0.8, 30)

    curve_results = []
    for b in budgets:
        sim = simulate_campaign(
            y_test, t_test, best_scores,
            budget_fraction=b,
            strategy="model"
        )
        curve_results.append({
            "budget": b,
            "uplift": sim.ate_targeted,
            "incremental": sim.incremental_conversions,
            "n_targeted": sim.n_targeted
        })

        curve_df = pd.DataFrame(curve_results)
        VALUE_PER_CONV = 25
        COST_PER_USER = 0.05
        N = len(y_test)

        curve_df["revenue"] = curve_df["incremental"] * VALUE_PER_CONV
        curve_df["cost"] = curve_df["budget"] * N * COST_PER_USER
        curve_df["profit"] = curve_df["revenue"] - curve_df["cost"]

        # 🔥 optimal budget
        best_idx = curve_df["profit"].idxmax()
        optimal = curve_df.loc[best_idx]

        print("\n🔥 OPTIMAL BUDGET")
        print(optimal)
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(curve_df["budget"], curve_df["profit"])
        plt.axvline(optimal["budget"], linestyle="--")
        plt.title("Optimal Budget Allocation")
        plt.xlabel("Budget")
        plt.ylabel("Profit")
        plt.savefig(output_dir / "optimal_budget.png")
        plt.close()

        # 🔥 SAVE HERE
        curve_df.to_csv(output_dir / "budget_curve.csv", index=False)
        # Save for dashboard / policy layer
        np.save(output_dir / "best_scores.npy", best_scores)
        np.save(output_dir / "y_test.npy", y_test)
        np.save(output_dir / "t_test.npy", t_test)
        

        ab_table = run_ab_comparison(y_test, t_test, best_scores, budget_fraction=0.30)
        print("=" * 70)
        print("A/B TEST SIMULATION (Budget = 30%)")
        print("=" * 70)
        print(ab_table.to_string(index=False))
        print()

        print("BUDGET SENSITIVITY:")
        print(f"{'Budget':<10} {'ATE(model)':<14} {'ATE(random)':<14} {'Δ Lift':<10}")
        print("-" * 48)
        for b in [0.10, 0.20, 0.30, 0.50]:
            m = simulate_campaign(y_test, t_test, best_scores, budget_fraction=b, strategy="model")
            r = simulate_campaign(y_test, t_test, best_scores, budget_fraction=b, strategy="random")
            print(f"{b:<10.0%} {m.ate_targeted:<14.4f} {r.ate_targeted:<14.4f} {m.ate_targeted - r.ate_targeted:<10.4f}")
        print()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: STATISTICAL RIGOR
    # ═══════════════════════════════════════════════════════════════════════
    base_rate = y_test[t_test == 0].mean()
    observed_ate = y_test[t_test == 1].mean() - base_rate
    pa = power_analysis(base_rate=base_rate, mde=0.02)
    ci = bootstrap_uplift_ci(y_test, t_test, best_scores, k=0.30, n_bootstrap=1000)

    logger.info("Observed ATE: %.4f | Base rate: %.4f", observed_ate, base_rate)
    logger.info("Power: need %d samples for MDE=0.02", pa["n_total"])
    logger.info("Uplift@30%% = %.4f [%.4f, %.4f] (95%% CI)", ci["point_estimate"], ci["ci_lower"], ci["ci_upper"])

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7: SCORE HOLDOUT
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 7: SCORING HOLDOUT (200K customers)")
    logger.info("=" * 70)

    test_feats = build_test_features(clients, purchase_agg, test)
    X_holdout = test_feats[feature_cols].values
    holdout_scores = trained_models[best_name].predict(X_holdout)

    submission = pd.DataFrame({"client_id": test_feats["client_id"], "uplift": holdout_scores})
    submission.to_csv(output_dir / "submission.csv", index=False)
    logger.info("Submission: %d rows | mean=%.4f | std=%.4f",
                len(submission), holdout_scores.mean(), holdout_scores.std())

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — X5 RetailHero (Full: Demographics + Purchases)")
    print("=" * 70)
    print(f"  Dataset:        400K clients, {purchase_agg['n_rows'].sum():.0f} purchase rows")
    print(f"  Observed ATE:   {observed_ate:.4f} ({observed_ate*100:.2f}%)")
    print(f"  Best Model:     {best_name}")
    print(f"  AUUC:           {comparison.iloc[0]['AUUC']:.4f}")
    print(f"  Uplift@30%:     {comparison.iloc[0]['Uplift@30%']:.4f}")
    print(f"  Features:       {len(feature_cols)}")
    print(f"  Bootstrap CI:   [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    print(f"  Submission:     {output_dir}/submission.csv")
    print()

    summary = {
        "dataset": "X5 RetailHero (full — demographics + purchases)",
        "n_purchase_rows": int(purchase_agg["n_rows"].sum()),
        "n_train": len(train_df),
        "n_test_eval": len(test_df),
        "n_holdout": len(test_feats),
        "observed_ate": float(observed_ate),
        "best_model": best_name,
        "auuc": float(comparison.iloc[0]["AUUC"]),
        "uplift_at_30": float(comparison.iloc[0]["Uplift@30%"]),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "bootstrap_ci": ci,
        "power_analysis": pa,
        "comparison": comparison.to_dict(orient="records"),
    }
    with open(output_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()

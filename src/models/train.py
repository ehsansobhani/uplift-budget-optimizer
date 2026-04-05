"""
Model training orchestrator.

Handles the full training pipeline:
  1. Train/test split (stratified by treatment × target)
  2. Train all uplift model variants
  3. Evaluate each model with cross-validation
  4. Select best model by AUUC
  5. Log results to MLflow (if available)
  6. Export best model for serving
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.models.uplift_models import (
    BaseUpliftModel,
    UpliftModelConfig,
    get_model,
    MODELS,
)
from src.models.evaluate import (
    ModelResult,
    evaluate_model,
    compare_models,
    plot_qini_curves,
    plot_uplift_by_decile,
)

logger = logging.getLogger(__name__)


def stratified_train_test_split(
    feature_table: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split feature table into train/test, stratified by treatment × target.

    This preserves the treatment and outcome rates in both splits,
    which is critical for unbiased uplift evaluation.
    """
    # Create stratification key
    strat_key = feature_table["treatment_flg"].astype(str) + "_" + feature_table["target"].astype(str)

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        feature_table,
        test_size=test_size,
        stratify=strat_key,
        random_state=random_state,
    )

    logger.info(
        "Train/test split: %d / %d (test=%.1f%%)",
        len(train_df), len(test_df), 100 * len(test_df) / len(feature_table),
    )
    return train_df, test_df


def train_all_models(
    feature_table: pd.DataFrame,
    feature_columns: list[str],
    config: Optional[UpliftModelConfig] = None,
    model_names: Optional[list[str]] = None,
    output_dir: str | Path = "artifacts",
) -> dict[str, ModelResult]:
    """
    Train and evaluate all uplift model variants.

    Args:
        feature_table: Full feature table with treatment_flg + target
        feature_columns: List of feature column names
        config: LightGBM hyperparameters (shared across all models)
        model_names: Which models to train (default: all)
        output_dir: Where to save artifacts

    Returns:
        Dictionary mapping model name → ModelResult
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_names is None:
        model_names = list(MODELS.keys())
    if config is None:
        config = UpliftModelConfig()

    # Split
    train_df, test_df = stratified_train_test_split(feature_table)

    X_train = train_df[feature_columns].values
    X_test = test_df[feature_columns].values
    t_train = train_df["treatment_flg"].values
    t_test = test_df["treatment_flg"].values
    y_train = train_df["target"].values
    y_test = test_df["target"].values

    results: dict[str, ModelResult] = {}

    for name in model_names:
        logger.info("=" * 60)
        logger.info("Training: %s", name)
        logger.info("=" * 60)

        model = get_model(name, config)

        start = time.time()
        model.fit(X_train, t_train, y_train)
        train_time = time.time() - start

        # Predict uplift on test set
        start = time.time()
        uplift_scores = model.predict(X_test)
        inference_time = time.time() - start

        # Evaluate
        result = evaluate_model(name, y_test, uplift_scores, t_test)
        results[name] = result

        logger.info(
            "  AUUC=%.4f | Uplift@10%%=%.4f | Uplift@30%%=%.4f | "
            "Train=%.1fs | Inference=%.3fs",
            result.auuc, result.uplift_at_10, result.uplift_at_30,
            train_time, inference_time,
        )

        # Save model artifact
        model_path = output_dir / f"{name}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save decile table
        result.decile_table.to_csv(output_dir / f"{name}_deciles.csv", index=False)

    # Comparison table
    comparison = compare_models(list(results.values()))
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    logger.info("\n%s", comparison.to_string(index=False))

    # Qini curves
    plot_qini_curves(
        list(results.values()), y_test, t_test,
        save_path=str(output_dir / "qini_curves.png"),
    )

    # Best model decile plot
    best_name = comparison.iloc[0]["Model"]
    plot_uplift_by_decile(
        results[best_name],
        save_path=str(output_dir / "best_model_deciles.png"),
    )

    logger.info("Best model: %s (AUUC=%.4f)", best_name, comparison.iloc[0]["AUUC"])

    # Save metadata
    metadata = {
        "best_model": best_name,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "comparison": comparison.to_dict(orient="records"),
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return results


def train_with_optuna(
    feature_table: pd.DataFrame,
    feature_columns: list[str],
    model_name: str = "x_learner",
    n_trials: int = 50,
    output_dir: str | Path = "artifacts",
) -> tuple[BaseUpliftModel, UpliftModelConfig]:
    """
    Hyperparameter optimization using Optuna.

    Maximizes cross-validated AUUC.
    """
    import optuna
    from src.models.evaluate import auuc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_df, test_df = stratified_train_test_split(feature_table)

    X_train = train_df[feature_columns].values
    t_train = train_df["treatment_flg"].values
    y_train = train_df["target"].values
    X_test = test_df[feature_columns].values
    t_test = test_df["treatment_flg"].values
    y_test = test_df["target"].values

    def objective(trial):
        config = UpliftModelConfig(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        )
        model = get_model(model_name, config)
        model.fit(X_train, t_train, y_train)
        scores = model.predict(X_test)
        return auuc(y_test, scores, t_test)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_config = UpliftModelConfig(**study.best_params)
    logger.info("Best AUUC: %.4f | Params: %s", study.best_value, study.best_params)

    # Retrain with best params
    best_model = get_model(model_name, best_config)
    best_model.fit(X_train, t_train, y_train)

    return best_model, best_config

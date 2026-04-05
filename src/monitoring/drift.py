"""
Production monitoring for uplift model drift and data quality.

Implements:
  - Population Stability Index (PSI) for feature drift
  - KS test for prediction distribution drift
  - Uplift decay tracking
  - Data quality checks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Summary of drift detection results."""
    feature_name: str
    metric: str
    value: float
    threshold: float
    is_drifted: bool
    severity: str  # "none", "warning", "critical"


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Population Stability Index (PSI).

    PSI < 0.1:  No significant drift
    PSI 0.1-0.2: Moderate drift (warning)
    PSI > 0.2:  Significant drift (retrain)

    Args:
        reference: Training/reference distribution
        current: Current/production distribution
        n_bins: Number of bins for discretization
    """
    # Use reference distribution to define bin edges
    edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]

    # Normalize to proportions (add small epsilon to avoid log(0))
    eps = 1e-6
    ref_pct = ref_counts / len(reference) + eps
    cur_pct = cur_counts / len(current) + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def ks_drift_test(
    reference: np.ndarray,
    current: np.ndarray,
    alpha: float = 0.01,
) -> tuple[float, float, bool]:
    """
    Kolmogorov-Smirnov test for distribution shift.

    Returns (statistic, p_value, is_drifted).
    """
    stat, p_value = stats.ks_2samp(reference, current)
    return float(stat), float(p_value), p_value < alpha


def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list[str],
    psi_threshold: float = 0.2,
    psi_warning: float = 0.1,
) -> list[DriftReport]:
    """
    Run drift detection across all features.

    Args:
        reference_df: Training data features
        current_df: Current production data features
        feature_columns: Columns to check
        psi_threshold: PSI threshold for critical drift
        psi_warning: PSI threshold for warning

    Returns:
        List of DriftReport objects, one per feature.
    """
    reports = []

    for col in feature_columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_vals = reference_df[col].dropna().values
        cur_vals = current_df[col].dropna().values

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue

        psi = population_stability_index(ref_vals, cur_vals)

        if psi > psi_threshold:
            severity = "critical"
        elif psi > psi_warning:
            severity = "warning"
        else:
            severity = "none"

        reports.append(DriftReport(
            feature_name=col,
            metric="PSI",
            value=round(psi, 4),
            threshold=psi_threshold,
            is_drifted=psi > psi_threshold,
            severity=severity,
        ))

    n_drifted = sum(1 for r in reports if r.is_drifted)
    n_warning = sum(1 for r in reports if r.severity == "warning")
    logger.info(
        "Drift check: %d features | %d critical | %d warning | %d ok",
        len(reports), n_drifted, n_warning,
        len(reports) - n_drifted - n_warning,
    )

    return reports


def detect_prediction_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
) -> DriftReport:
    """Check if the uplift score distribution has shifted."""
    ks_stat, p_value, is_drifted = ks_drift_test(reference_scores, current_scores)

    return DriftReport(
        feature_name="uplift_score",
        metric="KS_statistic",
        value=round(ks_stat, 4),
        threshold=0.01,
        is_drifted=is_drifted,
        severity="critical" if is_drifted else "none",
    )


def data_quality_checks(df: pd.DataFrame) -> dict:
    """
    Run basic data quality checks on incoming data.

    Returns a dict of check results.
    """
    checks = {}

    # Null rate per column
    null_rates = df.isnull().mean()
    checks["null_rates"] = null_rates[null_rates > 0].to_dict()
    checks["has_null_issues"] = bool((null_rates > 0.1).any())

    # Duplicate rows
    n_dupes = df.duplicated().sum()
    checks["n_duplicate_rows"] = int(n_dupes)
    checks["has_duplicate_issues"] = n_dupes > 0

    # Numeric range checks (detect extreme outliers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_cols = []
    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) < 10:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        n_outliers = ((vals < q1 - 3 * iqr) | (vals > q3 + 3 * iqr)).sum()
        if n_outliers / len(vals) > 0.05:
            outlier_cols.append(col)
    checks["extreme_outlier_columns"] = outlier_cols
    checks["has_outlier_issues"] = len(outlier_cols) > 0

    # Row count sanity
    checks["n_rows"] = len(df)
    checks["n_columns"] = len(df.columns)
    checks["has_empty_data"] = len(df) == 0

    # Overall pass/fail
    checks["passed"] = not any([
        checks["has_null_issues"],
        checks["has_duplicate_issues"],
        checks["has_outlier_issues"],
        checks["has_empty_data"],
    ])

    return checks


def generate_monitoring_report(
    feature_drift: list[DriftReport],
    prediction_drift: DriftReport,
    data_quality: dict,
) -> pd.DataFrame:
    """Aggregate all monitoring signals into a single report."""
    rows = []

    for dr in feature_drift:
        if dr.severity != "none":
            rows.append({
                "Signal": f"Feature Drift: {dr.feature_name}",
                "Metric": dr.metric,
                "Value": dr.value,
                "Threshold": dr.threshold,
                "Status": dr.severity.upper(),
            })

    rows.append({
        "Signal": "Prediction Drift",
        "Metric": prediction_drift.metric,
        "Value": prediction_drift.value,
        "Threshold": prediction_drift.threshold,
        "Status": prediction_drift.severity.upper(),
    })

    rows.append({
        "Signal": "Data Quality",
        "Metric": "overall",
        "Value": 1.0 if data_quality["passed"] else 0.0,
        "Threshold": 1.0,
        "Status": "OK" if data_quality["passed"] else "CRITICAL",
    })

    return pd.DataFrame(rows)

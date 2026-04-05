"""
Evaluation metrics for uplift models.

Implements:
  - Qini curve computation and plotting
  - Area Under Uplift Curve (AUUC)
  - Uplift@K% (practical targeting metric)
  - Uplift by decile analysis
  - Model comparison framework
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def qini_curve(
    y_true: np.ndarray,
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Qini curve.

    The Qini curve plots cumulative incremental conversions as we expand
    the targeted population from highest to lowest predicted uplift.

    Args:
        y_true: Binary outcome (0/1)
        uplift_scores: Predicted uplift scores (higher = more persuadable)
        treatment: Treatment indicator (0/1)
        n_bins: Number of points on the curve

    Returns:
        (fractions, qini_values): arrays for plotting
    """
    y_true = np.asarray(y_true)
    uplift_scores = np.asarray(uplift_scores)
    treatment = np.asarray(treatment)

    # Sort by descending uplift score
    order = np.argsort(-uplift_scores)
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    n = len(y_true)
    n_treat = treatment.sum()
    n_ctrl = (1 - treatment).sum()

    fractions = np.linspace(0, 1, n_bins + 1)
    qini_values = np.zeros(n_bins + 1)

    for i, frac in enumerate(fractions):
        if frac == 0:
            continue
        k = int(np.ceil(frac * n))
        y_k = y_sorted[:k]
        t_k = t_sorted[:k]

        n_t_k = t_k.sum()
        n_c_k = k - n_t_k

        if n_t_k == 0 or n_c_k == 0:
            qini_values[i] = qini_values[i - 1] if i > 0 else 0
            continue

        # Qini: incremental gains
        rate_t = y_k[t_k == 1].sum() / n_t_k
        rate_c = y_k[t_k == 0].sum() / n_c_k
        qini_values[i] = (rate_t - rate_c) * k

    return fractions, qini_values


def auuc(
    y_true: np.ndarray,
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
    n_bins: int = 100,
) -> float:
    """
    Area Under the Uplift Curve (AUUC).

    Higher is better. Normalized by the perfect AUUC for comparability.
    """
    fractions, qini_vals = qini_curve(y_true, uplift_scores, treatment, n_bins)
    # np.trapz was renamed to np.trapezoid in numpy 2.0
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(qini_vals, fractions))


def uplift_at_k(
    y_true: np.ndarray,
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
    k: float = 0.3,
) -> float:
    """
    Compute the actual uplift (ATE) within the top k% of model-ranked customers.

    This is the practical metric: if we target only 30% of customers,
    what incremental conversion rate do we achieve?

    Args:
        k: Fraction of population to target (e.g., 0.3 for top 30%)
    """
    y_true = np.asarray(y_true)
    uplift_scores = np.asarray(uplift_scores)
    treatment = np.asarray(treatment)

    # Select top-k by uplift score
    threshold = np.percentile(uplift_scores, 100 * (1 - k))
    mask = uplift_scores >= threshold

    y_k = y_true[mask]
    t_k = treatment[mask]

    n_treated = t_k.sum()
    n_control = len(t_k) - n_treated

    if n_treated == 0 or n_control == 0:
        return 0.0

    rate_t = y_k[t_k == 1].sum() / n_treated
    rate_c = y_k[t_k == 0].sum() / n_control

    return float(rate_t - rate_c)


def uplift_by_decile(
    y_true: np.ndarray,
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Compute actual uplift within each decile of predicted uplift scores.

    Returns a DataFrame with one row per decile (1=highest predicted uplift).
    """
    y_true = np.asarray(y_true)
    uplift_scores = np.asarray(uplift_scores)
    treatment = np.asarray(treatment)

    # Assign deciles (1 = highest uplift)
    decile_edges = np.percentile(uplift_scores, np.linspace(0, 100, n_deciles + 1))
    deciles = np.digitize(uplift_scores, decile_edges[1:-1]) + 1
    deciles = n_deciles + 1 - deciles  # flip so 1 = highest

    rows = []
    for d in range(1, n_deciles + 1):
        mask = deciles == d
        y_d = y_true[mask]
        t_d = treatment[mask]

        n_t = t_d.sum()
        n_c = len(t_d) - n_t

        if n_t > 0 and n_c > 0:
            rate_t = y_d[t_d == 1].sum() / n_t
            rate_c = y_d[t_d == 0].sum() / n_c
            actual_uplift = rate_t - rate_c
        else:
            rate_t = rate_c = actual_uplift = 0.0

        rows.append({
            "decile": d,
            "n_customers": int(mask.sum()),
            "n_treated": int(n_t),
            "n_control": int(n_c),
            "conversion_treated": round(rate_t, 4),
            "conversion_control": round(rate_c, 4),
            "actual_uplift": round(actual_uplift, 4),
            "avg_predicted_uplift": round(float(uplift_scores[mask].mean()), 4),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelResult:
    """Container for a single model's evaluation results."""
    name: str
    auuc: float
    uplift_at_10: float
    uplift_at_30: float
    uplift_at_50: float
    decile_table: pd.DataFrame
    uplift_scores: np.ndarray


def evaluate_model(
    name: str,
    y_true: np.ndarray,
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
) -> ModelResult:
    """Full evaluation of a single uplift model."""
    return ModelResult(
        name=name,
        auuc=auuc(y_true, uplift_scores, treatment),
        uplift_at_10=uplift_at_k(y_true, uplift_scores, treatment, k=0.10),
        uplift_at_30=uplift_at_k(y_true, uplift_scores, treatment, k=0.30),
        uplift_at_50=uplift_at_k(y_true, uplift_scores, treatment, k=0.50),
        decile_table=uplift_by_decile(y_true, uplift_scores, treatment),
        uplift_scores=uplift_scores,
    )


def compare_models(results: list[ModelResult]) -> pd.DataFrame:
    """Create a comparison table across all models."""
    rows = []
    for r in results:
        rows.append({
            "Model": r.name,
            "AUUC": round(r.auuc, 4),
            "Uplift@10%": round(r.uplift_at_10, 4),
            "Uplift@30%": round(r.uplift_at_30, 4),
            "Uplift@50%": round(r.uplift_at_50, 4),
        })
    return pd.DataFrame(rows).sort_values("AUUC", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_qini_curves(
    results: list[ModelResult],
    y_true: np.ndarray,
    treatment: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Qini curves for multiple models on the same axes."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for r in results:
        fracs, qvals = qini_curve(y_true, r.uplift_scores, treatment)
        ax.plot(fracs, qvals, label=f"{r.name} (AUUC={r.auuc:.4f})", linewidth=2)

    # Random targeting baseline
    fracs_rand, qvals_rand = qini_curve(
        y_true, np.random.RandomState(42).randn(len(y_true)), treatment
    )
    ax.plot(fracs_rand, qvals_rand, "--", color="gray", label="Random", linewidth=1)

    ax.set_xlabel("Fraction of Population Targeted", fontsize=12)
    ax.set_ylabel("Cumulative Incremental Conversions", fontsize=12)
    ax.set_title("Qini Curves — Model Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Qini plot saved → %s", save_path)

    return fig


def plot_uplift_by_decile(
    result: ModelResult,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of actual uplift by model-predicted decile."""
    fig, ax = plt.subplots(figsize=(10, 6))
    dt = result.decile_table

    colors = ["#2ecc71" if u > 0 else "#e74c3c" for u in dt["actual_uplift"]]
    ax.bar(dt["decile"], dt["actual_uplift"], color=colors, edgecolor="white")

    ax.set_xlabel("Decile (1 = Highest Predicted Uplift)", fontsize=12)
    ax.set_ylabel("Actual Uplift (Treatment Effect)", fontsize=12)
    ax.set_title(f"Uplift by Decile — {result.name}", fontsize=14)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(1, 11))
    ax.grid(True, alpha=0.3, axis="y")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

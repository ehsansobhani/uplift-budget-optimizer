"""
Advanced Qini curve analytics for retail media campaign optimization.

Three well-established extensions of the basic Qini curve:

1. BUDGET OPTIMIZATION CURVE (Sverdrup, Wu, Athey & Wager, 2025)
   Transforms the Qini curve into a cost-benefit tool by putting real
   dollar amounts on both axes. At each spend level B, shows the expected
   incremental revenue from targeting the top-B-worth of customers.
   Key output: the marginal ROI curve — shows exactly where each additional
   dollar of campaign spend stops being profitable.

2. OPTIMAL TARGETING CUTOFF (Radcliffe & Surry, 2011)
   Finds the fraction of the population to target that maximizes
   incremental profit (revenue minus cost). This is the "impact at cutoff"
   — the budget level where marginal uplift equals marginal cost.

3. NORMALIZED MODEL COMPARISON (Radcliffe, 2007; Belbahri et al., 2021)
   The raw AUUC is not comparable across datasets because the perfect
   Qini curve varies with the data. Three established normalizations:
   - Qini Coefficient: AUUC(model) - AUUC(random)
   - Adjusted Qini (q₀): AUUC(model) / AUUC(perfect)
   - Bootstrap hypothesis tests between models (Sverdrup et al., 2025)

References:
  - Radcliffe, N.J. (2007). Using Control Groups to Target on Predicted Lift.
  - Radcliffe, N.J. & Surry, P.D. (2011). Real-World Uplift Modelling with
    Significance-Based Uplift Trees. White paper, Stochastic Solutions.
  - Belbahri, M. et al. (2021). Qini-based Uplift Regression.
    Annals of Applied Statistics, 15(3):1247-1272.
  - Sverdrup, E., Wu, H., Athey, S. & Wager, S. (2025). Qini Curves for
    Multi-Armed Treatment Rules. J. Computational & Graphical Statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

logger = logging.getLogger(__name__)

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: FINE-GRAINED QINI DATA
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_qini_data(
    y_true: np.ndarray,
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
    n_points: int = 200,
) -> pd.DataFrame:
    """
    Compute a fine-grained Qini table with per-point statistics.

    Returns a DataFrame with one row per evaluation point, containing:
      - frac: fraction of population targeted
      - n_targeted: absolute count
      - n_treated / n_control within targeted group
      - conversion rates for treatment and control
      - incremental_rate: treatment effect in the targeted group
      - cum_incremental: cumulative incremental conversions (Qini value)
      - marginal_uplift: marginal uplift of the last slice added
    """
    y = np.asarray(y_true)
    s = np.asarray(uplift_scores)
    t = np.asarray(treatment)
    n = len(y)

    order = np.argsort(-s)
    y_sorted = y[order]
    t_sorted = t[order]

    fracs = np.linspace(0, 1, n_points + 1)
    rows = []

    prev_inc_conversions = 0.0
    prev_k = 0

    for frac in fracs:
        k = int(np.ceil(frac * n)) if frac > 0 else 0

        if k == 0:
            rows.append({
                "frac": 0.0, "n_targeted": 0,
                "n_treated": 0, "n_control": 0,
                "conv_treated": 0.0, "conv_control": 0.0,
                "incremental_rate": 0.0,
                "cum_incremental": 0.0,
                "marginal_uplift": 0.0,
            })
            continue

        y_k = y_sorted[:k]
        t_k = t_sorted[:k]
        n_t = int(t_k.sum())
        n_c = k - n_t

        if n_t > 0 and n_c > 0:
            conv_t = y_k[t_k == 1].sum() / n_t
            conv_c = y_k[t_k == 0].sum() / n_c
            inc_rate = conv_t - conv_c
            cum_inc = inc_rate * k
        else:
            conv_t = conv_c = inc_rate = 0.0
            cum_inc = rows[-1]["cum_incremental"] if rows else 0.0

        # Marginal uplift of the slice from prev_k to k
        delta_k = k - prev_k
        marginal = (cum_inc - prev_inc_conversions) / max(delta_k, 1)

        rows.append({
            "frac": frac,
            "n_targeted": k,
            "n_treated": n_t,
            "n_control": n_c,
            "conv_treated": round(conv_t, 6),
            "conv_control": round(conv_c, 6),
            "incremental_rate": round(inc_rate, 6),
            "cum_incremental": round(cum_inc, 4),
            "marginal_uplift": round(marginal, 6),
        })

        prev_inc_conversions = cum_inc
        prev_k = k

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BUDGET OPTIMIZATION CURVE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BudgetOptimizationResult:
    """Result of a budget optimization analysis."""
    budget_curve: pd.DataFrame
    optimal_budget: float          # spend level that maximizes profit
    optimal_frac: float            # fraction targeted at optimal budget
    optimal_profit: float          # profit at optimal budget
    optimal_roi: float             # ROI at optimal budget
    max_incremental_revenue: float # revenue if targeting everyone with positive uplift
    breakeven_budget: float        # spend level where profit hits zero


def budget_optimization_curve(
    y_true: np.ndarray,
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
    cost_per_contact: float = 0.05,
    revenue_per_conversion: float = 25.0,
    n_points: int = 200,
) -> BudgetOptimizationResult:
    """
    Transform the Qini curve into a budget optimization curve.

    For each fraction of the population targeted (sorted by predicted uplift),
    compute:
      - Total campaign cost = n_targeted × cost_per_contact
      - Incremental revenue = cumulative_incremental_conversions × revenue_per_conversion
      - Incremental profit = revenue - cost
      - Marginal ROI of the last slice

    The optimal targeting cutoff is where incremental profit is maximized,
    which is equivalent to where marginal revenue equals marginal cost.

    This follows the cost-benefit framework of Sverdrup et al. (2025),
    specialized to the single-arm case with constant per-unit costs.

    Args:
        cost_per_contact: Cost of sending one SMS/email (CAD)
        revenue_per_conversion: Average revenue per incremental conversion (CAD)
    """
    qd = _compute_qini_data(y_true, uplift_scores, treatment, n_points)
    n_total = len(y_true)

    # Dollar-denominated columns
    qd["total_cost"] = qd["n_targeted"] * cost_per_contact
    qd["spend_per_capita"] = qd["total_cost"] / n_total  # average spend per person in population
    qd["incremental_revenue"] = qd["cum_incremental"] * revenue_per_conversion
    qd["incremental_profit"] = qd["incremental_revenue"] - qd["total_cost"]
    qd["roi"] = np.where(
        qd["total_cost"] > 0,
        qd["incremental_profit"] / qd["total_cost"],
        0.0,
    )
    qd["marginal_revenue_per_contact"] = qd["marginal_uplift"] * revenue_per_conversion
    qd["marginal_profit_per_contact"] = qd["marginal_revenue_per_contact"] - cost_per_contact

    # Optimal targeting point: max incremental profit
    opt_idx = qd["incremental_profit"].idxmax()
    opt_row = qd.loc[opt_idx]

    # Breakeven: last point where profit > 0
    profitable = qd[qd["incremental_profit"] > 0]
    breakeven_frac = profitable["frac"].max() if len(profitable) > 0 else 0.0
    breakeven_budget = breakeven_frac * n_total * cost_per_contact

    result = BudgetOptimizationResult(
        budget_curve=qd,
        optimal_budget=round(float(opt_row["total_cost"]), 2),
        optimal_frac=round(float(opt_row["frac"]), 4),
        optimal_profit=round(float(opt_row["incremental_profit"]), 2),
        optimal_roi=round(float(opt_row["roi"]), 4),
        max_incremental_revenue=round(float(qd["incremental_revenue"].max()), 2),
        breakeven_budget=round(breakeven_budget, 2),
    )

    logger.info(
        "Budget optimization: optimal at %.1f%% targeting "
        "(cost=$%.0f, profit=$%.0f, ROI=%.0f%%)",
        result.optimal_frac * 100,
        result.optimal_budget,
        result.optimal_profit,
        result.optimal_roi * 100,
    )
    return result


def plot_budget_optimization(
    bopt: BudgetOptimizationResult,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Three-panel budget optimization visualization:
      1. Incremental profit vs. budget (with optimal point)
      2. ROI vs. fraction targeted
      3. Marginal profit per contact (shows where to stop spending)
    """
    qd = bopt.budget_curve
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Profit curve
    ax1 = axes[0]
    ax1.fill_between(
        qd["frac"], 0, qd["incremental_profit"],
        where=qd["incremental_profit"] > 0, color="#2ecc71", alpha=0.3, label="Profit > 0",
    )
    ax1.fill_between(
        qd["frac"], 0, qd["incremental_profit"],
        where=qd["incremental_profit"] <= 0, color="#e74c3c", alpha=0.3, label="Profit < 0",
    )
    ax1.plot(qd["frac"], qd["incremental_profit"], "k-", linewidth=2)
    ax1.axvline(bopt.optimal_frac, color="#3498db", linestyle="--", linewidth=1.5,
                label=f"Optimal: {bopt.optimal_frac:.0%}")
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.scatter([bopt.optimal_frac], [bopt.optimal_profit], color="#e67e22",
                s=120, zorder=5, marker="*", label=f"Max profit: ${bopt.optimal_profit:,.0f}")
    ax1.set_xlabel("Fraction Targeted")
    ax1.set_ylabel("Incremental Profit ($)")
    ax1.set_title("Profit vs. Targeting Depth")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))

    # Panel 2: ROI
    ax2 = axes[1]
    valid_roi = qd[qd["frac"] > 0]
    ax2.plot(valid_roi["frac"], valid_roi["roi"] * 100, color="#9b59b6", linewidth=2)
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_xlabel("Fraction Targeted")
    ax2.set_ylabel("Campaign ROI (%)")
    ax2.set_title("ROI vs. Targeting Depth")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))

    # Panel 3: Marginal profit per contact
    ax3 = axes[2]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in qd["marginal_profit_per_contact"]]
    ax3.bar(qd["frac"], qd["marginal_profit_per_contact"], width=1/len(qd),
            color=colors, alpha=0.7, edgecolor="none")
    ax3.axhline(0, color="black", linewidth=1)
    ax3.axvline(bopt.optimal_frac, color="#3498db", linestyle="--", linewidth=1.5,
                label=f"Stop here: {bopt.optimal_frac:.0%}")
    ax3.set_xlabel("Fraction Targeted")
    ax3.set_ylabel("Marginal Profit per Contact ($)")
    ax3.set_title("When to Stop Spending")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))

    fig.suptitle("Budget Optimization — Qini Cost-Benefit Analysis", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Budget optimization plot saved → %s", save_path)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. OPTIMAL TARGETING CUTOFF
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TargetingCutoffResult:
    """Result of optimal cutoff analysis."""
    optimal_score_threshold: float   # uplift score cutoff
    optimal_frac: float              # fraction of population to target
    n_target: int                    # absolute number to target
    expected_ate_targeted: float     # ATE within targeted group
    expected_incremental: float      # total incremental conversions
    n_persuadable: int               # customers with uplift > 0
    n_sleeping_dogs: int             # customers with uplift < -threshold
    segment_table: pd.DataFrame      # Persuadable/SureThing/LostCause/SleepingDog


def optimal_targeting_cutoff(
    y_true: np.ndarray,
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
    cost_per_contact: float = 0.05,
    revenue_per_conversion: float = 25.0,
) -> TargetingCutoffResult:
    """
    Find the optimal uplift score threshold for targeting.

    Two complementary approaches:

    (a) Profit-maximizing cutoff (Radcliffe & Surry, 2011 "Impact at cutoff"):
        Target everyone whose predicted uplift × revenue > cost.
        Threshold = cost_per_contact / revenue_per_conversion

    (b) Empirical cutoff from the budget optimization curve:
        The fraction where incremental profit is maximized.

    Also segments customers into the four canonical groups:
        Persuadable / Sure Thing / Lost Cause / Sleeping Dog
    """
    y = np.asarray(y_true)
    s = np.asarray(uplift_scores)
    t = np.asarray(treatment)

    # (a) Theoretical threshold: marginal revenue = marginal cost
    theoretical_threshold = cost_per_contact / revenue_per_conversion

    # (b) Empirical: find profit-maximizing fraction
    bopt = budget_optimization_curve(y, s, t, cost_per_contact, revenue_per_conversion)

    # Use the empirical optimum (more robust to model miscalibration)
    opt_frac = bopt.optimal_frac
    score_threshold = float(np.percentile(s, 100 * (1 - opt_frac))) if opt_frac > 0 else s.max()
    n_target = int(np.ceil(opt_frac * len(y)))

    # Compute ATE in targeted group
    mask = s >= score_threshold
    y_m, t_m = y[mask], t[mask]
    nt, nc = t_m.sum(), len(t_m) - t_m.sum()
    if nt > 0 and nc > 0:
        ate_targeted = y_m[t_m == 1].sum() / nt - y_m[t_m == 0].sum() / nc
    else:
        ate_targeted = 0.0

    # Segment table
    segments = pd.DataFrame({
        "Segment": ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"],
        "Definition": [
            f"uplift > {theoretical_threshold:.4f}",
            f"0 < uplift ≤ {theoretical_threshold:.4f}",
            f"-{theoretical_threshold:.4f} ≤ uplift ≤ 0",
            f"uplift < -{theoretical_threshold:.4f}",
        ],
        "Count": [
            int((s > theoretical_threshold).sum()),
            int(((s > 0) & (s <= theoretical_threshold)).sum()),
            int(((s >= -theoretical_threshold) & (s <= 0)).sum()),
            int((s < -theoretical_threshold).sum()),
        ],
    })
    segments["Pct"] = (segments["Count"] / len(s) * 100).round(1)
    segments["Avg Predicted Uplift"] = [
        round(float(s[s > theoretical_threshold].mean()), 4) if (s > theoretical_threshold).any() else 0,
        round(float(s[(s > 0) & (s <= theoretical_threshold)].mean()), 4) if ((s > 0) & (s <= theoretical_threshold)).any() else 0,
        round(float(s[(s >= -theoretical_threshold) & (s <= 0)].mean()), 4) if ((s >= -theoretical_threshold) & (s <= 0)).any() else 0,
        round(float(s[s < -theoretical_threshold].mean()), 4) if (s < -theoretical_threshold).any() else 0,
    ]

    return TargetingCutoffResult(
        optimal_score_threshold=round(score_threshold, 6),
        optimal_frac=opt_frac,
        n_target=n_target,
        expected_ate_targeted=round(ate_targeted, 4),
        expected_incremental=round(ate_targeted * n_target, 1),
        n_persuadable=int((s > theoretical_threshold).sum()),
        n_sleeping_dogs=int((s < -theoretical_threshold).sum()),
        segment_table=segments,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NORMALIZED MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NormalizedComparisonResult:
    """Normalized model comparison results."""
    comparison_table: pd.DataFrame
    pairwise_tests: pd.DataFrame  # bootstrap hypothesis tests


def _perfect_qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the perfect (oracle) Qini curve.

    The perfect model would rank individuals by their true ITE.
    In an RCT, the best we can do is rank by group-level outcomes:
      1. Treated responders first (Y=1, T=1)  — these might be persuadable
      2. Control non-responders (Y=0, T=0)    — also persuadable signal
      3. Control responders (Y=1, T=0)         — sure things
      4. Treated non-responders (Y=0, T=1)    — lost causes / sleeping dogs
    """
    y = np.asarray(y_true)
    t = np.asarray(treatment)
    n = len(y)

    # Assign oracle priority scores
    # Highest priority: treated AND converted (contribute positively to Qini)
    # Lowest priority: treated AND not converted (contribute negatively)
    oracle_score = np.zeros(n)
    oracle_score[(t == 1) & (y == 1)] = 3  # best for Qini
    oracle_score[(t == 0) & (y == 0)] = 2
    oracle_score[(t == 0) & (y == 1)] = 1
    oracle_score[(t == 1) & (y == 0)] = 0  # worst for Qini

    # Add small random tiebreaker
    oracle_score += np.random.RandomState(0).uniform(0, 0.1, n)

    # Compute Qini with oracle ranking
    order = np.argsort(-oracle_score)
    y_sorted = y[order]
    t_sorted = t[order]

    fracs = np.linspace(0, 1, n_points + 1)
    qini_vals = np.zeros(n_points + 1)

    for i, frac in enumerate(fracs):
        if frac == 0:
            continue
        k = int(np.ceil(frac * n))
        y_k = y_sorted[:k]
        t_k = t_sorted[:k]
        n_t = t_k.sum()
        n_c = k - n_t
        if n_t > 0 and n_c > 0:
            qini_vals[i] = (y_k[t_k == 1].sum() / n_t - y_k[t_k == 0].sum() / n_c) * k
        else:
            qini_vals[i] = qini_vals[i - 1]

    return fracs, qini_vals


def _random_qini_auuc(
    y_true: np.ndarray,
    treatment: np.ndarray,
    n_points: int = 200,
) -> float:
    """AUUC of random targeting baseline (straight line from 0 to overall ATE × N)."""
    y = np.asarray(y_true)
    t = np.asarray(treatment)
    n = len(y)
    nt, nc = t.sum(), (1 - t).sum()
    if nt > 0 and nc > 0:
        overall_ate = y[t == 1].sum() / nt - y[t == 0].sum() / nc
    else:
        overall_ate = 0
    # Random Qini is a straight line: Q_rand(f) = f × ATE × N
    # AUUC_rand = integral of f × ATE × N from 0 to 1 = ATE × N / 2
    return overall_ate * n / 2


def normalized_model_comparison(
    model_results: list,  # list of (name, uplift_scores) tuples
    y_true: np.ndarray,
    treatment: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
    n_points: int = 200,
) -> NormalizedComparisonResult:
    """
    Fair, normalized model comparison using three established metrics:

    1. Qini Coefficient (Radcliffe 2007):
       Q_coeff = AUUC(model) - AUUC(random)
       Interpretation: total incremental gain above random targeting.
       Analogous to the Gini coefficient in credit scoring.

    2. Adjusted Qini / q₀ (Belbahri et al. 2021):
       q₀ = [AUUC(model) - AUUC(random)] / [AUUC(perfect) - AUUC(random)]
       Interpretation: fraction of the theoretically achievable uplift
       that the model actually captures. Range [0, 1] (or negative if
       worse than random). Comparable across datasets.

    3. Qini Top-K (Radcliffe & Surry 2011):
       The Qini value at a specific targeting depth (e.g., top 20%).
       Shows practical value at a given budget constraint.

    Additionally, pairwise bootstrap hypothesis tests (Sverdrup et al. 2025)
    determine whether differences between models are statistically significant.
    """
    y = np.asarray(y_true)
    t = np.asarray(treatment)
    rng = np.random.default_rng(seed)

    # Compute reference curves
    auuc_random = _random_qini_auuc(y, t, n_points)
    fracs_perfect, qvals_perfect = _perfect_qini_curve(y, t, n_points)
    auuc_perfect = float(_trapz(qvals_perfect, fracs_perfect))

    # Helper to compute AUUC from scores
    def _auuc(y_b, s_b, t_b):
        from src.models.evaluate import qini_curve
        f, q = qini_curve(y_b, s_b, t_b, n_bins=n_points)
        return float(_trapz(q, f))

    # Compute metrics for each model
    rows = []
    for name, scores in model_results:
        scores = np.asarray(scores)
        model_auuc = _auuc(y, scores, t)

        q_coeff = model_auuc - auuc_random
        denom = auuc_perfect - auuc_random
        q0 = q_coeff / denom if abs(denom) > 1e-9 else 0.0

        # Qini Top-K at various cutoffs
        from src.models.evaluate import uplift_at_k
        top10 = uplift_at_k(y, scores, t, k=0.10)
        top20 = uplift_at_k(y, scores, t, k=0.20)
        top30 = uplift_at_k(y, scores, t, k=0.30)

        rows.append({
            "Model": name,
            "AUUC (raw)": round(model_auuc, 2),
            "Qini Coeff": round(q_coeff, 2),
            "Adjusted q₀": round(q0, 4),
            "Top-10% Uplift": round(top10, 4),
            "Top-20% Uplift": round(top20, 4),
            "Top-30% Uplift": round(top30, 4),
        })

    comparison = pd.DataFrame(rows).sort_values("Qini Coeff", ascending=False).reset_index(drop=True)

    # Pairwise bootstrap hypothesis tests
    # H₀: AUUC(model_A) = AUUC(model_B)
    model_names = [name for name, _ in model_results]
    model_scores = {name: np.asarray(scores) for name, scores in model_results}
    n = len(y)

    pairwise_rows = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a, name_b = model_names[i], model_names[j]
            s_a, s_b = model_scores[name_a], model_scores[name_b]

            # Bootstrap the AUUC difference
            diffs = []
            for _ in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                auuc_a = _auuc(y[idx], s_a[idx], t[idx])
                auuc_b = _auuc(y[idx], s_b[idx], t[idx])
                diffs.append(auuc_a - auuc_b)

            diffs = np.array(diffs)
            mean_diff = diffs.mean()
            se = diffs.std()
            # Two-sided p-value: proportion of bootstrap samples where the
            # sign of the difference disagrees with the observed mean
            if se > 0:
                z = abs(mean_diff / se)
                from scipy.stats import norm
                p_value = 2 * (1 - norm.cdf(z))
            else:
                p_value = 1.0

            pairwise_rows.append({
                "Model A": name_a,
                "Model B": name_b,
                "ΔAUUC (A-B)": round(mean_diff, 2),
                "SE": round(se, 2),
                "p-value": round(p_value, 4),
                "Significant (α=0.05)": p_value < 0.05,
            })

    pairwise = pd.DataFrame(pairwise_rows)

    return NormalizedComparisonResult(
        comparison_table=comparison,
        pairwise_tests=pairwise,
    )


def plot_normalized_comparison(
    model_results: list,
    y_true: np.ndarray,
    treatment: np.ndarray,
    save_path: Optional[str] = None,
    n_points: int = 200,
) -> plt.Figure:
    """
    Publication-quality Qini curve plot with random and perfect baselines.

    Shows all three reference curves:
      - Model curves (colored)
      - Random baseline (diagonal gray dashed)
      - Perfect/oracle curve (green dotted)
    """
    from src.models.evaluate import qini_curve

    y = np.asarray(y_true)
    t = np.asarray(treatment)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Perfect curve
    fracs_p, qvals_p = _perfect_qini_curve(y, t, n_points)
    ax.plot(fracs_p, qvals_p, ":", color="#2ecc71", linewidth=2, label="Perfect (oracle)")

    # Random baseline
    n = len(y)
    nt, nc = t.sum(), (1 - t).sum()
    overall_ate = (y[t == 1].sum() / nt - y[t == 0].sum() / nc) if nt > 0 and nc > 0 else 0
    ax.plot([0, 1], [0, overall_ate * n], "--", color="gray", linewidth=1.5, label="Random")

    # Model curves
    colors = plt.cm.tab10.colors
    auuc_random = _random_qini_auuc(y, t, n_points)
    for idx, (name, scores) in enumerate(model_results):
        fracs, qvals = qini_curve(y, scores, t, n_bins=n_points)
        model_auuc = float(_trapz(qvals, fracs))
        q_coeff = model_auuc - auuc_random
        ax.plot(fracs, qvals, color=colors[idx % len(colors)], linewidth=2,
                label=f"{name} (Qini={q_coeff:.1f})")

    ax.set_xlabel("Fraction of Population Targeted", fontsize=12)
    ax.set_ylabel("Cumulative Incremental Conversions", fontsize=12)
    ax.set_title("Normalized Qini Curves with Oracle & Random Baselines", fontsize=14)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Normalized Qini plot saved → %s", save_path)

    return fig

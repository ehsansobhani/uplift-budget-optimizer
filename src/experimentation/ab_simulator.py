"""
A/B Test Simulation & Incrementality Measurement.

Simulates deploying the uplift model as a campaign targeting policy
and measures its effectiveness versus random targeting baselines.

Implements:
  - A/B test simulation (model-targeted vs. random)
  - Power analysis for experiment design
  - Incrementality metrics (ATE, incremental conversions, ROI)
  - Bootstrap confidence intervals
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class CampaignResult:
    """Results from a simulated campaign deployment."""
    strategy: str
    n_targeted: int
    n_total: int
    budget_fraction: float
    conversion_rate_targeted_treated: float
    conversion_rate_targeted_control: float
    incremental_conversions: float
    ate_targeted: float
    cost_per_sms: float
    total_cost: float
    avg_revenue_per_conversion: float
    incremental_revenue: float
    campaign_roi: float


def simulate_campaign(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    budget_fraction: float = 0.30,
    cost_per_sms: float = 0.05,
    avg_revenue_per_conversion: float = 25.0,
    strategy: str = "model",
    seed: int = 42,
) -> CampaignResult:
    """
    Simulate deploying a targeting strategy and measure incrementality.

    Compares:
      - "model": Target top-k% by predicted uplift score
      - "random": Target random k% of customers
      - "all": Target everyone (no model)

    Args:
        y_true: Actual conversion outcomes (0/1)
        treatment: Treatment assignment (0/1)
        uplift_scores: Model's predicted uplift per customer
        budget_fraction: Fraction of population to target (0-1)
        cost_per_sms: Cost per SMS sent (CAD)
        avg_revenue_per_conversion: Average revenue per conversion (CAD)
        strategy: "model", "random", or "all"
        seed: Random seed for "random" strategy
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    n_target = int(n * budget_fraction) if strategy != "all" else n

    if strategy == "model":
        # Target customers with highest predicted uplift
        threshold = np.percentile(uplift_scores, 100 * (1 - budget_fraction))
        targeted_mask = uplift_scores >= threshold
    elif strategy == "random":
        indices = rng.choice(n, size=n_target, replace=False)
        targeted_mask = np.zeros(n, dtype=bool)
        targeted_mask[indices] = True
    elif strategy == "all":
        targeted_mask = np.ones(n, dtype=bool)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Measure ATE within the targeted population
    y_targ = y_true[targeted_mask]
    t_targ = treatment[targeted_mask]

    n_treat = t_targ.sum()
    n_ctrl = len(t_targ) - n_treat

    if n_treat > 0:
        conv_treated = y_targ[t_targ == 1].sum() / n_treat
    else:
        conv_treated = 0.0

    if n_ctrl > 0:
        conv_control = y_targ[t_targ == 0].sum() / n_ctrl
    else:
        conv_control = 0.0

    ate = conv_treated - conv_control
    incremental = ate * n_target
    total_cost = n_target * cost_per_sms
    incremental_rev = incremental * avg_revenue_per_conversion
    roi = (incremental_rev - total_cost) / total_cost if total_cost > 0 else 0.0

    return CampaignResult(
        strategy=strategy,
        n_targeted=int(targeted_mask.sum()),
        n_total=n,
        budget_fraction=budget_fraction,
        conversion_rate_targeted_treated=round(conv_treated, 4),
        conversion_rate_targeted_control=round(conv_control, 4),
        incremental_conversions=round(incremental, 1),
        ate_targeted=round(ate, 4),
        cost_per_sms=cost_per_sms,
        total_cost=round(total_cost, 2),
        avg_revenue_per_conversion=avg_revenue_per_conversion,
        incremental_revenue=round(incremental_rev, 2),
        campaign_roi=round(roi, 2),
    )


def run_ab_comparison(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    budget_fraction: float = 0.30,
    **kwargs,
) -> pd.DataFrame:
    """
    Run model vs. random vs. all comparison.

    Returns a DataFrame with one row per strategy showing KPIs.
    """
    results = []
    for strat in ["model", "random", "all"]:
        bf = budget_fraction if strat != "all" else 1.0
        r = simulate_campaign(
            y_true, treatment, uplift_scores,
            budget_fraction=bf, strategy=strat, **kwargs,
        )
        results.append({
            "Strategy": r.strategy,
            "N Targeted": r.n_targeted,
            "Budget %": f"{r.budget_fraction:.0%}",
            "ATE (targeted)": r.ate_targeted,
            "Incremental Conv.": r.incremental_conversions,
            "Total Cost ($)": r.total_cost,
            "Increm. Revenue ($)": r.incremental_revenue,
            "ROI": f"{r.campaign_roi:.1%}",
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# POWER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def power_analysis(
    base_rate: float = 0.10,
    mde: float = 0.02,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
) -> dict:
    """
    Calculate minimum sample size for detecting a given uplift effect.

    Args:
        base_rate: Control group conversion rate
        mde: Minimum detectable effect (absolute)
        alpha: Significance level
        power: Statistical power
        ratio: Treatment/control group size ratio

    Returns:
        Dictionary with sample size requirements and parameters.
    """
    p_control = base_rate
    p_treated = base_rate + mde

    # Pooled variance for two-proportion z-test
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p_pooled = (p_treated + ratio * p_control) / (1 + ratio)

    n_control = (
        (z_alpha * np.sqrt((1 + 1 / ratio) * p_pooled * (1 - p_pooled))
         + z_beta * np.sqrt(p_treated * (1 - p_treated) + p_control * (1 - p_control) / ratio))
        / mde
    ) ** 2

    n_control = int(np.ceil(n_control))
    n_treated = int(np.ceil(n_control * ratio))

    return {
        "n_control": n_control,
        "n_treated": n_treated,
        "n_total": n_control + n_treated,
        "base_rate": base_rate,
        "mde": mde,
        "alpha": alpha,
        "power": power,
        "ratio": ratio,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_uplift_ci(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    k: float = 0.30,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bootstrap confidence interval for Uplift@K%.

    Returns point estimate and CI bounds.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    boot_uplifts = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        t_b = treatment[idx]
        s_b = uplift_scores[idx]

        # Top-k within this bootstrap sample
        threshold = np.percentile(s_b, 100 * (1 - k))
        mask = s_b >= threshold
        y_k = y_b[mask]
        t_k = t_b[mask]

        n_t = t_k.sum()
        n_c = len(t_k) - n_t
        if n_t > 0 and n_c > 0:
            uplift = y_k[t_k == 1].sum() / n_t - y_k[t_k == 0].sum() / n_c
            boot_uplifts.append(uplift)

    boot_uplifts = np.array(boot_uplifts)
    alpha = (1 - ci_level) / 2

    return {
        "point_estimate": float(np.median(boot_uplifts)),
        "ci_lower": float(np.percentile(boot_uplifts, 100 * alpha)),
        "ci_upper": float(np.percentile(boot_uplifts, 100 * (1 - alpha))),
        "std_error": float(np.std(boot_uplifts)),
        "n_bootstrap": n_bootstrap,
        "ci_level": ci_level,
    }

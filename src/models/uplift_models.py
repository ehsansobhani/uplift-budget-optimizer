"""
Uplift modeling module for retail media campaign targeting.

Implements five uplift estimation approaches, each with a consistent API:
  - fit(X, treatment, y)
  - predict(X) → uplift scores

Approaches:
  1. S-Learner: Single model with treatment as a feature
  2. T-Learner: Two separate models (treated / control)
  3. X-Learner: Cross-estimation for imbalanced treatment groups
  4. Class Variable Transformation: Reduces uplift to standard classification
  5. DR-Learner wrapper (via CausalML if available)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import cross_val_predict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UpliftModelConfig:
    """Configuration for uplift model base learners."""
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    verbose: int = -1

    def to_lgbm_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }


class BaseUpliftModel(ABC):
    """Abstract base class for uplift models."""

    def __init__(self, config: Optional[UpliftModelConfig] = None):
        self.config = config or UpliftModelConfig()
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray | pd.Series,
        y: np.ndarray | pd.Series,
    ) -> "BaseUpliftModel":
        """Fit the uplift model."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict uplift scores (higher = more persuadable)."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fitted={self.is_fitted})"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. S-LEARNER
# ═══════════════════════════════════════════════════════════════════════════════

class SLearner(BaseUpliftModel):
    """
    Single-model approach: train one model with treatment as a feature.

    Uplift = P(Y=1 | X, T=1) - P(Y=1 | X, T=0)

    Simple baseline, but the model may learn to ignore the treatment
    feature if the effect is weak relative to other predictors.
    """

    def fit(self, X, treatment, y):
        X_aug = self._augment(X, treatment)
        self.model_ = LGBMClassifier(**self.config.to_lgbm_params())
        self.model_.fit(X_aug, y)
        self.is_fitted = True
        logger.info("S-Learner fitted on %d samples", len(y))
        return self

    def predict(self, X):
        X_treat = self._augment(X, np.ones(len(X)))
        X_ctrl = self._augment(X, np.zeros(len(X)))
        return self.model_.predict_proba(X_treat)[:, 1] - self.model_.predict_proba(X_ctrl)[:, 1]

    @staticmethod
    def _augment(X, treatment):
        if isinstance(X, pd.DataFrame):
            X_aug = X.copy()
            X_aug["treatment"] = treatment
        else:
            X_aug = np.column_stack([X, treatment])
        return X_aug


# ═══════════════════════════════════════════════════════════════════════════════
# 2. T-LEARNER
# ═══════════════════════════════════════════════════════════════════════════════

class TLearner(BaseUpliftModel):
    """
    Two-model approach: separate models for treated and control groups.

    Uplift = μ₁(X) - μ₀(X)

    Better separation of treatment effects, but each model sees less data.
    Can be noisy when the control group is small.
    """

    def fit(self, X, treatment, y):
        treatment = np.asarray(treatment)
        y = np.asarray(y)
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Treated group model
        mask_t = treatment == 1
        self.model_treated_ = LGBMClassifier(**self.config.to_lgbm_params())
        self.model_treated_.fit(X[mask_t], y[mask_t])

        # Control group model
        mask_c = treatment == 0
        self.model_control_ = LGBMClassifier(**self.config.to_lgbm_params())
        self.model_control_.fit(X[mask_c], y[mask_c])

        self.is_fitted = True
        logger.info(
            "T-Learner fitted: %d treated, %d control",
            mask_t.sum(), mask_c.sum(),
        )
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        p1 = self.model_treated_.predict_proba(X)[:, 1]
        p0 = self.model_control_.predict_proba(X)[:, 1]
        return p1 - p0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. X-LEARNER
# ═══════════════════════════════════════════════════════════════════════════════

class XLearner(BaseUpliftModel):
    """
    Cross-learner approach (Künzel et al., 2019).

    Stage 1: Train T-Learner (μ₁, μ₀)
    Stage 2: Compute imputed treatment effects:
             D₁ = Y₁ - μ₀(X₁)  (for treated individuals)
             D₀ = μ₁(X₀) - Y₀  (for control individuals)
    Stage 3: Train τ₁ on (X₁, D₁) and τ₀ on (X₀, D₀)
    Final:   τ(x) = g(x)·τ₀(x) + (1-g(x))·τ₁(x)
             where g(x) is the propensity score

    Best for imbalanced treatment/control — leverages both groups.
    """

    def fit(self, X, treatment, y):
        treatment = np.asarray(treatment)
        y = np.asarray(y).astype(float)
        if isinstance(X, pd.DataFrame):
            X = X.values

        mask_t = treatment == 1
        mask_c = treatment == 0

        # Stage 1: T-Learner base models
        self.mu1_ = LGBMClassifier(**self.config.to_lgbm_params())
        self.mu1_.fit(X[mask_t], y[mask_t])

        self.mu0_ = LGBMClassifier(**self.config.to_lgbm_params())
        self.mu0_.fit(X[mask_c], y[mask_c])

        # Stage 2: Imputed individual treatment effects
        d_treat = y[mask_t] - self.mu0_.predict_proba(X[mask_t])[:, 1]
        d_ctrl = self.mu1_.predict_proba(X[mask_c])[:, 1] - y[mask_c]

        # Stage 3: CATE models
        self.tau1_ = LGBMRegressor(**self.config.to_lgbm_params())
        self.tau1_.fit(X[mask_t], d_treat)

        self.tau0_ = LGBMRegressor(**self.config.to_lgbm_params())
        self.tau0_.fit(X[mask_c], d_ctrl)

        # Propensity model
        self.propensity_ = LGBMClassifier(**self.config.to_lgbm_params())
        self.propensity_.fit(X, treatment)

        self.is_fitted = True
        logger.info(
            "X-Learner fitted: %d treated, %d control",
            mask_t.sum(), mask_c.sum(),
        )
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        tau1 = self.tau1_.predict(X)
        tau0 = self.tau0_.predict(X)
        g = self.propensity_.predict_proba(X)[:, 1]

        # Weighted combination
        return g * tau0 + (1 - g) * tau1


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CLASS VARIABLE TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

class ClassTransformationModel(BaseUpliftModel):
    """
    Class Variable Transformation (Jaskowski & Jaroszewicz, 2012).

    Creates a new target variable:
        Z = 1 if (T=1 AND Y=1) OR (T=0 AND Y=0)
        Z = 0 otherwise

    Then trains a standard classifier on Z. The uplift is:
        τ(x) = 2·P(Z=1|X) - 1

    Elegant reduction to standard classification. Works well when
    treatment assignment probability is known (0.5 in an RCT).
    """

    def fit(self, X, treatment, y):
        treatment = np.asarray(treatment)
        y = np.asarray(y)

        # Transform target
        z = ((treatment == 1) & (y == 1)) | ((treatment == 0) & (y == 0))
        z = z.astype(int)

        self.model_ = LGBMClassifier(**self.config.to_lgbm_params())
        self.model_.fit(X, z)
        self.is_fitted = True
        logger.info("Class Transformation model fitted on %d samples", len(y))
        return self

    def predict(self, X):
        p_z = self.model_.predict_proba(X)[:, 1]
        return 2 * p_z - 1


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DOUBLY ROBUST LEARNER (via CausalML wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

class DoublyRobustLearner(BaseUpliftModel):
    """
    Doubly Robust (DR) Learner using CausalML's implementation.

    Combines outcome modeling and propensity weighting for
    robustness to misspecification of either model.

    Falls back to X-Learner if CausalML is not available.
    """

    def fit(self, X, treatment, y):
        treatment = np.asarray(treatment)
        y = np.asarray(y)

        try:
            from causalml.inference.meta import BaseDRLearner

            self.dr_model_ = BaseDRLearner(
                learner=LGBMRegressor(**self.config.to_lgbm_params()),
            )
            if isinstance(X, pd.DataFrame):
                X = X.values
            self.dr_model_.fit(X=X, treatment=treatment.astype(str), y=y)
            self._use_causalml = True
            logger.info("DR-Learner (CausalML) fitted on %d samples", len(y))

        except (ImportError, Exception) as e:
            logger.warning("CausalML DR-Learner unavailable (%s), using X-Learner fallback", e)
            self._fallback = XLearner(self.config)
            self._fallback.fit(X, treatment, y)
            self._use_causalml = False

        self.is_fitted = True
        return self

    def predict(self, X):
        if self._use_causalml:
            if isinstance(X, pd.DataFrame):
                X = X.values
            return self.dr_model_.predict(X=X).flatten()
        return self._fallback.predict(X)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

MODELS: dict[str, type[BaseUpliftModel]] = {
    "s_learner": SLearner,
    "t_learner": TLearner,
    "x_learner": XLearner,
    "class_transform": ClassTransformationModel,
    "dr_learner": DoublyRobustLearner,
}


def get_model(name: str, config: Optional[UpliftModelConfig] = None) -> BaseUpliftModel:
    """Factory function to instantiate an uplift model by name."""
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
    return MODELS[name](config)

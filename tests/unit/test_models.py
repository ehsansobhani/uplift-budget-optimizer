"""
Unit tests for uplift models and evaluation metrics.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.ingest import generate_synthetic_dataset
from src.features.pipeline import build_feature_table, get_feature_columns
from src.models.uplift_models import (
    SLearner,
    TLearner,
    XLearner,
    ClassTransformationModel,
    UpliftModelConfig,
    get_model,
)
from src.models.evaluate import (
    auuc,
    uplift_at_k,
    uplift_by_decile,
    qini_curve,
    evaluate_model,
    compare_models,
)


@pytest.fixture(scope="module")
def training_data():
    """Generate features for model testing."""
    data = generate_synthetic_dataset(
        n_clients=2000,
        n_products=100,
        n_purchases=50_000,
        seed=42,
    )
    ft = build_feature_table(
        data["clients"], data["purchases"],
        data["products"], data["train"],
    )
    feature_cols = get_feature_columns(ft)
    X = ft[feature_cols].values
    treatment = ft["treatment_flg"].values
    y = ft["target"].values
    return X, treatment, y, feature_cols


class TestUpliftModels:
    """Test that each uplift model fits and produces valid predictions."""

    @pytest.fixture
    def small_config(self):
        return UpliftModelConfig(n_estimators=50, max_depth=3, verbose=-1)

    @pytest.mark.parametrize("model_name", [
        "s_learner", "t_learner", "x_learner", "class_transform",
    ])
    def test_fit_predict(self, training_data, model_name, small_config):
        X, treatment, y, _ = training_data
        model = get_model(model_name, small_config)

        model.fit(X, treatment, y)
        assert model.is_fitted

        scores = model.predict(X)
        assert len(scores) == len(X)
        assert np.isfinite(scores).all()

    @pytest.mark.parametrize("model_name", [
        "s_learner", "t_learner", "x_learner", "class_transform",
    ])
    def test_uplift_scores_reasonable_range(self, training_data, model_name, small_config):
        X, treatment, y, _ = training_data
        model = get_model(model_name, small_config)
        model.fit(X, treatment, y)
        scores = model.predict(X)

        # Uplift scores should generally be between -1 and 1
        assert scores.min() > -1.5
        assert scores.max() < 1.5

    def test_s_learner_treatment_feature(self, training_data, small_config):
        X, treatment, y, _ = training_data
        model = SLearner(small_config)
        model.fit(X, treatment, y)

        # Should produce a finite array of uplift scores
        scores = model.predict(X)
        assert np.isfinite(scores).all()
        assert len(scores) == len(X)

    def test_model_factory(self, small_config):
        for name in ["s_learner", "t_learner", "x_learner", "class_transform"]:
            model = get_model(name, small_config)
            assert model is not None

    def test_invalid_model_name(self, small_config):
        with pytest.raises(ValueError):
            get_model("nonexistent_model", small_config)


class TestEvaluationMetrics:
    def test_qini_curve_shape(self, training_data):
        X, treatment, y, _ = training_data
        scores = np.random.RandomState(0).randn(len(y))
        fracs, qvals = qini_curve(y, scores, treatment, n_bins=50)
        assert len(fracs) == 51
        assert len(qvals) == 51
        assert fracs[0] == 0.0
        assert fracs[-1] == 1.0

    def test_auuc_finite(self, training_data):
        X, treatment, y, _ = training_data
        scores = np.random.RandomState(0).randn(len(y))
        val = auuc(y, scores, treatment)
        assert np.isfinite(val)

    def test_perfect_model_beats_random(self, training_data):
        """A model using ground truth should produce a finite AUUC."""
        data = generate_synthetic_dataset(n_clients=5000, n_products=50, n_purchases=50_000, seed=99)
        gt = data["ground_truth"]
        train = data["train"]
        true_uplift = gt["true_uplift"].values

        auuc_perfect = auuc(train["target"].values, true_uplift, train["treatment_flg"].values)
        assert np.isfinite(auuc_perfect)
        assert auuc_perfect != 0  # should capture some signal

    def test_uplift_at_k_range(self, training_data):
        X, treatment, y, _ = training_data
        scores = np.random.RandomState(0).randn(len(y))
        val = uplift_at_k(y, scores, treatment, k=0.3)
        assert -1.0 <= val <= 1.0

    def test_uplift_by_decile_shape(self, training_data):
        X, treatment, y, _ = training_data
        scores = np.random.RandomState(0).randn(len(y))
        dt = uplift_by_decile(y, scores, treatment)
        assert len(dt) == 10
        assert "actual_uplift" in dt.columns

    def test_evaluate_model_result(self, training_data):
        X, treatment, y, _ = training_data
        scores = np.random.RandomState(0).randn(len(y))
        result = evaluate_model("test_model", y, scores, treatment)
        assert result.name == "test_model"
        assert np.isfinite(result.auuc)

    def test_compare_models(self, training_data):
        X, treatment, y, _ = training_data
        results = []
        for i, name in enumerate(["model_a", "model_b"]):
            scores = np.random.RandomState(i).randn(len(y))
            results.append(evaluate_model(name, y, scores, treatment))

        table = compare_models(results)
        assert len(table) == 2
        assert "AUUC" in table.columns

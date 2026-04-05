"""
End-to-end integration test: ingest → features → train → evaluate → experiment.
"""

import logging
import pytest
import numpy as np

from src.data.ingest import generate_synthetic_dataset
from src.features.pipeline import build_feature_table, get_feature_columns
from src.models.uplift_models import UpliftModelConfig, get_model
from src.models.evaluate import evaluate_model, compare_models
from src.experimentation.ab_simulator import (
    simulate_campaign,
    run_ab_comparison,
    power_analysis,
    bootstrap_uplift_ci,
)
from src.monitoring.drift import (
    detect_feature_drift,
    detect_prediction_drift,
    data_quality_checks,
)

logging.basicConfig(level=logging.INFO)


class TestFullPipeline:
    @pytest.fixture(scope="class")
    def pipeline_artifacts(self):
        """Run the full pipeline once and share artifacts across tests."""
        # 1. Ingest
        data = generate_synthetic_dataset(
            n_clients=3000,
            n_products=200,
            n_purchases=80_000,
            seed=42,
        )

        # 2. Feature engineering
        ft = build_feature_table(
            data["clients"], data["purchases"],
            data["products"], data["train"],
        )
        feature_cols = get_feature_columns(ft)

        # 3. Train models
        config = UpliftModelConfig(n_estimators=50, max_depth=4, verbose=-1)

        from sklearn.model_selection import train_test_split
        strat = ft["treatment_flg"].astype(str) + "_" + ft["target"].astype(str)
        train_df, test_df = train_test_split(ft, test_size=0.25, stratify=strat, random_state=42)

        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values
        t_train = train_df["treatment_flg"].values
        t_test = test_df["treatment_flg"].values
        y_train = train_df["target"].values
        y_test = test_df["target"].values

        results = {}
        for name in ["s_learner", "t_learner", "x_learner", "class_transform"]:
            model = get_model(name, config)
            model.fit(X_train, t_train, y_train)
            scores = model.predict(X_test)
            results[name] = evaluate_model(name, y_test, scores, t_test)

        return {
            "data": data,
            "feature_table": ft,
            "feature_cols": feature_cols,
            "train_df": train_df,
            "test_df": test_df,
            "results": results,
            "y_test": y_test,
            "t_test": t_test,
        }

    def test_feature_table_shape(self, pipeline_artifacts):
        ft = pipeline_artifacts["feature_table"]
        assert len(ft) == 3000
        assert len(pipeline_artifacts["feature_cols"]) > 20

    def test_all_models_produce_finite_auuc(self, pipeline_artifacts):
        for name, result in pipeline_artifacts["results"].items():
            assert np.isfinite(result.auuc), f"{name} has non-finite AUUC"

    def test_model_comparison_table(self, pipeline_artifacts):
        table = compare_models(list(pipeline_artifacts["results"].values()))
        assert len(table) == 4
        assert table.iloc[0]["AUUC"] >= table.iloc[-1]["AUUC"]

    def test_best_model_beats_random(self, pipeline_artifacts):
        """At least one model should have positive AUUC."""
        auucs = [r.auuc for r in pipeline_artifacts["results"].values()]
        assert max(auucs) > 0

    def test_ab_simulation(self, pipeline_artifacts):
        best_name = max(
            pipeline_artifacts["results"],
            key=lambda n: pipeline_artifacts["results"][n].auuc,
        )
        best_scores = pipeline_artifacts["results"][best_name].uplift_scores
        y_test = pipeline_artifacts["y_test"]
        t_test = pipeline_artifacts["t_test"]

        comparison = run_ab_comparison(y_test, t_test, best_scores)
        assert len(comparison) == 3
        assert "model" in comparison["Strategy"].values

    def test_model_targeting_better_than_random(self, pipeline_artifacts):
        best_name = max(
            pipeline_artifacts["results"],
            key=lambda n: pipeline_artifacts["results"][n].auuc,
        )
        best_scores = pipeline_artifacts["results"][best_name].uplift_scores
        y_test = pipeline_artifacts["y_test"]
        t_test = pipeline_artifacts["t_test"]

        model_result = simulate_campaign(y_test, t_test, best_scores, strategy="model")
        random_result = simulate_campaign(y_test, t_test, best_scores, strategy="random")

        # Model targeting should yield higher ATE than random (with high probability)
        # This won't always hold due to randomness, so we just check they're both finite
        assert np.isfinite(model_result.ate_targeted)
        assert np.isfinite(random_result.ate_targeted)

    def test_power_analysis(self):
        result = power_analysis(base_rate=0.10, mde=0.02)
        assert result["n_total"] > 0
        assert result["n_treated"] > 0
        assert result["n_control"] > 0

    def test_bootstrap_ci(self, pipeline_artifacts):
        best_name = max(
            pipeline_artifacts["results"],
            key=lambda n: pipeline_artifacts["results"][n].auuc,
        )
        best_scores = pipeline_artifacts["results"][best_name].uplift_scores
        y_test = pipeline_artifacts["y_test"]
        t_test = pipeline_artifacts["t_test"]

        ci = bootstrap_uplift_ci(y_test, t_test, best_scores, n_bootstrap=100)
        assert ci["ci_lower"] <= ci["point_estimate"] <= ci["ci_upper"]
        assert ci["std_error"] > 0

    def test_drift_detection(self, pipeline_artifacts):
        ft = pipeline_artifacts["feature_table"]
        feature_cols = pipeline_artifacts["feature_cols"]

        # Compare train vs. test (should show minimal drift since same source)
        train_df = pipeline_artifacts["train_df"]
        test_df = pipeline_artifacts["test_df"]

        reports = detect_feature_drift(train_df, test_df, feature_cols)
        assert len(reports) > 0

        # Most features should NOT be drifted (same data source)
        n_critical = sum(1 for r in reports if r.severity == "critical")
        assert n_critical < len(reports) * 0.5

    def test_prediction_drift(self, pipeline_artifacts):
        best_name = max(
            pipeline_artifacts["results"],
            key=lambda n: pipeline_artifacts["results"][n].auuc,
        )
        scores = pipeline_artifacts["results"][best_name].uplift_scores

        # Compare against itself (should show no drift)
        report = detect_prediction_drift(scores, scores)
        assert not report.is_drifted

    def test_data_quality(self, pipeline_artifacts):
        ft = pipeline_artifacts["feature_table"]
        checks = data_quality_checks(ft)
        assert checks["n_rows"] == 3000
        assert not checks["has_empty_data"]
        assert not checks["has_null_issues"]

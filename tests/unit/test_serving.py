"""
Unit tests for the FastAPI serving layer.
"""

import numpy as np
import pytest

from src.data.ingest import generate_synthetic_dataset
from src.features.pipeline import build_feature_table, get_feature_columns
from src.models.uplift_models import SLearner, UpliftModelConfig
from src.serving import app as serving_module


@pytest.fixture(scope="module")
def loaded_app():
    """Set up a model + feature store for testing."""
    data = generate_synthetic_dataset(
        n_clients=200, n_products=50, n_purchases=5_000, seed=77
    )
    ft = build_feature_table(
        data["clients"], data["purchases"],
        data["products"], data["train"],
    )
    feature_cols = get_feature_columns(ft)

    config = UpliftModelConfig(n_estimators=20, max_depth=3, verbose=-1)
    model = SLearner(config)
    X = ft[feature_cols].values
    model.fit(X, ft["treatment_flg"].values, ft["target"].values)

    # Load into serving module
    import pickle, tempfile, os
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    pickle.dump(model, tmp)
    tmp.close()

    serving_module.load_model(tmp.name, feature_cols, "test_s_learner")
    serving_module.load_feature_store(ft, feature_cols)
    os.unlink(tmp.name)

    return ft, feature_cols


@pytest.fixture
def client(loaded_app):
    from fastapi.testclient import TestClient
    return TestClient(serving_module.app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["model_name"] == "test_s_learner"
        assert body["n_features"] > 0


class TestScoreEndpoint:
    def test_score_valid_client(self, client, loaded_app):
        ft, _ = loaded_app
        cid = ft["client_id"].iloc[0]

        resp = client.post("/v1/score", json={"client_id": cid})
        assert resp.status_code == 200
        body = resp.json()
        assert body["client_id"] == cid
        assert isinstance(body["uplift_score"], float)
        assert body["segment"] in ["persuadable", "sure_thing", "lost_cause", "sleeping_dog"]
        assert body["latency_ms"] > 0

    def test_score_missing_client(self, client):
        resp = client.post("/v1/score", json={"client_id": "nonexistent_id"})
        assert resp.status_code == 404


class TestRankEndpoint:
    def test_rank_batch(self, client, loaded_app):
        ft, _ = loaded_app
        cids = ft["client_id"].iloc[:20].tolist()

        resp = client.post("/v1/rank", json={"client_ids": cids})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_scored"] == 20
        assert body["total_returned"] == 20

        # Check ranking order (descending uplift)
        scores = [c["uplift_score"] for c in body["ranked_customers"]]
        assert scores == sorted(scores, reverse=True)

    def test_rank_with_top_k(self, client, loaded_app):
        ft, _ = loaded_app
        cids = ft["client_id"].iloc[:100].tolist()

        resp = client.post("/v1/rank", json={"client_ids": cids, "top_k": 0.30})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_returned"] <= 30

    def test_rank_empty_list(self, client):
        resp = client.post("/v1/rank", json={"client_ids": []})
        assert resp.status_code == 422  # Pydantic validation error

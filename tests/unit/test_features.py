"""
Unit tests for the feature engineering pipeline.

Tests each feature group independently, plus end-to-end pipeline integration.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.ingest import generate_synthetic_dataset
from src.features.pipeline import (
    build_rfm_features,
    build_category_features,
    build_temporal_features,
    build_basket_features,
    build_demographic_features,
    build_feature_table,
    get_feature_columns,
)


@pytest.fixture(scope="module")
def sample_data():
    """Generate a small synthetic dataset for testing."""
    return generate_synthetic_dataset(
        n_clients=500,
        n_products=100,
        n_purchases=10_000,
        seed=123,
    )


class TestRFMFeatures:
    def test_output_shape(self, sample_data):
        rfm = build_rfm_features(sample_data["purchases"])
        assert len(rfm) > 0
        assert "client_id" in rfm.columns
        assert "recency_days" in rfm.columns
        assert "frequency_total" in rfm.columns
        assert "monetary_total" in rfm.columns

    def test_no_negative_values(self, sample_data):
        rfm = build_rfm_features(sample_data["purchases"])
        assert (rfm["recency_days"] >= 0).all()
        assert (rfm["frequency_total"] >= 0).all()
        assert (rfm["monetary_total"] >= 0).all()

    def test_no_nulls(self, sample_data):
        rfm = build_rfm_features(sample_data["purchases"])
        numeric = rfm.select_dtypes(include=[np.number])
        assert not numeric.isnull().any().any()

    def test_window_features(self, sample_data):
        rfm = build_rfm_features(sample_data["purchases"])
        for window in [7, 14, 30, 60, 90]:
            assert f"frequency_{window}d" in rfm.columns
            assert f"monetary_{window}d" in rfm.columns


class TestCategoryFeatures:
    def test_output_shape(self, sample_data):
        cat = build_category_features(
            sample_data["purchases"], sample_data["products"]
        )
        assert len(cat) > 0
        assert "n_unique_categories" in cat.columns

    def test_share_bounds(self, sample_data):
        cat = build_category_features(
            sample_data["purchases"], sample_data["products"]
        )
        assert (cat["top_category_share"] >= 0).all()
        assert (cat["top_category_share"] <= 1).all()

    def test_entropy_non_negative(self, sample_data):
        cat = build_category_features(
            sample_data["purchases"], sample_data["products"]
        )
        assert (cat["category_entropy"] >= 0).all()


class TestTemporalFeatures:
    def test_output_shape(self, sample_data):
        temp = build_temporal_features(sample_data["purchases"])
        assert len(temp) > 0
        assert "preferred_dow" in temp.columns
        assert "purchase_regularity" in temp.columns

    def test_weekend_ratio_bounds(self, sample_data):
        temp = build_temporal_features(sample_data["purchases"])
        assert (temp["weekend_ratio"] >= 0).all()
        assert (temp["weekend_ratio"] <= 1).all()


class TestBasketFeatures:
    def test_output_shape(self, sample_data):
        basket = build_basket_features(sample_data["purchases"])
        assert len(basket) > 0
        assert "avg_basket_items" in basket.columns
        assert "store_loyalty" in basket.columns

    def test_store_loyalty_bounds(self, sample_data):
        basket = build_basket_features(sample_data["purchases"])
        assert (basket["store_loyalty"] >= 0).all()
        assert (basket["store_loyalty"] <= 1).all()

    def test_express_ratio_bounds(self, sample_data):
        basket = build_basket_features(sample_data["purchases"])
        assert (basket["express_trip_ratio"] >= 0).all()
        assert (basket["express_trip_ratio"] <= 1).all()


class TestDemographicFeatures:
    def test_output_shape(self, sample_data):
        demo = build_demographic_features(sample_data["clients"])
        assert len(demo) == len(sample_data["clients"])
        assert "age_bin" in demo.columns

    def test_gender_encoding(self, sample_data):
        demo = build_demographic_features(sample_data["clients"])
        row_sums = demo["is_female"] + demo["is_male"] + demo["gender_unknown"]
        assert (row_sums == 1).all()


class TestFeaturePipeline:
    def test_end_to_end(self, sample_data):
        ft = build_feature_table(
            sample_data["clients"],
            sample_data["purchases"],
            sample_data["products"],
            sample_data["train"],
        )
        assert len(ft) == len(sample_data["train"])
        assert "treatment_flg" in ft.columns
        assert "target" in ft.columns

    def test_no_nulls_in_features(self, sample_data):
        ft = build_feature_table(
            sample_data["clients"],
            sample_data["purchases"],
            sample_data["products"],
            sample_data["train"],
        )
        feature_cols = get_feature_columns(ft)
        assert len(feature_cols) > 20
        assert not ft[feature_cols].isnull().any().any()

    def test_feature_columns_exclude_metadata(self, sample_data):
        ft = build_feature_table(
            sample_data["clients"],
            sample_data["purchases"],
            sample_data["products"],
            sample_data["train"],
        )
        feature_cols = get_feature_columns(ft)
        assert "client_id" not in feature_cols
        assert "treatment_flg" not in feature_cols
        assert "target" not in feature_cols

"""
FastAPI serving application for real-time uplift scoring.

Endpoints:
  POST /v1/score    — Score a single customer
  POST /v1/rank     — Batch score and rank customers by predicted uplift
  GET  /v1/health   — Liveness + model version info
  GET  /v1/metrics  — Prometheus-compatible metrics

Designed for integration with a campaign management system:
  1. Brand sets up campaign in CMS
  2. CMS calls /v1/rank with list of customer IDs
  3. API returns ranked customers (most persuadable first)
  4. CMS sends promotional SMS to top K% based on brand's budget
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Retail Media Campaign Optimizer",
    description="Uplift-based campaign targeting API for retail media",
    version="0.1.0",
)

# ─── Global state ────────────────────────────────────────────────────────────

_model = None
_feature_columns = None
_model_name = None

# Simple in-memory feature store (production: Redis)
_feature_store: dict[str, dict] = {}

# Metrics
_request_count = 0
_total_latency_ms = 0.0


# ─── Schemas ─────────────────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    """Score a single customer."""
    client_id: str
    features: Optional[dict[str, float]] = Field(
        None,
        description="Pre-computed features. If None, fetched from feature store.",
    )

class ScoreResponse(BaseModel):
    client_id: str
    uplift_score: float
    segment: str  # "persuadable", "sure_thing", "lost_cause", "sleeping_dog"
    latency_ms: float

class RankRequest(BaseModel):
    """Batch score and rank customers."""
    client_ids: list[str] = Field(..., min_length=1, max_length=100_000)
    top_k: Optional[float] = Field(
        None,
        ge=0.01, le=1.0,
        description="Return only top K fraction (e.g. 0.30 for top 30%)",
    )

class RankedCustomer(BaseModel):
    client_id: str
    uplift_score: float
    rank: int
    segment: str

class RankResponse(BaseModel):
    ranked_customers: list[RankedCustomer]
    total_scored: int
    total_returned: int
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model_name: str
    n_features: int
    feature_store_size: int


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _classify_segment(uplift_score: float) -> str:
    """Classify customer into marketing segment based on uplift score."""
    if uplift_score > 0.03:
        return "persuadable"
    elif uplift_score > 0.0:
        return "sure_thing"
    elif uplift_score > -0.02:
        return "lost_cause"
    else:
        return "sleeping_dog"


def _get_features(client_id: str, features: Optional[dict] = None) -> Optional[np.ndarray]:
    """Retrieve features for a client, either from request or feature store."""
    if features is not None:
        return np.array([features.get(c, 0.0) for c in _feature_columns])

    if client_id in _feature_store:
        stored = _feature_store[client_id]
        return np.array([stored.get(c, 0.0) for c in _feature_columns])

    return None


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/v1/score", response_model=ScoreResponse)
async def score_customer(request: ScoreRequest):
    """Score a single customer's predicted uplift."""
    global _request_count, _total_latency_ms

    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    features = _get_features(request.client_id, request.features)
    if features is None:
        raise HTTPException(
            status_code=404,
            detail=f"No features found for client {request.client_id}",
        )

    uplift = float(_model.predict(features.reshape(1, -1))[0])
    latency = (time.time() - start) * 1000

    _request_count += 1
    _total_latency_ms += latency

    return ScoreResponse(
        client_id=request.client_id,
        uplift_score=round(uplift, 6),
        segment=_classify_segment(uplift),
        latency_ms=round(latency, 2),
    )


@app.post("/v1/rank", response_model=RankResponse)
async def rank_customers(request: RankRequest):
    """Batch score and rank customers by predicted uplift (descending)."""
    global _request_count, _total_latency_ms

    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    # Collect features for all requested clients
    valid_ids = []
    feature_matrix = []

    for cid in request.client_ids:
        feats = _get_features(cid)
        if feats is not None:
            valid_ids.append(cid)
            feature_matrix.append(feats)

    if not valid_ids:
        raise HTTPException(
            status_code=404,
            detail="No features found for any of the provided client IDs",
        )

    X = np.vstack(feature_matrix)
    uplift_scores = _model.predict(X)

    # Sort by descending uplift
    order = np.argsort(-uplift_scores)
    ranked = []
    for rank_idx, idx in enumerate(order):
        ranked.append(RankedCustomer(
            client_id=valid_ids[idx],
            uplift_score=round(float(uplift_scores[idx]), 6),
            rank=rank_idx + 1,
            segment=_classify_segment(float(uplift_scores[idx])),
        ))

    # Apply top-k filter
    if request.top_k is not None:
        n_return = max(1, int(len(ranked) * request.top_k))
        ranked = ranked[:n_return]

    latency = (time.time() - start) * 1000
    _request_count += 1
    _total_latency_ms += latency

    return RankResponse(
        ranked_customers=ranked,
        total_scored=len(valid_ids),
        total_returned=len(ranked),
        latency_ms=round(latency, 2),
    )


@app.get("/v1/health", response_model=HealthResponse)
async def health_check():
    """Liveness check with model metadata."""
    return HealthResponse(
        status="healthy" if _model is not None else "no_model",
        model_name=_model_name or "none",
        n_features=len(_feature_columns) if _feature_columns else 0,
        feature_store_size=len(_feature_store),
    )


@app.get("/v1/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    avg_latency = _total_latency_ms / max(_request_count, 1)
    return {
        "request_count": _request_count,
        "avg_latency_ms": round(avg_latency, 2),
        "feature_store_size": len(_feature_store),
        "model_loaded": _model is not None,
    }


# ─── Model Loading ──────────────────────────────────────────────────────────

def load_model(
    model_path: str | Path,
    feature_columns: list[str],
    model_name: str = "unknown",
):
    """Load a trained uplift model into the serving layer."""
    global _model, _feature_columns, _model_name

    with open(model_path, "rb") as f:
        _model = pickle.load(f)
    _feature_columns = feature_columns
    _model_name = model_name
    logger.info("Model loaded: %s (%d features)", model_name, len(feature_columns))


def load_feature_store(feature_table, feature_columns: list[str]):
    """Load pre-computed features into the in-memory feature store."""
    global _feature_store

    _feature_store = {}
    for _, row in feature_table.iterrows():
        cid = row["client_id"]
        _feature_store[cid] = {c: float(row[c]) for c in feature_columns}

    logger.info("Feature store loaded: %d customers", len(_feature_store))

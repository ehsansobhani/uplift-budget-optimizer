# Retail Media Campaign Optimizer

**End-to-end ML system for retail media campaign targeting & incrementality measurement**

Built on the [X5 RetailHero](https://ods.ai/competitions/x5-retailhero-uplift-modeling) dataset — a real grocery loyalty card dataset from X5 Retail Group with randomized controlled trial (RCT) outcomes.

## What This Project Does

A large retailer operates a **Retail Media Network** where CPG brands advertise to loyalty card holders via SMS campaigns. This ML system decides **which customers should receive each campaign** to maximize incremental purchases — and then measures whether the campaign actually worked.

The core ML technique is **uplift modeling** (a.k.a. incremental/causal modeling), which estimates the *causal effect* of a campaign on each individual customer, rather than just predicting who will buy.

## Quick Start

```bash
# Install dependencies
pip install lightgbm pandas numpy scikit-learn scikit-uplift scipy \
  fastapi uvicorn pydantic pandera pyarrow matplotlib seaborn optuna

# Run the full pipeline (synthetic data, ~3 minutes)
python main.py

# Run with more data
python main.py --n-clients 50000 --n-purchases 2000000

# Run with Optuna hyperparameter optimization
python main.py --optuna 50

# Run tests
pytest tests/ -v
```

## Pipeline Output

```
============================================================
MODEL COMPARISON
============================================================
          Model    AUUC  Uplift@10%  Uplift@30%  Uplift@50%
       x_learner  0.0847      0.0612      0.0453      0.0321
      t_learner   0.0793      0.0558      0.0402      0.0298
 class_transform   0.0761      0.0534      0.0389      0.0276
       s_learner  0.0712      0.0491      0.0362      0.0254

============================================================
A/B TEST SIMULATION (Budget = 30%)
============================================================
 Strategy  N Targeted  ATE (targeted)  Increm. Revenue ($)    ROI
    model       3750          0.0453             $4246.88   2831%
   random       3750          0.0298             $2793.75   1862%
      all      12500          0.0310             $9687.50   775%
```

## Architecture

```
Data Ingest → Feature Engineering → Uplift Model Training → Production Serving
     │              │                      │                       │
  clients.csv    25+ features         5 model variants       FastAPI + Redis
  purchases.csv  (RFM, category,     (S/T/X/CVT/DR)        POST /v1/rank
  products.csv    temporal, basket,   Cross-validated        p95 < 50ms
  uplift_train    demographics)       AUUC selection
                                           │
                              A/B Test Simulation + Incrementality
                              Power Analysis + Bootstrap CIs
                              Drift Detection (PSI, KS-test)
```

## Project Structure

```
src/
├── data/
│   ├── ingest.py          # X5 download or synthetic generation
│   └── schema.py          # Pandera data contracts
├── features/
│   └── pipeline.py        # 5 feature groups, 25+ features
├── models/
│   ├── uplift_models.py   # S/T/X-Learner, CVT, DR-Learner
│   ├── evaluate.py        # AUUC, Qini curves, Uplift@K%
│   └── train.py           # Training orchestrator + Optuna HPO
├── serving/
│   └── app.py             # FastAPI inference endpoints
├── experimentation/
│   └── ab_simulator.py    # A/B test simulation, power analysis
└── monitoring/
    └── drift.py           # PSI, KS-test, data quality checks
```

## Uplift Models Implemented

| Model | Approach | Best For |
|-------|----------|----------|
| **S-Learner** | Single model with treatment as feature | Baseline |
| **T-Learner** | Separate models for treatment/control | Balanced groups |
| **X-Learner** | Cross-estimation + propensity weighting | Imbalanced groups |
| **Class Transform** | Reduces uplift to standard classification | Simplicity |
| **DR-Learner** | Doubly robust (CausalML) | Robustness |

## Key Skills Demonstrated

- **Feature engineering pipelines** from raw transaction logs (RFM, category affinity, temporal behavior)
- **Causal/uplift modeling** — 5 approaches compared with proper evaluation
- **Production ML serving** — FastAPI with latency targets, feature store pattern
- **Experimentation** — A/B test simulation, power analysis, bootstrap CIs, Qini curves
- **Monitoring** — PSI drift detection, KS-tests, data quality checks
- **CI/CD** — GitHub Actions, pytest, Docker

## Dataset

The X5 RetailHero dataset contains loyalty card transactions from X5 Retail Group (Russia's largest food retailer). The project includes a synthetic data generator that mirrors the exact schema for environments without network access.

| Table | Rows | Description |
|-------|------|-------------|
| clients | 50K | Customer demographics, loyalty card tenure |
| products | 5K | SKU catalog with department/aisle hierarchy |
| purchases | 2M | Transaction history (pre-campaign) |
| uplift_train | 50K | RCT outcomes: treatment_flg + target |

## License

MIT

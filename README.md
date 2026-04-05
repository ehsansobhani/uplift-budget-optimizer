# 💰 Optimal Budget Allocation for Retail Media (Uplift Modeling)

An end-to-end machine learning system for **causal marketing optimization** — identifying *which customers to target* and *how much budget to allocate* to maximize incremental profit.

Built on the **X5 RetailHero dataset**, this project moves beyond prediction and focuses on **decision-making under budget constraints**.

---

# 📊 Dataset

This project uses the:

👉 **X5 RetailHero Uplift Modeling Dataset**
https://ods.ai/competitions/x5-retailhero-uplift-modeling

It is a real-world dataset from a large grocery retailer containing:

| Table              | Description                          |
| ------------------ | ------------------------------------ |
| `clients.csv`      | Customer demographics & loyalty info |
| `products.csv`     | Product catalog                      |
| `purchases.csv`    | Transaction history (45M+ rows)      |
| `uplift_train.csv` | RCT data (treatment + outcome)       |
| `uplift_test.csv`  | Holdout set for scoring              |

---

# 🚀 What This System Does

* Estimates **individual treatment effect (uplift)**
* Ranks customers by **incremental impact**
* Simulates campaign performance under **budget constraints**
* Computes **profit = revenue − cost**
* Finds **optimal budget (argmax profit)**

---

# 🧠 Core Pipeline

```
Raw Data → Feature Engineering → Uplift Models → Evaluation → Budget Optimization → Dashboard
```

---

# 📈 Results

* **Best Model:** S-Learner
* **AUUC:** ~1137
* **Uplift@30%:** ~6.7%
* **Observed ATE:** ~3.3%

### 💰 Business Impact

* +60% revenue vs random targeting
* ROI: ~3259%
* Optimal targeting uses **partial population**, not all users

---

# 💻 Project Structure (Actual Repo)

```
retail-media-campaign-optimizer/
│
├── artifacts/                  # Generated outputs
│   ├── submission.csv
│   ├── model_comparison.csv
│   ├── qini_curves.png
│   ├── budget_curve.csv
│   ├── optimal_budget.png
│   ├── best_scores.npy
│   ├── y_test.npy
│   └── t_test.npy
│
├── configs/                    # Config files
├── dashboard/                  # (optional future dashboard modules)
├── data/                       # Local dataset (not committed)
├── src/                        # Core ML pipeline
├── tests/                      # Unit tests
│
├── app.py                      # Streamlit dashboard
├── main_full.py                # Full pipeline
├── Dockerfile
├── .gitignore
└── README.md
```

---

# ⚙️ How to Run

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Run full pipeline

```bash
python main_full.py
```

This generates:

* model outputs
* Qini curves
* budget optimization results
* `submission.csv`

---

## 3. Run Streamlit dashboard

```bash
streamlit run app.py
```

---

# 📊 Streamlit Dashboard (Important)

The dashboard:

* automatically loads:

```python
artifacts/submission.csv
```

👉 **No file upload is required**

---

## ⚠️ If You Use Your Own Data

You must update this line in:

```
app.py
```

```python
GITHUB_RAW_URL = "https://raw.githubusercontent.com/<your-repo>/artifacts/submission.csv"
```

Otherwise the dashboard will not load your data.

---

# 📈 Dashboard Features

* 📊 Profit vs Budget curve
* 📈 Incremental conversions
* 💰 Optimal budget (argmax profit)
* ⚙️ Adjustable business parameters
* 🧠 Decision-focused insights

---

# 🧠 Key Insights

* Uplift is **heterogeneous across customers**
* Increasing budget leads to **diminishing returns**
* Optimal strategy is **not full targeting**
* Decision-making must be **profit-driven**

---

# 🔥 Why This Project Stands Out

Most ML projects:

* focus on prediction

This project:

* focuses on **decision optimization under constraints**

---

# 🧪 Tech Stack

* Python
* Pandas / NumPy
* LightGBM
* Scikit-learn
* Plotly / Matplotlib
* Streamlit

---

# 🚀 Future Improvements

* Policy learning (threshold optimization)
* FastAPI deployment
* Multi-campaign optimization
* Real-time inference
* Docker + CI/CD

---

# 📌 Summary

This project demonstrates:

* scalable ML pipelines
* causal inference (uplift modeling)
* business-driven optimization
* end-to-end system design

---

# 📎 License

MIT

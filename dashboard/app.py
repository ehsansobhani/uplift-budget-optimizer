# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt


# plt.style.use("seaborn-v0_8-whitegrid")

# st.set_page_config(page_title="Uplift Budget Optimizer", layout="wide")

# st.title("📊 Uplift Modeling – Budget Optimization Dashboard")

# # Upload fileimport pandas as pd

# GITHUB_RAW_URL = "https://github.com/ehsansobhani/uplift-budget-optimizer/blob/main/artifacts/submission.csv"

# @st.cache_data
# def load_data():
#     return pd.read_csv(GITHUB_RAW_URL)

# df = load_data()
# # uploaded_file = st.file_uploader("Upload CSV with columns: uplift", type=["csv"])

# if df:
#     # df = pd.read_csv(uploaded_file)

#     if "uplift" not in df.columns:
#         st.error("CSV must contain 'uplift' column")
#     else:
#         df = df.sort_values("uplift", ascending=False).reset_index(drop=True)

#         st.sidebar.header("💰 Business Parameters")
#         value_per_conversion = st.sidebar.number_input("Value per Conversion ($)", value=25.0)
#         cost_per_user = st.sidebar.number_input("Cost per Targeted User ($)", value=0.05)

#         n = len(df)
#         budgets = np.linspace(0.01, 1.0, 50)

#         profits = []
#         incremental_gains = []

#         for k in budgets:
#             top_k = df.iloc[: int(k * n)]
#             incremental = top_k["uplift"].sum()
#             cost = k * n * cost_per_user
#             revenue = incremental * value_per_conversion
#             profit = revenue - cost

#             profits.append(profit)
#             incremental_gains.append(incremental)

#         results = pd.DataFrame({
#             "budget": budgets,
#             "profit": profits,
#             "incremental": incremental_gains
#         })

#         best_idx = results["profit"].idxmax()
#         best_row = results.loc[best_idx]

#         st.subheader("🔥 TRUE Optimal Budget (argmax profit)")

#         best_budget = best_row["budget"]
#         best_profit = best_row["profit"]
#         best_incremental = best_row["incremental"]

#         col1, col2, col3 = st.columns(3)

#         col1.metric("Optimal Budget (%)", f"{best_budget*100:.2f}%")
#         col2.metric("Max Profit ($)", f"{best_profit:,.2f}")
#         col3.metric("Incremental Conversions", f"{best_incremental:,.0f}")
#         # Plot Profit Curve
#         fig1, ax1 = plt.subplots()
#         ax1.plot(results["budget"], results["profit"])

#         # highlight optimal point
#         ax1.scatter(best_budget, best_profit)
#         ax1.axvline(best_budget, linestyle="--")

#         ax1.set_title("Profit vs Budget")
#         ax1.set_xlabel("Budget Fraction")
#         ax1.set_ylabel("Profit")

#         st.pyplot(fig1)

#         # Plot Incremental Gain
#         fig2, ax2 = plt.subplots()
#         ax2.plot(results["budget"], results["incremental"])
#         ax2.set_title("Incremental Gain vs Budget")
#         ax2.set_xlabel("Budget Fraction")
#         ax2.set_ylabel("Incremental Conversions")
#         st.pyplot(fig2)

#         st.subheader("📋 Full Results")
#         st.dataframe(results)

# else:
#     st.info("Upload a CSV file with an 'uplift' column to begin.")




import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")

st.set_page_config(page_title="Uplift Budget Optimizer", layout="wide")

st.title("📊 Uplift Modeling – Budget Optimization Dashboard")

# ─────────────────────────────────────────────
# LOAD DATA FROM GITHUB (FIXED)
# ─────────────────────────────────────────────
GITHUB_RAW_URL = "https://raw.githubusercontent.com/ehsansobhani/uplift-budget-optimizer/main/artifacts/submission.csv"

@st.cache_data
def load_data():
    try:
        return pd.read_csv(GITHUB_RAW_URL)
    except Exception as e:
        st.error(f"❌ Failed to load data from GitHub: {e}")
        return None

df = load_data()

# ─────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────
if df is not None and not df.empty:

    if "uplift" not in df.columns:
        st.error("CSV must contain 'uplift' column")
    else:
        df = df.sort_values("uplift", ascending=False).reset_index(drop=True)

        # ─────────────────────────────────────────────
        # SIDEBAR INPUTS
        # ─────────────────────────────────────────────
        st.sidebar.header("💰 Business Parameters")

        value_per_conversion = st.sidebar.number_input(
            "Value per Conversion ($)", value=25.0
        )

        cost_per_user = st.sidebar.number_input(
            "Cost per Targeted User ($)", value=0.05
        )

        # ─────────────────────────────────────────────
        # BUDGET SIMULATION
        # ─────────────────────────────────────────────
        n = len(df)
        budgets = np.linspace(0.01, 1.0, 50)

        profits = []
        incremental_gains = []

        for k in budgets:
            top_k = df.iloc[: int(k * n)]

            incremental = top_k["uplift"].sum()
            cost = k * n * cost_per_user
            revenue = incremental * value_per_conversion
            profit = revenue - cost

            profits.append(profit)
            incremental_gains.append(incremental)

        results = pd.DataFrame({
            "budget": budgets,
            "profit": profits,
            "incremental": incremental_gains
        })

        # ─────────────────────────────────────────────
        # OPTIMAL POINT
        # ─────────────────────────────────────────────
        best_idx = results["profit"].idxmax()
        best_row = results.loc[best_idx]

        best_budget = best_row["budget"]
        best_profit = best_row["profit"]
        best_incremental = best_row["incremental"]

        st.subheader("🔥 Optimal Budget (Profit Maximization)")

        col1, col2, col3 = st.columns(3)

        col1.metric("Optimal Budget (%)", f"{best_budget*100:.2f}%")
        col2.metric("Max Profit ($)", f"{best_profit:,.2f}")
        col3.metric("Incremental Conversions", f"{best_incremental:,.0f}")

        # ─────────────────────────────────────────────
        # PLOT 1: PROFIT CURVE (CLEAN)
        # ─────────────────────────────────────────────
        fig1, ax1 = plt.subplots(figsize=(8, 5))

        ax1.plot(results["budget"], results["profit"], linewidth=2)
        ax1.scatter(best_budget, best_profit, s=80)

        ax1.axvline(best_budget, linestyle="--")

        ax1.set_title("Optimal Budget Allocation", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Budget Fraction")
        ax1.set_ylabel("Profit")

        # remove ugly borders
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # annotation
        ax1.annotate(
            f"Optimal\n{best_budget:.2%}",
            (best_budget, best_profit),
            textcoords="offset points",
            xytext=(10,10),
        )

        st.pyplot(fig1)

        # ─────────────────────────────────────────────
        # PLOT 2: INCREMENTAL GAIN
        # ─────────────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(8, 5))

        ax2.plot(results["budget"], results["incremental"], linewidth=2)

        ax2.set_title("Incremental Gain vs Budget", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Budget Fraction")
        ax2.set_ylabel("Incremental Conversions")

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        st.pyplot(fig2)

        # ─────────────────────────────────────────────
        # TABLE
        # ─────────────────────────────────────────────
        st.subheader("📋 Budget Simulation Table")
        st.dataframe(results)

else:
    st.warning("⚠️ No data loaded. Check GitHub path or file availability.")
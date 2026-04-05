import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Uplift Budget Optimizer", layout="wide")

st.title("📊 Uplift Modeling – Budget Optimization Dashboard")
import pandas as pd

GITHUB_RAW_URL = "https://github.com/ehsansobhani/uplift-budget-optimizer/blob/main/artifacts/submission.csv"

@st.cache_data
def load_data():
    return pd.read_csv(GITHUB_RAW_URL)

df = load_data()
# Upload file
# uploaded_file = st.file_uploader("Upload CSV with columns: uplift", type=["csv"])

if df:
    # df = pd.read_csv(uploaded_file)

    if "uplift" not in df.columns:
        st.error("CSV must contain 'uplift' column")
    else:
        df = df.sort_values("uplift", ascending=False).reset_index(drop=True)

        st.sidebar.header("💰 Business Parameters")
        value_per_conversion = st.sidebar.number_input("Value per Conversion ($)", value=25.0)
        cost_per_user = st.sidebar.number_input("Cost per Targeted User ($)", value=0.05)

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

        best_idx = results["profit"].idxmax()
        best_row = results.loc[best_idx]

        st.subheader("🔥 TRUE Optimal Budget (argmax profit)")

        best_budget = best_row["budget"]
        best_profit = best_row["profit"]
        best_incremental = best_row["incremental"]

        col1, col2, col3 = st.columns(3)

        col1.metric("Optimal Budget (%)", f"{best_budget*100:.2f}%")
        col2.metric("Max Profit ($)", f"{best_profit:,.2f}")
        col3.metric("Incremental Conversions", f"{best_incremental:,.0f}")
        # Plot Profit Curve
        fig1, ax1 = plt.subplots()
        ax1.plot(results["budget"], results["profit"])

        # highlight optimal point
        ax1.scatter(best_budget, best_profit)
        ax1.axvline(best_budget, linestyle="--")

        ax1.set_title("Profit vs Budget")
        ax1.set_xlabel("Budget Fraction")
        ax1.set_ylabel("Profit")

        st.pyplot(fig1)

        # Plot Incremental Gain
        fig2, ax2 = plt.subplots()
        ax2.plot(results["budget"], results["incremental"])
        ax2.set_title("Incremental Gain vs Budget")
        ax2.set_xlabel("Budget Fraction")
        ax2.set_ylabel("Incremental Conversions")
        st.pyplot(fig2)

        st.subheader("📋 Full Results")
        st.dataframe(results)

else:
    st.info("Upload a CSV file with an 'uplift' column to begin.")

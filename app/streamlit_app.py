import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.pricing_model import (
    load_data,
    train_model,
    get_product_snapshot,
    optimize_price,
    scenario_test,
    sensitivity_label
)

st.set_page_config(page_title="Pricing Analytics Simulator", layout="wide")

st.title("Pricing Analytics Simulator")
st.markdown(
    "Interactive pricing scenario tool demonstrating how data-driven models can support "
    "pricing decisions in a B2B distribution setting."
)
# Load data and train model
@st.cache_data
def get_data():
    return load_data("data/synthetic_pricing_data.csv")

@st.cache_resource
def get_trained_model(df):
    return train_model(df)

df = get_data()
model, feature_columns = get_trained_model(df)

# Sidebar inputs
st.sidebar.header("Scenario Inputs")

sku_options = sorted(df["SKU"].unique())
selected_sku = st.sidebar.selectbox("Select Product", sku_options)

branch_options = sorted(df[df["SKU"] == selected_sku]["Branch"].unique())
selected_branch = st.sidebar.selectbox("Select Branch", branch_options)

month_options = df["Month"].cat.categories.tolist()
available_months = df[
    (df["SKU"] == selected_sku) & (df["Branch"] == selected_branch)
]["Month"].astype(str).unique().tolist()

filtered_month_options = [m for m in month_options if m in available_months]
selected_month = st.sidebar.selectbox("Select Month", filtered_month_options)

scenario_pct = st.sidebar.slider("Scenario Price Change (%)", min_value=-10, max_value=10, value=5, step=1)

snapshot = get_product_snapshot(df, selected_sku, selected_branch, selected_month)

if snapshot is None:
    st.error("No matching product snapshot found.")
    st.stop()

current_price = snapshot["price"]
category = snapshot["category"]
unit_cost = snapshot["unit_cost"]

# Run model logic
opt = optimize_price(
    model=model,
    category=category,
    branch=selected_branch,
    month=selected_month,
    unit_cost=unit_cost,
    feature_columns=feature_columns,
    price_min=max(10, current_price * 0.7),
    price_max=current_price * 1.3,
    step=1
)

scenario = scenario_test(
    model=model,
    current_price=current_price,
    scenario_pct=scenario_pct,
    category=category,
    branch=selected_branch,
    month=selected_month,
    feature_columns=feature_columns
)

signal = sensitivity_label(scenario["demand_change_pct"])

# Top metrics
col1, col2, col3 = st.columns(3)

col1.metric("Current Price", f"${current_price:,.2f}")
col2.metric("Revenue-Optimized Price", f"${opt['revenue_best_price']:,.2f}")
col3.metric("Profit-Optimized Price", f"${opt['profit_best_price']:,.2f}")

# Scenario output
st.subheader("Scenario Impact")

out1, out2, out3 = st.columns(3)
out1.metric("Scenario Price", f"${scenario['scenario_price']:,.2f}")
out2.metric("Predicted Demand Change", f"{scenario['demand_change_pct']:.2f}%")
out3.metric("Predicted Revenue Change", f"{scenario['revenue_change_pct']:.2f}%")

st.markdown(f"**Pricing Response Signal:** {signal}")

summary_text = (
    f"Current Price: ${current_price:,.2f}\n\n"
    f"Recommended Range:\n"
    f"- Revenue-Optimized Price: ${opt['revenue_best_price']:,.2f}\n"
    f"- Profit-Optimized Price: ${opt['profit_best_price']:,.2f}\n\n"
    f"Scenario Impact ({scenario_pct:+.0f}% change):\n"
    f"- Demand: {scenario['demand_change_pct']:.2f}%\n"
    f"- Revenue: {scenario['revenue_change_pct']:.2f}%\n\n"
    f"Pricing Signal: {signal}"
)

st.markdown(
    f"""
    <div style="
        background-color:#f9fafb;
        padding:18px;
        border-radius:10px;
        border:1px solid #e5e7eb;
        font-size:16px;
        line-height:1.6;
    ">
    {summary_text.replace('\n', '<br>')}
    </div>
    """,
    unsafe_allow_html=True
)

# Historical chart
st.subheader("Historical Price Trend")

hist = df[(df["SKU"] == selected_sku) & (df["Branch"] == selected_branch)].copy()
month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
hist["Month"] = pd.Categorical(hist["Month"], categories=month_order, ordered=True)
hist = hist.sort_values("Month")
hist["Month"] = hist["Month"].astype(str)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(hist["Month"], hist["Price"], marker="o")
ax.set_xlabel("Month")
ax.set_ylabel("Price")
ax.set_title(f"Historical Price Trend: {selected_sku} in {selected_branch}")
plt.xticks(rotation=45)
st.pyplot(fig)

# Optimization curve
st.subheader("Simulated Revenue and Profit by Price")

curve_df = opt["results_table"].copy()

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(curve_df["Price"], curve_df["Revenue"], label="Revenue")
ax2.plot(curve_df["Price"], curve_df["Profit"], label="Profit")
ax2.set_xlabel("Price")
ax2.set_ylabel("Value")
ax2.set_title("Simulated Revenue and Profit Across Candidate Prices")
ax2.legend()
st.pyplot(fig2)

# Detail table
st.subheader("Selected Product Snapshot")
st.dataframe(pd.DataFrame([snapshot]), use_container_width=True)

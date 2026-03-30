import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# PAGE CONFIG
st.set_page_config(
    page_title="Pricing Simulator",
    layout="wide"
)

# LOAD DATA

@st.cache_data
def load_data():
    return pd.read_csv("data/synthetic_pricing_data.csv")

df = load_data()

# SIDEBAR CONTROLS
st.sidebar.header("Scenario Inputs")

sku = st.sidebar.selectbox("Select Product (SKU)", df["SKU"].unique())
branch = st.sidebar.selectbox("Select Branch", df["Branch"].unique())

price_change_pct = st.sidebar.slider(
    "Price Change (%)",
    -20, 20, 0
)

# Filter data
filtered_df = df[(df["SKU"] == sku) & (df["Branch"] == branch)]

# Assume latest price
current_price = filtered_df["Price"].iloc[-1]
scenario_price = current_price * (1 + price_change_pct / 100)

# SIMULATION LOGIC (TEMP / SIMPLE)
def simulate(price):
    base_demand = 100
    elasticity = -1.2
    
    demand = base_demand * (price / current_price) ** elasticity
    revenue = demand * price
    margin = revenue * 0.3  # assume 30% margin
    
    return demand, revenue, margin

current_demand, current_revenue, current_margin = simulate(current_price)
scenario_demand, scenario_revenue, scenario_margin = simulate(scenario_price)

# % changes
demand_change = (scenario_demand - current_demand) / current_demand * 100
revenue_change = (scenario_revenue - current_revenue) / current_revenue * 100
margin_change = (scenario_margin - current_margin) / current_margin * 100

# HEADER
st.title("Pricing Analytics Simulator")
st.caption("Evaluate pricing scenarios across demand, revenue, and margin tradeoffs")

# KPI ROW
col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("Scenario Price", f"${scenario_price:.2f}")
col3.metric("Demand Change", f"{demand_change:.2f}%")
col4.metric("Revenue Change", f"{revenue_change:.2f}%")
col5.metric("Margin Change", f"{margin_change:.2f}%")
col6.metric("Pricing Signal", "Resilient" if abs(demand_change) < 5 else "Sensitive")

# RECOMMENDATION BANNER
if revenue_change > 0:
    st.success("Controlled price increase appears viable")
else:
    st.warning("Price change may negatively impact revenue")

# SUMMARY SECTION
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Scenario Summary")
    st.write(f"Current Price: ${current_price:.2f}")
    st.write(f"Scenario Price: ${scenario_price:.2f}")
    st.write(f"Demand Change: {demand_change:.2f}%")
    st.write(f"Revenue Change: {revenue_change:.2f}%")
    st.write(f"Margin Change: {margin_change:.2f}%")

with col_right:
    st.subheader("Recommended Pricing")
    
    # simple optimization sweep
    prices = np.linspace(current_price * 0.8, current_price * 1.2, 50)
    results = [simulate(p) for p in prices]
    
    revenues = [r[1] for r in results]
    margins = [r[2] for r in results]
    
    rev_opt_price = prices[np.argmax(revenues)]
    margin_opt_price = prices[np.argmax(margins)]
    
    st.write(f"Best for Revenue: ${rev_opt_price:.2f}")
    st.write(f"Best for Margin: ${margin_opt_price:.2f}")
    st.write("Suggested Test Range: ±5%")

# TOGGLE (Revenue vs Margin)
view = st.radio(
    "View Optimization",
    ["Revenue", "Margin"],
    horizontal=True
)

# MAIN CHART
chart_data = pd.DataFrame({
    "Price": prices,
    "Revenue": revenues,
    "Margin": margins
})

if view == "Revenue":
    fig = px.line(chart_data, x="Price", y="Revenue", title="Revenue vs Price")
    fig.add_scatter(x=[current_price], y=[current_revenue], mode="markers", name="Current")
    fig.add_scatter(x=[scenario_price], y=[scenario_revenue], mode="markers", name="Scenario")
else:
    fig = px.line(chart_data, x="Price", y="Margin", title="Margin vs Price")
    fig.add_scatter(x=[current_price], y=[current_margin], mode="markers", name="Current")
    fig.add_scatter(x=[scenario_price], y=[scenario_margin], mode="markers", name="Scenario")

st.plotly_chart(fig, use_container_width=True)

# HISTORICAL PRICE TREND
st.subheader("Historical Price Trend")

hist_fig = px.line(
    filtered_df,
    x="Month",
    y="Price",
    title="Price Over Time"
)

st.plotly_chart(hist_fig, use_container_width=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# PAGE CONFIG
st.set_page_config(
    page_title="Pricing Analytics Simulator",
    layout="wide"
)

# LOAD DATA
@st.cache_data
def load_data():
    return pd.read_csv("data/synthetic_pricing_data.csv")

df = load_data()

# OPTIONAL: CLEAN COLUMN NAMES
df.columns = [col.strip() for col in df.columns]

# CHECK REQUIRED COLUMNS
required_cols = ["SKU", "Branch", "Price", "Month"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing required columns in synthetic_pricing_data.csv: {missing_cols}")
    st.stop()

# SIDEBAR CONTROLS
st.sidebar.header("Scenario Inputs")

sku = st.sidebar.selectbox("Select Product (SKU)", sorted(df["SKU"].dropna().unique()))
branch = st.sidebar.selectbox("Select Branch", sorted(df["Branch"].dropna().unique()))

filtered_df = df[(df["SKU"] == sku) & (df["Branch"] == branch)].copy()

if filtered_df.empty:
    st.error("No data found for the selected SKU and Branch.")
    st.stop()

# Use latest row as current price
current_price = float(filtered_df["Price"].iloc[-1])

price_change_pct = st.sidebar.slider(
    "Price Change (%)",
    min_value=-20,
    max_value=20,
    value=0,
    step=1
)

slider_price = current_price * (1 + price_change_pct / 100)

scenario_price = st.sidebar.number_input(
    "Or Enter Scenario Price Directly",
    min_value=0.0,
    value=float(round(slider_price, 2)),
    step=0.25
)

# SIMULATION LOGIC (TEMP PLACEHOLDER)
def simulate(price, current_price):
    base_demand = 100
    elasticity = -1.2
    margin_rate = 0.30

    demand = base_demand * (price / current_price) ** elasticity

    penalty = np.exp(-0.0008 * (price - current_price) ** 2)
    adjusted_demand = demand * penalty

    revenue = adjusted_demand * price
    margin = revenue * margin_rate

    return adjusted_demand, revenue, margin

current_demand, current_revenue, current_margin = simulate(current_price, current_price)
scenario_demand, scenario_revenue, scenario_margin = simulate(scenario_price, current_price)

demand_change = ((scenario_demand - current_demand) / current_demand) * 100 if current_demand != 0 else 0
revenue_change = ((scenario_revenue - current_revenue) / current_revenue) * 100 if current_revenue != 0 else 0
margin_change = ((scenario_margin - current_margin) / current_margin) * 100 if current_margin != 0 else 0

# PRICING SIGNAL
if abs(demand_change) < 5:
    pricing_signal = "Resilient"
elif abs(demand_change) < 10:
    pricing_signal = "Moderate"
else:
    pricing_signal = "Sensitive"

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
col6.metric("Pricing Signal", pricing_signal)

# RECOMMENDATION BANNER
if abs(scenario_price - current_price) < 0.01:
    st.info("No price change selected yet")
elif revenue_change > 0 and margin_change >= 0:
    st.success("Controlled price increase appears viable")
elif revenue_change > 0:
    st.warning("Revenue may improve, but tradeoffs should be reviewed")
else:
    st.warning("Price change may negatively impact revenue")

# OPTIMIZATION SWEEP
prices = np.linspace(current_price * 0.8, current_price * 1.2, 50)
results = [simulate(p, current_price) for p in prices]

demands = [r[0] for r in results]
revenues = [r[1] for r in results]
margins = [r[2] for r in results]

rev_opt_price = float(prices[np.argmax(revenues)])
margin_opt_price = float(prices[np.argmax(margins)])

# DECISION GUIDANCE LOGIC
risk_level = "Low" if abs(demand_change) < 3 else "Moderate" if abs(demand_change) < 8 else "High"

if scenario_price > current_price:
    scenario_direction = "Price Increase"
elif scenario_price < current_price:
    scenario_direction = "Price Decrease"
else:
    scenario_direction = "No Change"

hist_min = float(filtered_df["Price"].min())
hist_max = float(filtered_df["Price"].max())

if hist_min <= scenario_price <= hist_max:
    hist_context = "Within historical price range"
else:
    hist_context = "Outside historical price range"

num_obs = len(filtered_df)
price_std = filtered_df["Price"].std()

if num_obs >= 10 and pd.notna(price_std) and price_std < 6:
    confidence_level = "High"
elif num_obs >= 6:
    confidence_level = "Moderate"
else:
    confidence_level = "Low"

if revenue_change > 0 and margin_change >= 0:
    suggested_action = "Test scenario appears viable"
elif revenue_change > 0 and margin_change < 0:
    suggested_action = "Revenue improves, but margin should be monitored"
else:
    suggested_action = "Scenario should be reviewed before use"

# DECISION GUIDANCE + RECOMMENDED PRICING
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("Decision Guidance")
    st.write(f"**Scenario Direction:** {scenario_direction}")
    st.write(f"**Risk Level:** {risk_level}")
    st.write(f"**Historical Context:** {hist_context}")
    st.write(f"**Confidence Level:** {confidence_level}")
    st.write(f"**Suggested Action:** {suggested_action}")

with right_col:
    st.subheader("Recommended Pricing")
    st.write(f"**Best for Revenue:** ${rev_opt_price:.2f}")
    st.write(f"**Best for Margin:** ${margin_opt_price:.2f}")
    st.write("**Suggested Test Range:** ±5%")

# TOGGLE (Revenue vs Margin)
view = st.radio(
    "View Optimization",
    ["Revenue", "Margin"],
    horizontal=True
)

# MAIN CHART
chart_data = pd.DataFrame({
    "Price": prices,
    "Demand": demands,
    "Revenue": revenues,
    "Margin": margins
})

if view == "Revenue":
    fig = px.line(chart_data, x="Price", y="Revenue", title="Revenue vs Price")
    fig.add_scatter(x=[current_price], y=[current_revenue], mode="markers", name="Current")
    fig.add_scatter(x=[scenario_price], y=[scenario_revenue], mode="markers", name="Scenario")
    fig.add_scatter(
        x=[rev_opt_price],
        y=[max(revenues)],
        mode="markers",
        name="Revenue Optimal"
    )
else:
    fig = px.line(chart_data, x="Price", y="Margin", title="Margin vs Price")
    fig.add_scatter(x=[current_price], y=[current_margin], mode="markers", name="Current")
    fig.add_scatter(x=[scenario_price], y=[scenario_margin], mode="markers", name="Scenario")
    fig.add_scatter(
        x=[margin_opt_price],
        y=[max(margins)],
        mode="markers",
        name="Margin Optimal"
    )

fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# HISTORICAL PRICE TREND
st.subheader("Historical Price Trend")

month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

if filtered_df["Month"].dtype == object:
    filtered_df["Month"] = pd.Categorical(
        filtered_df["Month"],
        categories=month_order,
        ordered=True
    )
    filtered_df = filtered_df.sort_values("Month")

hist_fig = px.line(
    filtered_df,
    x="Month",
    y="Price",
    title="Price Over Time"
)

hist_fig.update_layout(template="plotly_dark")
st.plotly_chart(hist_fig, use_container_width=True)

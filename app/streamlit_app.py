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

# HISTORICAL TREND SORTING
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

if filtered_df["Month"].dtype == object:
    filtered_df["Month"] = filtered_df["Month"].astype(str).str.strip()
    filtered_df["Month"] = pd.Categorical(
        filtered_df["Month"],
        categories=month_order,
        ordered=True
    )
    filtered_df = filtered_df.sort_values("Month")

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
# Replace later with your real model
def simulate(price):
    peak_revenue_price = 146.0
    max_revenue = 14500.0
    unit_cost = 92.0

    # Revenue curve with a visible peak
    revenue = max_revenue - 8.0 * (price - peak_revenue_price) ** 2
    revenue = max(revenue, 0)

    # Demand implied by revenue and price
    demand = revenue / price if price > 0 else 0

    # Margin uses unit cost, so margin optimum can differ from revenue optimum
    margin = (price - unit_cost) * demand if price > unit_cost else 0

    return demand, revenue, margin

current_demand, current_revenue, current_margin = simulate(current_price)
scenario_demand, scenario_revenue, scenario_margin = simulate(scenario_price)

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
elif revenue_change > 0 and margin_change > 0 and abs(demand_change) < 5:
    st.success("Controlled price increase appears viable")
elif revenue_change > 0 and abs(demand_change) < 10:
    st.warning("Revenue may improve, but customer response should be monitored")
else:
    st.warning("Scenario may weaken overall performance")

# OPTIMIZATION SWEEP
prices = np.linspace(current_price * 0.8, current_price * 1.2, 50)
results = [simulate(p) for p in prices]

demands = [r[0] for r in results]
revenues = [r[1] for r in results]
margins = [r[2] for r in results]

rev_opt_idx = int(np.argmax(revenues))
margin_opt_idx = int(np.argmax(margins))

rev_opt_price = float(prices[rev_opt_idx])
margin_opt_price = float(prices[margin_opt_idx])

rev_opt_value = float(revenues[rev_opt_idx])
margin_opt_value = float(margins[margin_opt_idx])

# DECISION GUIDANCE LOGIC
distance_from_revenue_opt = abs(scenario_price - rev_opt_price) / rev_opt_price * 100 if rev_opt_price != 0 else 0
distance_from_margin_opt = abs(scenario_price - margin_opt_price) / margin_opt_price * 100 if margin_opt_price != 0 else 0
nearest_opt_distance = min(distance_from_revenue_opt, distance_from_margin_opt)

if scenario_price > current_price:
    scenario_direction = "Price Increase"
elif scenario_price < current_price:
    scenario_direction = "Price Decrease"
else:
    scenario_direction = "No Change"

hist_min = float(filtered_df["Price"].min())
hist_max = float(filtered_df["Price"].max())
hist_mid = (hist_min + hist_max) / 2
hist_range = hist_max - hist_min if hist_max > hist_min else 1
distance_from_history_mid = abs(scenario_price - hist_mid) / hist_range

if hist_min <= scenario_price <= hist_max:
    hist_context = "Within historical price range"
else:
    hist_context = "Outside historical price range"

if abs(demand_change) < 3 and nearest_opt_distance <= 3:
    risk_level = "Low"
elif abs(demand_change) < 8 and nearest_opt_distance <= 8:
    risk_level = "Moderate"
else:
    risk_level = "High"

num_obs = len(filtered_df)

if num_obs < 6:
    confidence_level = "Low"
elif nearest_opt_distance <= 3 and abs(demand_change) <= 5 and distance_from_history_mid <= 0.6:
    confidence_level = "High"
elif nearest_opt_distance <= 8 and abs(demand_change) <= 10 and distance_from_history_mid <= 1.0:
    confidence_level = "Moderate"
else:
    confidence_level = "Low"

if abs(scenario_price - current_price) < 0.01:
    suggested_action = "No change selected; use the slider or input box to test a scenario"
elif revenue_change > 2 and margin_change > 2 and abs(demand_change) < 5:
    suggested_action = "Strong candidate for testing: revenue and margin improve with limited volume risk"
elif revenue_change > 0 and margin_change > 0 and abs(demand_change) < 10:
    suggested_action = "Viable scenario: gains appear positive, but monitor customer response"
elif revenue_change > 0 and margin_change <= 0:
    suggested_action = "Revenue may improve, but margin tradeoff should be reviewed"
elif revenue_change < 0 and margin_change > 0:
    suggested_action = "Margin improves, but revenue declines; suitable only if margin protection is the priority"
else:
    suggested_action = "Not recommended: modeled tradeoff suggests the scenario weakens overall performance"

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
    st.write(f"**Suggested Test Range:** ${rev_opt_price * 0.98:.2f} – ${rev_opt_price * 1.02:.2f}")

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
    fig.add_scatter(
        x=[current_price],
        y=[current_revenue],
        mode="markers",
        name="Current"
    )
    fig.add_scatter(
        x=[scenario_price],
        y=[scenario_revenue],
        mode="markers",
        name="Scenario"
    )
    fig.add_scatter(
        x=[rev_opt_price],
        y=[rev_opt_value],
        mode="markers",
        name="Revenue Optimal"
    )
else:
    fig = px.line(chart_data, x="Price", y="Margin", title="Margin vs Price")
    fig.add_scatter(
        x=[current_price],
        y=[current_margin],
        mode="markers",
        name="Current"
    )
    fig.add_scatter(
        x=[scenario_price],
        y=[scenario_margin],
        mode="markers",
        name="Scenario"
    )
    fig.add_scatter(
        x=[margin_opt_price],
        y=[margin_opt_value],
        mode="markers",
        name="Margin Optimal"
    )

fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# HISTORICAL PRICE TREND
st.subheader("Historical Price Trend")

hist_fig = px.line(
    filtered_df,
    x="Month",
    y="Price",
    title="Price Over Time"
)

hist_fig.update_layout(template="plotly_dark")
st.plotly_chart(hist_fig, use_container_width=True)

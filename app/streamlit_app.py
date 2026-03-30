"""
app/streamlit_app.py
Pricing Analytics Simulator — Streamlit frontend.
Model backend: src/pricing_model.py (XGBoost gradient boosting)

Startup sequence
1. load_data()           — load CSV once, cached
2. train_model()         — train XGBoost once, cached
3. precompute_sweeps()   — sweep all SKU x Branch combos, cached
   ↑ all three steps run once on first load (~2-3 sec), then never again

Runtime (every slider move)
4. lookup_scenario()     — interpolate from precomputed sweep, instant
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

from src.pricing_model import (
    load_data,
    train_model,
    precompute_sweeps,
    lookup_scenario,
    get_product_snapshot,
    get_portfolio_summary,
    sensitivity_label,
)

#  PAGE CONFIG
st.set_page_config(
    page_title="Pricing Analytics Simulator",
    layout="wide"
)

# STARTUP: LOAD → TRAIN → PRECOMPUTE 
@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_model(df: pd.DataFrame):
    """Train XGBoost. cache_resource keeps the model object in memory."""
    return train_model(df)

@st.cache_resource
def get_sweeps(_df: pd.DataFrame, _model, _feature_columns: list):
    """
    Precompute price sweeps for all SKU x Branch combos.
    Underscore prefix tells Streamlit not to hash these args
    (model objects aren't hashable).
    """
    return precompute_sweeps(_df, _model, _feature_columns)

df = get_data()

with st.spinner("Training pricing model and precomputing price curves..."):
    model, feature_columns = get_model(df)
    sweeps = get_sweeps(df, model, feature_columns)

# SIDEBAR 

st.sidebar.header("Scenario Inputs")

sku    = st.sidebar.selectbox("Select Product (SKU)",  sorted(df["SKU"].dropna().unique()))
branch = st.sidebar.selectbox("Select Branch",         sorted(df["Branch"].dropna().unique()))

sku_branch_key = f"{sku}__{branch}"
if st.session_state.get("last_sku_branch") != sku_branch_key:
    snap = get_product_snapshot(df, sku, branch)
    st.session_state["scenario_price"] = snap["price"] if snap else 100.0
    st.session_state["last_sku_branch"] = sku_branch_key

snap        = get_product_snapshot(df, sku, branch)
current_price = snap["price"]
unit_cost     = snap["unit_cost"]

price_change_pct = st.sidebar.slider(
    "Price Change (%)",
    min_value=-20,
    max_value=20,
    value=0,
    step=1,
)

slider_price = round(current_price * (1 + price_change_pct / 100), 2)

scenario_price = st.sidebar.number_input(
    "Or Enter Scenario Price Directly",
    min_value=0.01,
    value=float(slider_price),
    step=0.25,
)

#  CORE LOOKUPS
current  = lookup_scenario(sweeps, sku, branch, current_price)
scenario = lookup_scenario(sweeps, sku, branch, scenario_price)

# Derived metrics
demand_change  = ((scenario["predicted_units"] - current["predicted_units"])
                  / current["predicted_units"] * 100) if current["predicted_units"] > 0 else 0.0
revenue_change = ((scenario["revenue"] - current["revenue"])
                  / current["revenue"] * 100) if current["revenue"] > 0 else 0.0
margin_change  = ((scenario["margin"] - current["margin"])
                  / current["margin"] * 100) if current["margin"] > 0 else 0.0

pricing_signal = sensitivity_label(demand_change)

rev_opt_price    = current["rev_opt_price"]
margin_opt_price = current["margin_opt_price"]
rev_opt          = lookup_scenario(sweeps, sku, branch, rev_opt_price)
margin_opt       = lookup_scenario(sweeps, sku, branch, margin_opt_price)

breakeven_units = (unit_cost / (scenario_price - unit_cost)
                   if scenario_price > unit_cost else float("inf"))

# Decision guidance
distance_from_rev_opt    = abs(scenario_price - rev_opt_price) / rev_opt_price * 100 if rev_opt_price else 0
distance_from_margin_opt = abs(scenario_price - margin_opt_price) / margin_opt_price * 100 if margin_opt_price else 0
nearest_opt_distance     = min(distance_from_rev_opt, distance_from_margin_opt)

hist_min = float(df[(df["SKU"] == sku) & (df["Branch"] == branch)]["Price"].min())
hist_max = float(df[(df["SKU"] == sku) & (df["Branch"] == branch)]["Price"].max())
hist_mid = (hist_min + hist_max) / 2
hist_range = hist_max - hist_min if hist_max > hist_min else 1
distance_from_hist_mid = abs(scenario_price - hist_mid) / hist_range

hist_context     = "Within historical price range" if hist_min <= scenario_price <= hist_max else "Outside historical price range"
scenario_direction = ("Price Increase" if scenario_price > current_price
                      else "Price Decrease" if scenario_price < current_price
                      else "No Change")

if abs(demand_change) < 3 and nearest_opt_distance <= 3:
    risk_level = "Low"
elif abs(demand_change) < 8 and nearest_opt_distance <= 8:
    risk_level = "Moderate"
else:
    risk_level = "High"

num_obs = len(df[(df["SKU"] == sku) & (df["Branch"] == branch)])

if num_obs < 6:
    confidence_level = "Low"
elif nearest_opt_distance <= 3 and abs(demand_change) <= 5 and distance_from_hist_mid <= 0.6:
    confidence_level = "High"
elif nearest_opt_distance <= 8 and abs(demand_change) <= 10 and distance_from_hist_mid <= 1.0:
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

#PAGE HEADER 

st.title("Pricing Analytics Simulator")
st.caption("Demand model: XGBoost (gradient boosting) · Trained on historical pricing data")

#TABS

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Scenario Simulator",
    "Volume Analysis",
    "Diagnostics",
    "Portfolio View",
    "Model Details",
])

# TAB 1 — SCENARIO SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

with tab1:

    # Row 1: prices
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Current Price",   f"${current_price:.2f}")
    r1c2.metric("Scenario Price",  f"${scenario_price:.2f}")
    r1c3.metric("Pricing Signal",  pricing_signal)

    # Row 2: % changes
    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.metric("Demand Change",  f"{demand_change:.2f}%")
    r2c2.metric("Revenue Change", f"{revenue_change:.2f}%")
    r2c3.metric("Margin Change",  f"{margin_change:.2f}%")

    # Row 3: units
    r3c1, r3c2, r3c3 = st.columns(3)
    r3c1.metric("Modeled Units (Current)",
                f"{current['predicted_units']:.1f}")
    r3c2.metric("Modeled Units (Scenario)",
                f"{scenario['predicted_units']:.1f}",
                delta=f"{scenario['predicted_units'] - current['predicted_units']:.1f} units")
    r3c3.metric("Breakeven Units",
                f"{breakeven_units:.0f}" if breakeven_units != float("inf") else "Below cost")

    # Banner
    if abs(scenario_price - current_price) < 0.01:
        st.info("No price change selected yet")
    elif revenue_change > 0 and margin_change > 0 and abs(demand_change) < 5:
        st.success("Controlled price increase appears viable")
    elif revenue_change > 0 and abs(demand_change) < 10:
        st.warning("Revenue may improve, but customer response should be monitored")
    else:
        st.warning("Scenario may weaken overall performance")

    # Decision guidance + recommended pricing
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
        low  = rev_opt_price * 0.98
        high = rev_opt_price * 1.02
        st.markdown(f"**Revenue-Optimal Price:** \\${rev_opt_price:.2f}")
        st.markdown(f"**Margin-Optimal Price:** \\${margin_opt_price:.2f}")
        st.markdown(f"**Suggested Test Range:** \\${low:.2f} to \\${high:.2f}")

    # Chart toggle
    view = st.radio("View Optimization", ["Revenue", "Margin"], horizontal=True)

    # Build chart from precomputed sweep curves
    curve_prices = current["prices"]

    if view == "Revenue":
        y_current  = current["revenue"]
        y_scenario = scenario["revenue"]
        y_opt      = rev_opt["revenue"]
        y_curve    = current["revenues"]
        opt_price  = rev_opt_price
        y_label    = "Revenue ($)"
        chart_title = "Revenue vs Price  ·  Gradient Boosting Model"
        zone_lo, zone_hi = low, high
    else:
        y_current  = current["margin"]
        y_scenario = scenario["margin"]
        y_opt      = margin_opt["margin"]
        y_curve    = current["margins"]
        opt_price  = margin_opt_price
        y_label    = "Margin ($)"
        chart_title = "Margin vs Price  ·  Gradient Boosting Model"
        zone_lo = margin_opt_price * 0.98
        zone_hi = margin_opt_price * 1.02

    fig = go.Figure()

    # Curve
    fig.add_trace(go.Scatter(
        x=curve_prices, y=y_curve,
        mode="lines", name=view,
        line=dict(color="steelblue", width=2)
    ))

    # Recommended zone
    fig.add_shape(type="rect",
        x0=zone_lo, x1=zone_hi, y0=0, y1=1,
        xref="x", yref="paper",
        fillcolor="green", opacity=0.08, line_width=0
    )
    fig.add_annotation(
        x=(zone_lo + zone_hi) / 2, y=0.05,
        xref="x", yref="paper",
        text="Recommended Zone", showarrow=False,
        font=dict(color="lightgreen", size=11)
    )

    # Vertical lines
    fig.add_vline(x=current_price,  line_dash="dash", line_color="red",
                  annotation_text="Current",  annotation_position="bottom right")
    fig.add_vline(x=scenario_price, line_dash="dash", line_color="green",
                  annotation_text="Scenario", annotation_position="bottom left")
    fig.add_vline(x=opt_price,      line_dash="dash", line_color="purple",
                  annotation_text="Optimal",  annotation_position="bottom right")

    # Marker dots
    fig.add_trace(go.Scatter(
        x=[current_price, scenario_price, opt_price],
        y=[y_current, y_scenario, y_opt],
        mode="markers",
        marker=dict(size=10, color=["red", "green", "purple"]),
        showlegend=False,
        hovertemplate="%{y:,.0f}<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark",
        title=chart_title,
        xaxis_title="Price ($)",
        yaxis_title=y_label,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics under chart
    m1, m2, m3 = st.columns(3)
    if view == "Revenue":
        m1.metric("Current Revenue",  f"${current['revenue']:,.0f}")
        m2.metric("Scenario Revenue", f"${scenario['revenue']:,.0f}")
        m3.metric("Optimal Revenue",  f"${rev_opt['revenue']:,.0f}")
    else:
        m1.metric("Current Margin",  f"${current['margin']:,.0f}")
        m2.metric("Scenario Margin", f"${scenario['margin']:,.0f}")
        m3.metric("Optimal Margin",  f"${margin_opt['margin']:,.0f}")

    # Historical price trend
    st.subheader("Historical Price Trend")
    hist_df = df[(df["SKU"] == sku) & (df["Branch"] == branch)].copy()
    hist_df["Month"] = hist_df["Month"].astype(str)
    avg_hist_price = float(hist_df["Price"].mean())

    hist_fig = px.line(hist_df, x="Month", y="Price", title="Price Over Time")
    hist_fig.add_hline(
        y=avg_hist_price, line_dash="dash",
        annotation_text="Historical Average", annotation_position="top left"
    )
    hist_fig.update_layout(template="plotly_dark")
    st.plotly_chart(hist_fig, use_container_width=True)

# TAB 2 — VOLUME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Volume Analysis")
    st.caption("Explore how units sold affect revenue, margin, and breakeven at any price")

    vc1, vc2 = st.columns(2)
    with vc1:
        vol_price = st.number_input(
            "Price to Analyse",
            min_value=0.01,
            value=float(scenario_price),
            step=0.25,
        )
    with vc2:
        max_units = st.slider("Max Units to Model", 10, 500, 100, 10)

    vol_unit_margin = vol_price - unit_cost
    breakeven_vol   = unit_cost / vol_unit_margin if vol_unit_margin > 0 else float("inf")

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Price per Unit",          f"${vol_price:.2f}")
    kc2.metric("Unit Cost (est.)",         f"${unit_cost:.2f}")
    kc3.metric("Gross Margin per Unit",
               f"${vol_unit_margin:.2f}" if vol_unit_margin > 0 else "Below cost")
    kc4.metric("Breakeven Units",
               f"{breakeven_vol:.0f}" if breakeven_vol != float("inf") else "N/A")

    unit_range  = np.arange(1, max_units + 1)
    vol_revenues = unit_range * vol_price
    vol_margins  = unit_range * vol_unit_margin if vol_unit_margin > 0 else np.zeros_like(unit_range, dtype=float)
    vol_costs    = unit_range * unit_cost

    lv, rv = st.columns(2)

    with lv:
        st.markdown("### Revenue & Cost vs Units Sold")
        rcfig = go.Figure()
        rcfig.add_trace(go.Scatter(x=unit_range, y=vol_revenues,
                                   mode="lines", name="Revenue",
                                   line=dict(color="steelblue")))
        rcfig.add_trace(go.Scatter(x=unit_range, y=vol_costs,
                                   mode="lines", name="Total Cost",
                                   line=dict(color="tomato", dash="dash")))
        if breakeven_vol != float("inf") and breakeven_vol <= max_units:
            rcfig.add_vline(x=breakeven_vol, line_dash="dot", line_color="yellow",
                            annotation_text=f"Breakeven ({breakeven_vol:.0f} units)",
                            annotation_position="top right")
        rcfig.update_layout(template="plotly_dark",
                             xaxis_title="Units Sold", yaxis_title="Value ($)",
                             legend=dict(orientation="h", yanchor="bottom",
                                         y=1.02, xanchor="right", x=1))
        st.plotly_chart(rcfig, use_container_width=True)

    with rv:
        st.markdown("### Margin vs Units Sold")
        mfig = go.Figure()
        mfig.add_trace(go.Scatter(x=unit_range, y=vol_margins,
                                   mode="lines", name="Margin",
                                   line=dict(color="mediumseagreen"),
                                   fill="tozeroy",
                                   fillcolor="rgba(60,179,113,0.1)"))
        if breakeven_vol != float("inf") and breakeven_vol <= max_units:
            mfig.add_vline(x=breakeven_vol, line_dash="dot", line_color="yellow",
                           annotation_text=f"Breakeven ({breakeven_vol:.0f} units)",
                           annotation_position="top right")
        mfig.add_hline(y=0, line_color="gray", line_width=1)
        mfig.update_layout(template="plotly_dark",
                            xaxis_title="Units Sold", yaxis_title="Total Margin ($)")
        st.plotly_chart(mfig, use_container_width=True)

    # Volume comparison table
    st.markdown("### Volume Scenario Comparison")
    compare_units = [u for u in [5, 10, 20, 30, 50, 75, 100] if u <= max_units]
    compare_rows  = []
    for u in compare_units:
        tr  = u * vol_price
        tc  = u * unit_cost
        tm  = u * vol_unit_margin if vol_unit_margin > 0 else 0
        mp  = tm / tr * 100 if tr > 0 else 0
        be  = "✅" if vol_unit_margin > 0 and u >= breakeven_vol else "❌"
        compare_rows.append({
            "Units Sold": u,
            "Total Revenue": f"${tr:,.2f}",
            "Total Cost":    f"${tc:,.2f}",
            "Total Margin":  f"${tm:,.2f}",
            "Margin %":      f"{mp:.1f}%",
            "Above Breakeven": be,
        })
    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

    # Price vs volume tradeoff
    st.markdown("### Price vs Volume Tradeoff")
    tradeoff_units = np.arange(5, max_units + 1, 5)
    tfig = go.Figure()
    tfig.add_trace(go.Scatter(
        x=tradeoff_units,
        y=[u * current_price for u in tradeoff_units],
        mode="lines", name=f"Revenue @ Current (\\${current_price:.2f})",
        line=dict(color="tomato")
    ))
    tfig.add_trace(go.Scatter(
        x=tradeoff_units,
        y=[u * rev_opt_price for u in tradeoff_units],
        mode="lines", name=f"Revenue @ Optimal (\\${rev_opt_price:.2f})",
        line=dict(color="mediumpurple")
    ))
    tfig.add_trace(go.Scatter(
        x=tradeoff_units,
        y=[u * (current_price - unit_cost) if current_price > unit_cost else 0
           for u in tradeoff_units],
        mode="lines", name=f"Margin @ Current (\\${current_price:.2f})",
        line=dict(color="tomato", dash="dash")
    ))
    tfig.add_trace(go.Scatter(
        x=tradeoff_units,
        y=[u * (rev_opt_price - unit_cost) if rev_opt_price > unit_cost else 0
           for u in tradeoff_units],
        mode="lines", name=f"Margin @ Optimal (\\${rev_opt_price:.2f})",
        line=dict(color="mediumpurple", dash="dash")
    ))
    tfig.update_layout(
        template="plotly_dark",
        title="Revenue & Margin at Current vs Optimal Price Across Unit Volumes",
        xaxis_title="Units Sold", yaxis_title="Value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(tfig, use_container_width=True)

    tradeoff_rows = []
    for u in tradeoff_units:
        tradeoff_rows.append({
            "Units": u,
            "Min Breakeven Price":          f"${unit_cost:.2f}",
            "Revenue at Current Price":     f"${u * current_price:,.2f}",
            "Revenue at Optimal Price":     f"${u * rev_opt_price:,.2f}",
            "Margin at Current Price":      f"${u * (current_price - unit_cost) if current_price > unit_cost else 0:,.2f}",
            "Margin at Optimal Price":      f"${u * (rev_opt_price - unit_cost) if rev_opt_price > unit_cost else 0:,.2f}",
        })
    with st.expander("View full tradeoff table"):
        st.dataframe(pd.DataFrame(tradeoff_rows), use_container_width=True, hide_index=True)
        
# TAB 3 — DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Diagnostics")
    st.caption("Distribution, volatility, cost sensitivity, and Gradient Boosting model response curves")

    diag_df = df[(df["SKU"] == sku) & (df["Branch"] == branch)].copy()
    diag_df["Month"] = diag_df["Month"].astype(str)

    price_std = diag_df["Price"].std()

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Average Historical Price", f"${diag_df['Price'].mean():.2f}")
    d2.metric("Price Std. Dev.",          f"${price_std:.2f}" if pd.notna(price_std) else "$0.00")
    d3.metric("Min Historical Price",     f"${diag_df['Price'].min():.2f}")
    d4.metric("Max Historical Price",     f"${diag_df['Price'].max():.2f}")

    dl, dr = st.columns(2)

    with dl:
        st.markdown("### Price Distribution")
        dfig = px.histogram(diag_df, x="Price", nbins=10,
                            title="Historical Price Distribution")
        dfig.update_layout(template="plotly_dark")
        st.plotly_chart(dfig, use_container_width=True)

    with dr:
        st.markdown("### Historical Price Trend")
        tfig2 = px.line(diag_df, x="Month", y="Price",
                        title="Historical Price Trend")
        tfig2.add_hline(y=diag_df["Price"].mean(), line_dash="dash",
                        annotation_text="Average Price",
                        annotation_position="top left")
        tfig2.update_layout(template="plotly_dark")
        st.plotly_chart(tfig2, use_container_width=True)

    # GB model response curves (from precomputed sweep)
    dl2, dr2 = st.columns(2)
    sweep_prices   = current["prices"]
    sweep_units    = current["units"]
    sweep_revenues = current["revenues"]

    with dl2:
        st.markdown("### Gradient Boosting Model: Price vs Demand")
        demand_fig = go.Figure()
        demand_fig.add_trace(go.Scatter(x=sweep_prices, y=sweep_units,
                                        mode="lines", name="Predicted Demand",
                                        line=dict(color="steelblue")))
        demand_fig.add_vline(x=current_price,  line_dash="dash", line_color="red")
        demand_fig.add_vline(x=scenario_price, line_dash="dash", line_color="green")
        demand_fig.update_layout(template="plotly_dark",
                                  xaxis_title="Price ($)", yaxis_title="Predicted Units",
                                  title="Demand Response (GB Model)")
        st.plotly_chart(demand_fig, use_container_width=True)

    with dr2:
        st.markdown("### GB Model: Price vs Revenue")
        rev_fig = go.Figure()
        rev_fig.add_trace(go.Scatter(x=sweep_prices, y=sweep_revenues,
                                     mode="lines", name="Predicted Revenue",
                                     line=dict(color="mediumseagreen")))
        rev_fig.add_vline(x=current_price,  line_dash="dash", line_color="red")
        rev_fig.add_vline(x=rev_opt_price,  line_dash="dash", line_color="purple")
        rev_fig.update_layout(template="plotly_dark",
                               xaxis_title="Price ($)", yaxis_title="Predicted Revenue ($)",
                               title="Revenue Response (GB Model)")
        st.plotly_chart(rev_fig, use_container_width=True)

    # Cost sensitivity
    st.markdown("### Cost Sensitivity Analysis")
    st.caption("Margin at current and optimal price under different cost assumptions")
    cost_rows = []
    for mult in np.linspace(0.50, 0.80, 7):
        alt_cost   = diag_df["Price"].mean() * mult
        alt_margin = (current_price - alt_cost) * current["predicted_units"] if current_price > alt_cost else 0
        alt_opt_margin = (rev_opt_price - alt_cost) * rev_opt["predicted_units"] if rev_opt_price > alt_cost else 0
        cost_rows.append({
            "Cost Assumption": f"{int(mult * 100)}% of avg price",
            "Implied Unit Cost": f"${alt_cost:.2f}",
            "Margin @ Current Price":  round(alt_margin, 2),
            "Margin @ Optimal Price":  round(alt_opt_margin, 2),
        })
    st.dataframe(pd.DataFrame(cost_rows), use_container_width=True, hide_index=True)

    # Pricing context
    st.markdown("### Pricing Context")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown(f"**SKU:** {sku}")
        st.markdown(f"**Branch:** {branch}")
        st.markdown(f"**Category:** {snap['category']}")
        st.markdown(f"**Unit Cost:** \\${unit_cost:.2f}")
        st.markdown(f"**Historical Mean Price:** \\${diag_df['Price'].mean():.2f}")
    with pc2:
        st.markdown(f"**Current Price:** \\${current_price:.2f}")
        st.markdown(f"**Scenario Price:** \\${scenario_price:.2f}")
        st.markdown(f"**Revenue-Optimal Price:** \\${rev_opt_price:.2f}")
        st.markdown(f"**Margin-Optimal Price:** \\${margin_opt_price:.2f}")
        st.markdown(f"**Historical Price Range:** \\${hist_min:.2f} to \\${hist_max:.2f}")

# TAB 4 — PORTFOLIO VIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Portfolio View")

    portfolio_branch = st.selectbox(
        "Filter Portfolio by Branch",
        options=["All"] + sorted(df["Branch"].dropna().unique().tolist()),
        key="portfolio_branch_filter"
    )

    @st.cache_data
    def get_portfolio(branch_filter: str):
        return get_portfolio_summary(df, sweeps, branch_filter)

    portfolio_df = get_portfolio(portfolio_branch)

    if portfolio_df.empty:
        st.warning("No portfolio data available.")
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("SKUs Reviewed",             len(portfolio_df["SKU"].unique()))
        p2.metric("Largest Price Gap (%)",     f"{portfolio_df['Revenue Gap (%)'].max():.2f}%")
        p3.metric("Average Price Gap (%)",     f"{portfolio_df['Revenue Gap (%)'].mean():.2f}%")
        p4.metric("Avg Revenue Upside (%)",    f"{portfolio_df['Revenue Upside (%)'].mean():.2f}%")

        pl, pr = st.columns(2)

        with pl:
            st.markdown("### Top Pricing Opportunities")
            top_fig = px.bar(
                portfolio_df.sort_values("Revenue Upside (%)", ascending=False).head(10),
                x="SKU", y="Revenue Upside (%)",
                title="Top 10 SKUs by Revenue Upside"
            )
            top_fig.update_layout(template="plotly_dark")
            st.plotly_chart(top_fig, use_container_width=True)

        with pr:
            st.markdown("### Portfolio Price Opportunity Map")
            opp_fig = px.scatter(
                portfolio_df,
                x="Current Price", y="Revenue-Optimal Price",
                size="Revenue Upside (%)", hover_name="SKU",
                color="Branch",
                title="Current vs Revenue-Optimal Price"
            )
            opp_fig.update_layout(template="plotly_dark")
            st.plotly_chart(opp_fig, use_container_width=True)

        st.markdown("### Portfolio Pricing Table")
        st.dataframe(
            portfolio_df.sort_values("Revenue Upside (%)", ascending=False),
            use_container_width=True
        )

        # Downloads
        csv_bytes = portfolio_df.to_csv(index=False).encode("utf-8")

        def to_excel(d: dict) -> bytes | None:
            try:
                import openpyxl  # noqa
            except ImportError:
                return None
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                for sheet, frame in d.items():
                    frame.to_excel(w, sheet_name=sheet[:31], index=False)
            buf.seek(0)
            return buf.read()

        excel_bytes = to_excel({"Portfolio View": portfolio_df})

        dl_c1, dl_c2 = st.columns(2)
        with dl_c1:
            st.download_button("Download CSV",
                               data=csv_bytes,
                               file_name="portfolio_view.csv",
                               mime="text/csv")
        with dl_c2:
            if excel_bytes:
                st.download_button("Download Excel",
                                   data=excel_bytes,
                                   file_name="portfolio_view.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.warning("Install openpyxl for Excel export: `pip install openpyxl`")


# TAB 5 — MODEL DETAILS
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader("Model Details")

    st.markdown("""
    ### What the model does
    An **XGBoost gradient boosting** model is trained on your historical pricing data
    when the app first loads. It learns the relationship between price and units sold,
    controlling for product category, branch, and month.

    ### Features used
    - **Price** — the primary input lever
    - **Category** — product type (Battery, Brake, Lighting, Tire, etc.)
    - **Branch** — location (Atlanta, Dallas, Houston, Phoenix, etc.)
    - **Month** — captures any residual seasonal patterns in the training data

    ### How the simulator uses it
    On startup, the model pre-computes a full price-demand curve for every
    SKU × Branch combination (±30% around historical average price).
    When you move the slider, the app interpolates from that curve —
    no re-inference happens at runtime, so the response is instant.

    ### Optimisation
    The revenue-optimal and margin-optimal prices are identified from the
    same precomputed sweep. Both use a mid-year (June) baseline to represent
    general pricing decisions rather than a specific seasonal month.

    ### Limitations
    - Trained on synthetic data — predictions reflect simulated patterns, not real elasticity
    - With ~12 observations per SKU/Branch the model has limited price variation to learn from
    - No competitor pricing, promotions, or external signals are included yet
    - When you replace this with real company data, retrain by restarting the app
    """)

    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("### Selected SKU Parameters")
        st.markdown(f"**SKU:** {sku}")
        st.markdown(f"**Branch:** {branch}")
        st.markdown(f"**Category:** {snap['category']}")
        st.markdown(f"**Current Price:** \\${current_price:.2f}")
        st.markdown(f"**Unit Cost:** \\${unit_cost:.2f}")

    with mc2:
        st.markdown("### Optimisation Outputs")
        st.markdown(f"**Revenue-Optimal Price:** \\${rev_opt_price:.2f}")
        st.markdown(f"**Margin-Optimal Price:** \\${margin_opt_price:.2f}")
        st.markdown(f"**Revenue @ Optimal:** \\${rev_opt['revenue']:,.2f}")
        st.markdown(f"**Margin @ Optimal:** \\${margin_opt['margin']:,.2f}")
        st.markdown(f"**Predicted Units @ Current:** {current['predicted_units']:.1f}")
        st.markdown(f"**Predicted Units @ Optimal:** {rev_opt['predicted_units']:.1f}")

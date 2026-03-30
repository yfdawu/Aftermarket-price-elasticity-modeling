import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

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

# CLEAN COLUMN NAMES
df.columns = [col.strip() for col in df.columns]

# CHECK REQUIRED COLUMNS
required_cols = ["SKU", "Branch", "Price", "Month"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing required columns in synthetic_pricing_data.csv: {missing_cols}")
    st.stop()

# HELPERS
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def prepare_months(input_df: pd.DataFrame) -> pd.DataFrame:
    out = input_df.copy()
    if "Month" in out.columns and out["Month"].dtype == object:
        out["Month"] = out["Month"].astype(str).str.strip()
        unknown_months = set(out["Month"].dropna()) - set(month_order)
        if unknown_months:
            st.warning(f"Unrecognized month values found and excluded: {unknown_months}")
        out["Month"] = pd.Categorical(
            out["Month"],
            categories=month_order,
            ordered=True
        )
        out = out.sort_values("Month").dropna(subset=["Month"])
    return out

def simulate(price, peak_rev_price, max_rev, cost):
    curvature = max_rev / ((peak_rev_price * 0.25) ** 2)
    revenue = max_rev - curvature * (price - peak_rev_price) ** 2
    revenue = max(revenue, 0)
    demand = revenue / price if price > 0 else 0
    margin = (price - cost) * demand if price > cost else 0
    return demand, revenue, margin

@st.cache_data
def build_portfolio_table(source_df: pd.DataFrame) -> pd.DataFrame:
    portfolio_rows = []

    for sku_name in sorted(source_df["SKU"].dropna().unique()):
        sku_df = source_df[source_df["SKU"] == sku_name].copy()
        if sku_df.empty:
            continue

        sku_df = prepare_months(sku_df)
        current_sku_price = float(sku_df["Price"].iloc[-1])
        avg_sku_price = float(sku_df["Price"].mean())
        unit_cost_sku = avg_sku_price * 0.63
        peak_rev_sku = avg_sku_price * 1.02
        max_rev_sku = avg_sku_price * 100

        sku_prices = np.linspace(current_sku_price * 0.8, current_sku_price * 1.2, 50)
        sku_results = [simulate(p, peak_rev_sku, max_rev_sku, unit_cost_sku) for p in sku_prices]
        sku_revenues = [r[1] for r in sku_results]
        sku_margins = [r[2] for r in sku_results]

        sku_rev_opt_idx = int(np.argmax(sku_revenues))
        sku_margin_opt_idx = int(np.argmax(sku_margins))

        sku_rev_opt_price = float(sku_prices[sku_rev_opt_idx])
        sku_margin_opt_price = float(sku_prices[sku_margin_opt_idx])

        current_demand, current_revenue, current_margin = simulate(
            current_sku_price, peak_rev_sku, max_rev_sku, unit_cost_sku
        )
        _, optimal_revenue_value, _ = simulate(
            sku_rev_opt_price, peak_rev_sku, max_rev_sku, unit_cost_sku
        )

        revenue_gap_pct = ((sku_rev_opt_price - current_sku_price) / current_sku_price) * 100 if current_sku_price != 0 else 0
        revenue_upside_pct = ((optimal_revenue_value - current_revenue) / current_revenue) * 100 if current_revenue != 0 else 0

        portfolio_rows.append({
            "SKU": sku_name,
            "Current Price": round(current_sku_price, 2),
            "Revenue-Optimal Price": round(sku_rev_opt_price, 2),
            "Margin-Optimal Price": round(sku_margin_opt_price, 2),
            "Revenue Gap (%)": round(revenue_gap_pct, 2),
            "Revenue Upside (%)": round(revenue_upside_pct, 2),
            "Current Revenue": round(current_revenue, 2),
            "Optimal Revenue": round(optimal_revenue_value, 2),
            "Observations": len(sku_df)
        })

    return pd.DataFrame(portfolio_rows)

def to_excel_bytes(dataframe_dict: dict) -> bytes | None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        return None

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df_sheet in dataframe_dict.items():
            df_sheet.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    output.seek(0)
    return output.read()

# SIDEBAR CONTROLS
st.sidebar.header("Scenario Inputs")

sku = st.sidebar.selectbox("Select Product (SKU)", sorted(df["SKU"].dropna().unique()))
branch = st.sidebar.selectbox("Select Branch", sorted(df["Branch"].dropna().unique()))

filtered_df = df[(df["SKU"] == sku) & (df["Branch"] == branch)].copy()
filtered_df = prepare_months(filtered_df)

if filtered_df.empty:
    st.error("No data found for the selected SKU and Branch.")
    st.stop()

current_price = float(filtered_df["Price"].iloc[-1])

sku_branch_key = f"{sku}__{branch}"
if st.session_state.get("last_sku_branch") != sku_branch_key:
    st.session_state["scenario_price_input"] = current_price
    st.session_state["last_sku_branch"] = sku_branch_key

price_change_pct = st.sidebar.slider(
    "Price Change (%)",
    min_value=-20,
    max_value=20,
    value=0,
    step=1,
    key="price_change_pct"
)

slider_driven_price = round(current_price * (1 + price_change_pct / 100), 2)

scenario_price = st.sidebar.number_input(
    "Or Enter Scenario Price Directly",
    min_value=0.0,
    value=float(slider_driven_price),
    step=0.25,
    key="scenario_price_input"
)

hist_avg_price = float(filtered_df["Price"].mean())
unit_cost = hist_avg_price * 0.63
peak_revenue_price = hist_avg_price * 1.02
max_revenue = hist_avg_price * 100

current_demand, current_revenue, current_margin = simulate(
    current_price, peak_revenue_price, max_revenue, unit_cost
)
scenario_demand, scenario_revenue, scenario_margin = simulate(
    scenario_price, peak_revenue_price, max_revenue, unit_cost
)

demand_change = ((scenario_demand - current_demand) / current_demand) * 100 if current_demand != 0 else 0
revenue_change = ((scenario_revenue - current_revenue) / current_revenue) * 100 if current_revenue != 0 else 0
margin_change = ((scenario_margin - current_margin) / current_margin) * 100 if current_margin != 0 else 0

if abs(demand_change) < 5:
    pricing_signal = "Resilient"
elif abs(demand_change) < 10:
    pricing_signal = "Moderate"
else:
    pricing_signal = "Sensitive"

# OPTIMIZATION SWEEP
prices = np.linspace(current_price * 0.8, current_price * 1.2, 50)
results = [simulate(p, peak_revenue_price, max_revenue, unit_cost) for p in prices]

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

hist_context = "Within historical price range" if hist_min <= scenario_price <= hist_max else "Outside historical price range"

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

# PAGE HEADER
st.title("Pricing Analytics Simulator")
st.caption("Evaluate pricing scenarios across demand, revenue, and margin tradeoffs")

# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "Scenario Simulator",
    "Diagnostics",
    "Portfolio View",
    "Model Details"
])

# ── TAB 1: SCENARIO SIMULATOR ──────────────────────────────────────────────
with tab1:
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    row1_col1.metric("Current Price", f"${current_price:.2f}")
    row1_col2.metric("Scenario Price", f"${scenario_price:.2f}")
    row1_col3.metric("Pricing Signal", pricing_signal)
    row2_col1.metric("Demand Change", f"{demand_change:.2f}%")
    row2_col2.metric("Revenue Change", f"{revenue_change:.2f}%")
    row2_col3.metric("Margin Change", f"{margin_change:.2f}%")

    if abs(scenario_price - current_price) < 0.01:
        st.info("No price change selected yet")
    elif revenue_change > 0 and margin_change > 0 and abs(demand_change) < 5:
        st.success("Controlled price increase appears viable")
    elif revenue_change > 0 and abs(demand_change) < 10:
        st.warning("Revenue may improve, but customer response should be monitored")
    else:
        st.warning("Scenario may weaken overall performance")

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
        st.markdown(f"**Revenue-Optimal Price:** \\${rev_opt_price:.2f}")
        st.markdown(f"**Margin-Optimal Price:** \\${margin_opt_price:.2f}")
        low = rev_opt_price * 0.98
        high = rev_opt_price * 1.02
        st.markdown(f"**Suggested Test Range:** \\${low:.2f} to \\${high:.2f}")

    view = st.radio("View Optimization", ["Revenue", "Margin"], horizontal=True)

    chart_data = pd.DataFrame({
        "Price": prices,
        "Demand": demands,
        "Revenue": revenues,
        "Margin": margins
    })

    if view == "Revenue":
        fig = px.line(chart_data, x="Price", y="Revenue", title="Revenue vs Price")
        fig.add_vrect(
            x0=low, x1=high,
            fillcolor="green", opacity=0.08, line_width=0,
            annotation_text="Recommended Zone", annotation_position="top left"
        )
        fig.add_vline(x=current_price, line_dash="dash", line_color="red")
        fig.add_vline(x=scenario_price, line_dash="dash", line_color="green")
        fig.add_vline(x=rev_opt_price, line_dash="dash", line_color="purple")
        fig.add_trace(go.Scatter(
            x=[current_price, scenario_price, rev_opt_price],
            y=[current_revenue, scenario_revenue, rev_opt_value],
            mode="markers+text",
            text=["Current", "Scenario", "Optimal"],
            textposition=["top left", "top right", "top center"],
            marker=dict(size=10),
            showlegend=False
        ))
        compare_df = pd.DataFrame({
            "Category": ["Current", "Scenario", "Optimal"],
            "Value": [current_revenue, scenario_revenue, rev_opt_value]
        })
    else:
        margin_low = margin_opt_price * 0.98
        margin_high = margin_opt_price * 1.02
        fig = px.line(chart_data, x="Price", y="Margin", title="Margin vs Price")
        fig.add_vrect(
            x0=margin_low, x1=margin_high,
            fillcolor="green", opacity=0.08, line_width=0,
            annotation_text="Recommended Zone", annotation_position="top left"
        )
        fig.add_vline(x=current_price, line_dash="dash", line_color="red")
        fig.add_vline(x=scenario_price, line_dash="dash", line_color="green")
        fig.add_vline(x=margin_opt_price, line_dash="dash", line_color="purple")
        fig.add_trace(go.Scatter(
            x=[current_price, scenario_price, margin_opt_price],
            y=[current_margin, scenario_margin, margin_opt_value],
            mode="markers+text",
            text=["Current", "Scenario", "Optimal"],
            textposition=["top left", "top right", "top center"],
            marker=dict(size=10),
            showlegend=False
        ))
        compare_df = pd.DataFrame({
            "Category": ["Current", "Scenario", "Optimal"],
            "Value": [current_margin, scenario_margin, margin_opt_value]
        })

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        m1, m2, m3 = st.columns(3)
        if view == "Revenue":
            m1.metric("Current Revenue", f"${current_revenue:,.0f}")
            m2.metric("Scenario Revenue", f"${scenario_revenue:,.0f}")
            m3.metric("Optimal Revenue", f"${rev_opt_value:,.0f}")
        else:
            m1.metric("Current Margin", f"${current_margin:,.0f}")
            m2.metric("Scenario Margin", f"${scenario_margin:,.0f}")
            m3.metric("Optimal Margin", f"${margin_opt_value:,.0f}")

    with c2:
        compare_fig = px.bar(
            compare_df, x="Category", y="Value",
            title="Current vs Scenario vs Optimal"
        )
        compare_fig.update_layout(template="plotly_dark", height=280)
        st.plotly_chart(compare_fig, use_container_width=True)

    st.subheader("Historical Price Trend")
    hist_fig = px.line(filtered_df, x="Month", y="Price", title="Price Over Time")
    hist_fig.add_hline(
        y=hist_avg_price,
        line_dash="dash",
        annotation_text="Historical Average",
        annotation_position="top left"
    )
    hist_fig.update_layout(template="plotly_dark")
    st.plotly_chart(hist_fig, use_container_width=True)

# ── TAB 2: DIAGNOSTICS ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Diagnostics")
    st.caption("Distribution, volatility, cost sensitivity, and modeled response curves for the selected SKU")

    insights_df = filtered_df.copy()
    insights_df["Month"] = insights_df["Month"].astype(str)

    price_std = insights_df["Price"].std()

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Average Historical Price", f"${insights_df['Price'].mean():.2f}")
    top2.metric("Price Std. Dev.", f"${price_std:.2f}" if pd.notna(price_std) else "$0.00")
    top3.metric("Min Historical Price", f"${insights_df['Price'].min():.2f}")
    top4.metric("Max Historical Price", f"${insights_df['Price'].max():.2f}")

    modeled_prices = np.linspace(
        insights_df["Price"].min() * 0.85,
        insights_df["Price"].max() * 1.15,
        40
    )
    modeled_results = [simulate(p, peak_revenue_price, max_revenue, unit_cost) for p in modeled_prices]
    modeled_demands = [r[0] for r in modeled_results]
    modeled_revenues = [r[1] for r in modeled_results]

    modeled_df = pd.DataFrame({
        "Price": modeled_prices,
        "Modeled Demand": modeled_demands,
        "Modeled Revenue": modeled_revenues
    })

    left_i, right_i = st.columns(2)

    with left_i:
        st.markdown("### Price Distribution")
        dist_fig = px.histogram(
            insights_df, x="Price", nbins=10,
            title="Historical Price Distribution"
        )
        dist_fig.update_layout(template="plotly_dark")
        st.plotly_chart(dist_fig, use_container_width=True)

    with right_i:
        st.markdown("### Historical Price Trend")
        trend_fig = px.line(
            insights_df, x="Month", y="Price",
            title="Historical Price Trend"
        )
        trend_fig.add_hline(
            y=insights_df["Price"].mean(),
            line_dash="dash",
            annotation_text="Average Price",
            annotation_position="top left"
        )
        trend_fig.update_layout(template="plotly_dark")
        st.plotly_chart(trend_fig, use_container_width=True)

    left_i2, right_i2 = st.columns(2)

    with left_i2:
        st.markdown("### Modeled Price vs Demand")
        demand_fig = px.scatter(
            modeled_df, x="Price", y="Modeled Demand",
            title="Modeled Demand Response"
        )
        demand_fig.add_vline(x=current_price, line_dash="dash", line_color="red")
        demand_fig.add_vline(x=scenario_price, line_dash="dash", line_color="green")
        demand_fig.update_layout(template="plotly_dark")
        st.plotly_chart(demand_fig, use_container_width=True)

    with right_i2:
        st.markdown("### Modeled Price vs Revenue")
        revenue_scatter_fig = px.scatter(
            modeled_df, x="Price", y="Modeled Revenue",
            title="Modeled Revenue Response"
        )
        revenue_scatter_fig.add_vline(x=current_price, line_dash="dash", line_color="red")
        revenue_scatter_fig.add_vline(x=rev_opt_price, line_dash="dash", line_color="purple")
        revenue_scatter_fig.update_layout(template="plotly_dark")
        st.plotly_chart(revenue_scatter_fig, use_container_width=True)

    # Cost sensitivity table 
    st.markdown("### Cost Sensitivity Analysis")
    st.caption("Shows how margin changes under different unit cost assumptions")
    cost_multipliers = np.linspace(0.50, 0.80, 7)
    cost_rows = []
    for mult in cost_multipliers:
        alt_cost = hist_avg_price * mult
        _, _, alt_margin = simulate(current_price, peak_revenue_price, max_revenue, alt_cost)
        _, _, alt_opt_margin = simulate(rev_opt_price, peak_revenue_price, max_revenue, alt_cost)
        cost_rows.append({
            "Cost Assumption (% of avg price)": f"{int(mult * 100)}%",
            "Margin at Current Price": round(alt_margin, 2),
            "Margin at Revenue-Optimal Price": round(alt_opt_margin, 2)
        })
    cost_sensitivity_df = pd.DataFrame(cost_rows)
    st.dataframe(cost_sensitivity_df, use_container_width=True)

    st.markdown("### Pricing Context")
    dcol1, dcol2 = st.columns(2)

    with dcol1:
        st.markdown(f"**Selected SKU:** {sku}")
        st.markdown(f"**Selected Branch:** {branch}")
        st.markdown(f"**Historical Mean Price:** \\${hist_avg_price:.2f}")
        st.markdown(f"**Estimated Unit Cost:** \\${unit_cost:.2f}")
        st.markdown(f"**Revenue-Peak Anchor Price:** \\${peak_revenue_price:.2f}")

    with dcol2:
        st.markdown(f"**Current Price:** \\${current_price:.2f}")
        st.markdown(f"**Scenario Price:** \\${scenario_price:.2f}")
        st.markdown(f"**Revenue-Optimal Price:** \\${rev_opt_price:.2f}")
        st.markdown(f"**Margin-Optimal Price:** \\${margin_opt_price:.2f}")
        st.markdown(f"**Historical Price Range:** \\${hist_min:.2f} to \\${hist_max:.2f}")

# ── TAB 3: PORTFOLIO VIEW ──────────────────────────────────────────────────
with tab3:
    st.subheader("Portfolio View")

    portfolio_branch = st.selectbox(
        "Filter Portfolio by Branch",
        options=["All"] + sorted(df["Branch"].dropna().unique().tolist()),
        key="portfolio_branch_filter"
    )

    portfolio_source_df = df.copy() if portfolio_branch == "All" else df[df["Branch"] == portfolio_branch].copy()
    portfolio_df = build_portfolio_table(portfolio_source_df)

    if portfolio_df.empty:
        st.warning("No portfolio data available.")
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("SKUs Reviewed", len(portfolio_df))
        p2.metric("Largest Price Gap (%)", f"{portfolio_df['Revenue Gap (%)'].max():.2f}%")
        p3.metric("Average Price Gap (%)", f"{portfolio_df['Revenue Gap (%)'].mean():.2f}%")
        p4.metric("Average Revenue Upside (%)", f"{portfolio_df['Revenue Upside (%)'].mean():.2f}%")

        left_p, right_p = st.columns(2)

        with left_p:
            st.markdown("### Top Pricing Opportunities")
            top_portfolio_fig = px.bar(
                portfolio_df.sort_values("Revenue Upside (%)", ascending=False).head(10),
                x="SKU", y="Revenue Upside (%)",
                title="Top 10 SKUs by Revenue Upside"
            )
            top_portfolio_fig.update_layout(template="plotly_dark")
            st.plotly_chart(top_portfolio_fig, use_container_width=True)

        with right_p:
            st.markdown("### Current vs Revenue-Optimal Price")
            gap_scatter_fig = px.scatter(
                portfolio_df,
                x="Current Price", y="Revenue-Optimal Price",
                size="Revenue Upside (%)", hover_name="SKU",
                title="Portfolio Price Opportunity Map"
            )
            gap_scatter_fig.update_layout(template="plotly_dark")
            st.plotly_chart(gap_scatter_fig, use_container_width=True)

        st.markdown("### Portfolio Pricing Table")
        st.dataframe(
            portfolio_df.sort_values("Revenue Upside (%)", ascending=False),
            use_container_width=True
        )

        csv_bytes = portfolio_df.to_csv(index=False).encode("utf-8")
        excel_bytes = to_excel_bytes({"Portfolio View": portfolio_df})

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                label="Download Portfolio CSV",
                data=csv_bytes,
                file_name="portfolio_view.csv",
                mime="text/csv"
            )
        with dl2:
            if excel_bytes is not None:
                st.download_button(
                    label="Download Portfolio Excel",
                    data=excel_bytes,
                    file_name="portfolio_view.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("Excel export unavailable. Install openpyxl: `pip install openpyxl`")

# ── TAB 4: MODEL DETAILS ───────────────────────────────────────────────────
with tab4:
    st.subheader("Model Details")

    st.markdown("""
    ### Overview
    This simulator evaluates pricing scenarios by mapping candidate prices to modeled demand, revenue, and gross margin outcomes.

    ### Pricing Logic
    - A smooth quadratic revenue-response curve is used as a pricing surface.
    - Demand is inferred as revenue divided by price.
    - Margin is computed using an estimated unit cost based on historical average price.

    ### Optimization
    For each selected SKU and branch, the model evaluates a range of candidate prices and identifies:
    - the revenue-optimal price
    - the margin-optimal price

    ### Decision Guidance
    The simulator classifies scenarios using:
    - distance from modeled optimal price
    - magnitude of demand change
    - historical price context
    - available number of observations

    ### Current Assumptions
    - Data in this project is synthetic
    - Unit cost is estimated as 63% of historical average price
    - Revenue peak is anchored near the historical average price

    ### Limitations
    - This is a decision-support prototype, not a production pricing engine
    - It does not yet incorporate competitor prices, promotions, or customer-level heterogeneity
    - Demand behavior is approximated using a stylized response curve
    """)

    details_col1, details_col2 = st.columns(2)

    with details_col1:
        st.markdown("### Selected SKU Parameters")
        st.markdown(f"**SKU:** {sku}")
        st.markdown(f"**Branch:** {branch}")
        st.markdown(f"**Current Price:** \\${current_price:.2f}")
        st.markdown(f"**Average Historical Price:** \\${hist_avg_price:.2f}")
        st.markdown(f"**Estimated Unit Cost:** \\${unit_cost:.2f}")

    with details_col2:
        st.markdown("### Optimization Outputs")
        st.markdown(f"**Revenue-Optimal Price:** \\${rev_opt_price:.2f}")
        st.markdown(f"**Margin-Optimal Price:** \\${margin_opt_price:.2f}")
        st.markdown(f"**Revenue-Optimal Value:** \\${rev_opt_value:,.2f}")
        st.markdown(f"**Margin-Optimal Value:** \\${margin_opt_value:,.2f}")

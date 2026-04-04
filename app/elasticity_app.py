"""
Pricing Elasticity Simulator
Streamlit app — run with: streamlit run elasticity_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from elasticity_model import (
    load_and_preprocess,
    train_model,
    evaluate_model,
    price_sweep,
    comp_price_sweep,
    build_features,
    MACRO_FEATURES,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "Data"
TRAIN_PATH = DATA_DIR / "PROJ_PELAST_TRAIN v2.csv"
TEST_PATH = DATA_DIR / "PROJ_PELAST_TEST.csv"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pricing Elasticity Simulator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card { background:#f8f9fa; border-radius:8px; padding:12px 16px; margin-bottom:8px; }
    .zone-elastic { color:#d62728; font-weight:700; }
    .zone-inelastic { color:#2ca02c; font-weight:700; }
    .zone-unitary { color:#ff7f0e; font-weight:700; }
    h1 { font-size:1.6rem !important; }
    h2 { font-size:1.25rem !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model training (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training XGBoost model on TRAIN_v2...")
def get_model():
    train_df = load_and_preprocess(TRAIN_PATH, is_train=True)
    test_df = load_and_preprocess(TEST_PATH, is_train=False)
    model, feature_cols = train_model(train_df)
    test_metrics = evaluate_model(model, feature_cols, test_df)
    train_metrics = evaluate_model(model, feature_cols, train_df)
    return model, feature_cols, train_df, test_df, train_metrics, test_metrics


@st.cache_data(show_spinner=False)
def load_raw_test():
    return pd.read_csv(TEST_PATH)


model, feature_cols, train_df, test_df, train_metrics, test_metrics = get_model()

# Keep comp_price for simulation (dropped during preprocessing)
raw_test = load_raw_test()
if "(No column name)" in raw_test.columns:
    raw_test = raw_test.rename(columns={"(No column name)": "invoice_date"})
raw_test["invoice_date"] = pd.to_datetime(raw_test["invoice_date"])
# Attach comp_price back to test_df index-aligned
test_df = test_df.copy()
test_df["comp_price"] = raw_test["comp_price"].values[: len(test_df)]

parts = sorted(test_df["PartID"].unique())
custs = sorted(test_df["CustID"].unique())

# ---------------------------------------------------------------------------
# Sidebar — simulation controls
# ---------------------------------------------------------------------------

st.sidebar.title("Simulation Controls")

selected_part = st.sidebar.selectbox("Part ID", parts, format_func=lambda x: x.split("_")[1][:12])
selected_cust = st.sidebar.selectbox("Customer ID", custs, format_func=lambda x: x.split("_")[1][:12])
selected_oem = st.sidebar.radio("OEM Flag", [1, 0], format_func=lambda x: "OEM" if x == 1 else "Non-OEM")

# Filter to matching test rows
mask = (
    (test_df["PartID"] == selected_part)
    & (test_df["CustID"] == selected_cust)
    & (test_df["oem_flag"] == selected_oem)
)
segment_df = test_df[mask]

if segment_df.empty:
    # Loosen to part + oem
    mask = (test_df["PartID"] == selected_part) & (test_df["oem_flag"] == selected_oem)
    segment_df = test_df[mask]

if segment_df.empty:
    segment_df = test_df[test_df["PartID"] == selected_part]

if segment_df.empty:
    segment_df = test_df.copy()

# Build base row as median of segment
base_row = segment_df.median(numeric_only=True)
for col in ["PartID", "CustID", "oem_flag"]:
    base_row[col] = (
        selected_part if col == "PartID"
        else selected_cust if col == "CustID"
        else selected_oem
    )

base_price = float(base_row["unit_retail"])
base_comp = float(base_row.get("comp_price", 0))

st.sidebar.markdown("---")
st.sidebar.subheader("Price Range for Simulation")
price_low_pct = st.sidebar.slider("Min price (% of base)", 30, 90, 50, step=5)
price_high_pct = st.sidebar.slider("Max price (% of base)", 110, 300, 200, step=10)
price_range = (price_low_pct / 100, price_high_pct / 100)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Base price: **${base_price:.2f}**  \n"
    f"Comp price: **${base_comp:.2f}**  \n"
    f"N segment rows: **{len(segment_df)}**"
)

# ---------------------------------------------------------------------------
# Precompute sweeps
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_price_sweep(_model, feature_cols, base_row_dict, price_range):
    base = pd.Series(base_row_dict)
    return price_sweep(_model, feature_cols, base, price_range=price_range)


@st.cache_data(show_spinner=False)
def get_comp_sweep(_model, feature_cols, base_row_dict):
    base = pd.Series(base_row_dict)
    return comp_price_sweep(_model, feature_cols, base)


@st.cache_data(show_spinner=False)
def get_oem_sweeps(_model, feature_cols, base_row_dict, price_range):
    base_oem = pd.Series({**base_row_dict, "oem_flag": 1})
    base_non = pd.Series({**base_row_dict, "oem_flag": 0})
    return (
        price_sweep(_model, feature_cols, base_oem, price_range=price_range),
        price_sweep(_model, feature_cols, base_non, price_range=price_range),
    )


base_dict = base_row.to_dict()
sweep_df = get_price_sweep(model, feature_cols, base_dict, price_range)
comp_df = get_comp_sweep(model, feature_cols, base_dict)
oem_df, non_oem_df = get_oem_sweeps(model, feature_cols, base_dict, price_range)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Pricing Elasticity Simulator")
st.caption(
    f"Part: `{selected_part.split('_')[1][:16]}`  |  "
    f"Customer: `{selected_cust.split('_')[1][:16]}`  |  "
    f"OEM: {'Yes' if selected_oem == 1 else 'No'}"
)

# Point elasticity at base price
base_idx = (sweep_df["unit_retail"] - base_price).abs().idxmin()
base_elasticity = sweep_df.loc[base_idx, "elasticity"]
base_qty = sweep_df.loc[base_idx, "predicted_quantity"]
base_zone = sweep_df.loc[base_idx, "elasticity_zone"]

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Base Price", f"${base_price:.2f}")
k2.metric("Predicted Demand", f"{base_qty:.1f} units")
k3.metric("Point Elasticity", f"{base_elasticity:.3f}")
k4.metric("Elasticity Zone", base_zone)
k5.metric("Test R²", f"{test_metrics['r2']:.3f}")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Elasticity Simulator",
    "Competitor Gap Analysis",
    "OEM vs Non-OEM",
    "Model Performance",
])

# ============================================================
# TAB 1: Elasticity Simulator
# ============================================================
with tab1:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Price-Demand Curve with Elasticity Zones")

        # Color bands: elastic (red), inelastic (green)
        elastic_prices = sweep_df[sweep_df["elasticity_zone"] == "Elastic"]["unit_retail"]
        inelastic_prices = sweep_df[sweep_df["elasticity_zone"] == "Inelastic"]["unit_retail"]

        fig = go.Figure()

        # Background shading for elastic zone
        if not elastic_prices.empty:
            fig.add_vrect(
                x0=elastic_prices.min(), x1=elastic_prices.max(),
                fillcolor="rgba(214,39,40,0.08)", line_width=0,
                annotation_text="Elastic zone", annotation_position="top left",
                annotation_font_color="#d62728",
            )

        # Background shading for inelastic zone
        if not inelastic_prices.empty:
            fig.add_vrect(
                x0=inelastic_prices.min(), x1=inelastic_prices.max(),
                fillcolor="rgba(44,160,44,0.08)", line_width=0,
                annotation_text="Inelastic zone", annotation_position="top right",
                annotation_font_color="#2ca02c",
            )

        # Demand curve
        fig.add_trace(go.Scatter(
            x=sweep_df["unit_retail"],
            y=sweep_df["predicted_quantity"],
            mode="lines",
            name="Predicted Demand",
            line=dict(color="#1f77b4", width=2.5),
        ))

        # Revenue curve (secondary axis)
        fig.add_trace(go.Scatter(
            x=sweep_df["unit_retail"],
            y=sweep_df["revenue"],
            mode="lines",
            name="Revenue",
            line=dict(color="#9467bd", width=1.5, dash="dot"),
            yaxis="y2",
        ))

        # Current price marker
        fig.add_vline(
            x=base_price,
            line_dash="dash",
            line_color="#ff7f0e",
            annotation_text=f"Current ${base_price:.2f}",
            annotation_position="top",
            annotation_font_color="#ff7f0e",
        )

        fig.update_layout(
            xaxis_title="Unit Retail Price ($)",
            yaxis_title="Predicted Quantity (units)",
            yaxis2=dict(
                title="Revenue ($)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(x=0.01, y=0.99),
            height=420,
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Elasticity by Price Point")

        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=sweep_df["unit_retail"],
            y=sweep_df["elasticity"],
            mode="lines",
            line=dict(color="#8c564b", width=2),
            name="Point Elasticity",
        ))
        fig_e.add_hline(y=-1, line_dash="dash", line_color="gray",
                        annotation_text="Unit elastic (ε = -1)", annotation_position="bottom right")
        fig_e.add_hline(y=0, line_color="lightgray", line_width=0.8)
        fig_e.add_vline(x=base_price, line_dash="dash", line_color="#ff7f0e")

        # Color fill above/below -1
        fig_e.add_trace(go.Scatter(
            x=sweep_df["unit_retail"],
            y=sweep_df["elasticity"].clip(upper=-1),
            fill="tozeroy",
            fillcolor="rgba(214,39,40,0.12)",
            line=dict(width=0),
            showlegend=False,
            name="Elastic fill",
        ))

        fig_e.update_layout(
            xaxis_title="Unit Retail Price ($)",
            yaxis_title="Price Elasticity of Demand",
            height=420,
            margin=dict(t=30, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_e, use_container_width=True)

    # Interpretation panel
    st.subheader("Elasticity Interpretation")
    ia, ib, ic = st.columns(3)

    revenue_max_idx = sweep_df["revenue"].idxmax()
    revenue_max_price = sweep_df.loc[revenue_max_idx, "unit_retail"]
    revenue_max_qty = sweep_df.loc[revenue_max_idx, "predicted_quantity"]

    ia.markdown(f"""
**Current price:** ${base_price:.2f}
**Point elasticity:** {base_elasticity:.3f}
**Zone:** {base_zone}
    """)

    ib.markdown(f"""
**Revenue-maximizing price:** ${revenue_max_price:.2f}
**Expected demand at max rev:** {revenue_max_qty:.1f} units
**Price gap to max-rev:** {((revenue_max_price - base_price) / base_price * 100):+.1f}%
    """)

    # Elasticity guide
    ic.markdown("""
**Zone guide:**
- |ε| > 1 → **Elastic** — demand sensitive to price, revenue rises with cuts
- |ε| < 1 → **Inelastic** — demand stable, revenue rises with increases
- |ε| = 1 → **Unitary** — revenue-maximizing point
    """)

# ============================================================
# TAB 2: Competitor Gap Analysis
# ============================================================
with tab2:
    st.subheader("Demand vs. Competitor Price Gap")
    st.caption(
        "Unit retail price held constant. Competitor price varies — showing how your "
        "relative positioning (price_vs_comp = our price − comp price) drives demand."
    )

    if base_comp == 0:
        st.info(
            "No competitor price recorded for this segment. "
            "Showing simulated competitor range based on current unit retail."
        )

    col_c1, col_c2 = st.columns([2, 1])

    with col_c1:
        fig_comp = go.Figure()

        fig_comp.add_trace(go.Scatter(
            x=comp_df["price_vs_comp"],
            y=comp_df["predicted_quantity"],
            mode="lines",
            line=dict(color="#1f77b4", width=2.5),
            name="Predicted Demand",
        ))

        # Current gap
        current_gap = float(base_row.get("price_vs_comp", 0))
        if not np.isnan(current_gap):
            fig_comp.add_vline(
                x=current_gap,
                line_dash="dash",
                line_color="#ff7f0e",
                annotation_text=f"Current gap ${current_gap:+.2f}",
                annotation_position="top",
                annotation_font_color="#ff7f0e",
            )

        fig_comp.add_vline(
            x=0,
            line_color="gray",
            line_width=1,
            annotation_text="Price parity",
            annotation_position="bottom",
            annotation_font_color="gray",
        )

        # Shade premium vs discount regions
        fig_comp.add_vrect(
            x0=comp_df["price_vs_comp"].min(), x1=0,
            fillcolor="rgba(44,160,44,0.07)", line_width=0,
            annotation_text="We're cheaper", annotation_position="top left",
            annotation_font_color="#2ca02c",
        )
        fig_comp.add_vrect(
            x0=0, x1=comp_df["price_vs_comp"].max(),
            fillcolor="rgba(214,39,40,0.07)", line_width=0,
            annotation_text="We're more expensive", annotation_position="top right",
            annotation_font_color="#d62728",
        )

        fig_comp.update_layout(
            xaxis_title="Price vs. Competitor (our price − comp price, $)",
            yaxis_title="Predicted Quantity (units)",
            height=420,
            margin=dict(t=30, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_c2:
        st.subheader("Competitive Position Summary")

        gap_at_parity_idx = comp_df["price_vs_comp"].abs().idxmin()
        demand_at_parity = comp_df.loc[gap_at_parity_idx, "predicted_quantity"]

        if not np.isnan(current_gap):
            demand_at_current = comp_df.loc[
                (comp_df["price_vs_comp"] - current_gap).abs().idxmin(),
                "predicted_quantity",
            ]
            delta_to_parity = demand_at_parity - demand_at_current
            st.metric(
                "Demand at price parity",
                f"{demand_at_parity:.1f} units",
                delta=f"{delta_to_parity:+.1f} vs current",
            )
            st.metric("Current price gap", f"${current_gap:+.2f}")
            st.metric("Current comp price", f"${base_comp:.2f}")

        demand_at_cheapest = comp_df.iloc[0]["predicted_quantity"]
        demand_at_most_exp = comp_df.iloc[-1]["predicted_quantity"]
        st.metric(
            "Demand range (cheapest → priciest)",
            f"{demand_at_cheapest:.1f} → {demand_at_most_exp:.1f} units",
        )

        st.markdown("""
**Reading the chart:**
- Negative gap = your price < competitor (price advantage)
- Positive gap = your price > competitor (premium positioning)
- Steeper slope = customers are more price-sensitive to relative positioning
        """)

# ============================================================
# TAB 3: OEM vs Non-OEM Comparison
# ============================================================
with tab3:
    st.subheader("OEM vs. Non-OEM Demand Comparison")
    st.caption("All other features held at segment median. Isolates the demand effect of OEM status.")

    col_o1, col_o2 = st.columns([2, 1])

    with col_o1:
        fig_oem = go.Figure()

        fig_oem.add_trace(go.Scatter(
            x=oem_df["unit_retail"],
            y=oem_df["predicted_quantity"],
            mode="lines",
            name="OEM",
            line=dict(color="#1f77b4", width=2.5),
        ))

        fig_oem.add_trace(go.Scatter(
            x=non_oem_df["unit_retail"],
            y=non_oem_df["predicted_quantity"],
            mode="lines",
            name="Non-OEM",
            line=dict(color="#d62728", width=2.5, dash="dash"),
        ))

        fig_oem.add_vline(
            x=base_price,
            line_dash="dot",
            line_color="#ff7f0e",
            annotation_text=f"Current ${base_price:.2f}",
            annotation_position="top",
            annotation_font_color="#ff7f0e",
        )

        fig_oem.update_layout(
            xaxis_title="Unit Retail Price ($)",
            yaxis_title="Predicted Quantity (units)",
            legend=dict(x=0.75, y=0.99),
            height=420,
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_oem, use_container_width=True)

    with col_o2:
        st.subheader("Elasticity Comparison")

        fig_e_oem = go.Figure()
        fig_e_oem.add_trace(go.Scatter(
            x=oem_df["unit_retail"],
            y=oem_df["elasticity"],
            mode="lines",
            name="OEM",
            line=dict(color="#1f77b4", width=2),
        ))
        fig_e_oem.add_trace(go.Scatter(
            x=non_oem_df["unit_retail"],
            y=non_oem_df["elasticity"],
            mode="lines",
            name="Non-OEM",
            line=dict(color="#d62728", width=2, dash="dash"),
        ))
        fig_e_oem.add_hline(
            y=-1, line_dash="dash", line_color="gray",
            annotation_text="ε = -1", annotation_position="bottom right",
        )
        fig_e_oem.update_layout(
            xaxis_title="Unit Retail Price ($)",
            yaxis_title="Elasticity",
            legend=dict(x=0.6, y=0.99),
            height=420,
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_e_oem, use_container_width=True)

    # Summary table
    oem_e_at_base = oem_df.loc[(oem_df["unit_retail"] - base_price).abs().idxmin(), "elasticity"]
    non_oem_e_at_base = non_oem_df.loc[(non_oem_df["unit_retail"] - base_price).abs().idxmin(), "elasticity"]
    oem_q_at_base = oem_df.loc[(oem_df["unit_retail"] - base_price).abs().idxmin(), "predicted_quantity"]
    non_oem_q_at_base = non_oem_df.loc[(non_oem_df["unit_retail"] - base_price).abs().idxmin(), "predicted_quantity"]

    summary = pd.DataFrame({
        "Metric": ["Predicted Demand at Base Price", "Point Elasticity at Base Price", "Elasticity Zone"],
        "OEM": [f"{oem_q_at_base:.1f} units", f"{oem_e_at_base:.3f}",
                "Elastic" if oem_e_at_base < -1 else "Inelastic"],
        "Non-OEM": [f"{non_oem_q_at_base:.1f} units", f"{non_oem_e_at_base:.3f}",
                    "Elastic" if non_oem_e_at_base < -1 else "Inelastic"],
    })
    st.table(summary.set_index("Metric"))

# ============================================================
# TAB 4: Model Performance
# ============================================================
with tab4:
    st.subheader("Model Evaluation — XGBoost Gradient Boosting")
    st.caption("Trained on PROJ_PELAST_TRAIN_v2 only. TEST set is fully held out.")

    # Metrics table
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown("**Train set**")
        st.metric("R²", f"{train_metrics['r2']:.4f}")
        st.metric("MAE", f"{train_metrics['mae']:.3f} units")
        st.metric("RMSE", f"{train_metrics['rmse']:.3f} units")
    with col_m2:
        st.markdown("**Test set**")
        st.metric("R²", f"{test_metrics['r2']:.4f}")
        st.metric("MAE", f"{test_metrics['mae']:.3f} units")
        st.metric("RMSE", f"{test_metrics['rmse']:.3f} units")
    with col_m3:
        st.markdown("**Data split**")
        st.metric("Train rows", f"{len(train_df):,}")
        st.metric("Test rows", f"{len(test_df):,}")
        st.metric("Features", len(feature_cols))

    st.divider()

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.subheader("Predicted vs. Actual Quantity (Test)")
        fig_pva = go.Figure()
        fig_pva.add_trace(go.Scatter(
            x=test_metrics["y_true"],
            y=test_metrics["y_pred"],
            mode="markers",
            marker=dict(color="#1f77b4", size=6, opacity=0.6),
            name="Observations",
        ))
        max_val = max(test_metrics["y_true"].max(), test_metrics["y_pred"].max())
        fig_pva.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="Perfect fit",
        ))
        fig_pva.update_layout(
            xaxis_title="Actual Quantity",
            yaxis_title="Predicted Quantity",
            height=380,
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_pva, use_container_width=True)

    with col_p2:
        st.subheader("Top 20 Feature Importances")
        importances = model.feature_importances_
        feat_imp = (
            pd.Series(importances, index=feature_cols)
            .sort_values(ascending=False)
            .head(20)
        )

        fig_fi = go.Figure(go.Bar(
            x=feat_imp.values,
            y=feat_imp.index,
            orientation="h",
            marker_color="#1f77b4",
        ))
        fig_fi.update_layout(
            xaxis_title="Importance (gain)",
            yaxis=dict(autorange="reversed"),
            height=380,
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # Residuals
    st.subheader("Residual Distribution (Test)")
    residuals = test_metrics["y_pred"] - test_metrics["y_true"]
    fig_res = px.histogram(
        x=residuals,
        nbins=40,
        color_discrete_sequence=["#1f77b4"],
        labels={"x": "Residual (predicted − actual)"},
    )
    fig_res.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_res.update_layout(height=280, margin=dict(t=20, b=40), showlegend=False)
    st.plotly_chart(fig_res, use_container_width=True)

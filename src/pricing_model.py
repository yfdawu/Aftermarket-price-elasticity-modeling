"""
src/pricing_model.py
====================
XGBoost (gradient boosting) pricing model.
 
Flow
----
1. load_data()           — load and clean the CSV
2. train_model()         — train XGBoost on all rows, returns (model, feature_columns)
3. precompute_sweeps()   — for every SKU x Branch combo, sweep prices and store
                           predicted units / revenue / margin in a lookup dict
4. lookup_scenario()     — given a price, interpolate from the precomputed sweep
                           (called on every slider move — essentially instant)
 
The app imports and calls these four functions only.
Everything else here is internal.
"""
 
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
 
#CONSTANTS
 
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
 
# Mid-year average: average month number used for all predictions
# June (6) sits at the midpoint of the year.
MID_YEAR_MONTH = "Jun"

# Price sweep range: ±30% around each SKU's historical average price.
# Wide enough to find the true optimum, narrow enough to stay realistic.
SWEEP_PCT_LOW  = 0.70
SWEEP_PCT_HIGH = 1.30
SWEEP_STEPS    = 60   # 60 points = smooth curve, fast to compute
 
 
#1. LOAD DATA 
 
def load_data(file_path: str = "data/synthetic_pricing_data.csv") -> pd.DataFrame:
    """Load CSV and sort months correctly."""
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    df["Month"] = pd.Categorical(df["Month"], categories=MONTH_ORDER, ordered=True)
    return df
 
 
# 2. TRAIN MODEL
 
def _prepare_features(df: pd.DataFrame):
    """
    Build feature matrix and target vector.
 
    Features used:
      - Price          (continuous — the main lever)
      - Category       (one-hot)
      - Branch         (one-hot)
      - Month          (one-hot — captures any residual seasonal signal
                        in the training data without us hard-coding it)
 
    Target: Units (actual demand from the data)
    """
    X = pd.get_dummies(
        df[["Price", "Category", "Branch", "Month"]],
        drop_first=False
    )
    y = df["Units"]
    return X, y
 
 
def train_model(df: pd.DataFrame):
    """
    Train XGBoost on the full dataset.
 
    Returns
    -------
    model : trained XGBRegressor
    feature_columns : list[str]
        Column order used during training — must be reproduced exactly
        when building prediction rows later.
    """
    X, y = _prepare_features(df)
 
    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42
    )
    model.fit(X, y)
 
    return model, X.columns.tolist()
 
 
# INTERNAL: single prediction
 
def _build_row(price: float, category: str, branch: str,
               month: str, feature_columns: list) -> pd.DataFrame:
    """Build a single one-row feature DataFrame matching training columns."""
    row = pd.DataFrame({
        "Price":    [price],
        "Category": [category],
        "Branch":   [branch],
        "Month":    [month]
    })
    row_enc = pd.get_dummies(row, drop_first=False)
 
    for col in feature_columns:
        if col not in row_enc.columns:
            row_enc[col] = 0
 
    return row_enc[feature_columns]
 
 
def _predict_units(model, price: float, category: str, branch: str,
                   month: str, feature_columns: list) -> float:
    """Predict units at a single price point. Returns 0 if negative."""
    row = _build_row(price, category, branch, month, feature_columns)
    return max(0.0, float(model.predict(row)[0]))
 
 
# 3. PRECOMPUTE SWEEPS
 
def precompute_sweeps(df: pd.DataFrame, model, feature_columns: list) -> dict:
    """
    For every SKU x Branch combination, sweep a range of prices and store
    the full demand / revenue / margin curve.
 
    This runs ONCE on app load (takes ~1-3 seconds).
    After that all slider interactions are instant lookups.
 
    Returns
    -------
    sweeps : dict
        Key: (sku, branch)
        Value: {
            "prices":    np.ndarray,
            "units":     np.ndarray,
            "revenue":   np.ndarray,
            "margin":    np.ndarray,
            "unit_cost": float,
            "category":  str,
            "rev_opt_price":    float,
            "margin_opt_price": float,
        }
    """
    sweeps = {}
 
    for sku in sorted(df["SKU"].dropna().unique()):
        sku_df = df[df["SKU"] == sku]
 
        for branch in sorted(sku_df["Branch"].dropna().unique()):
            branch_df = sku_df[sku_df["Branch"] == branch]
 
            if branch_df.empty:
                continue
 
            category  = branch_df["Category"].iloc[0]
            unit_cost = float(branch_df["Unit_Cost"].iloc[0])
            avg_price = float(branch_df["Price"].mean())
 
            price_min = avg_price * SWEEP_PCT_LOW
            price_max = avg_price * SWEEP_PCT_HIGH
            prices    = np.linspace(price_min, price_max, SWEEP_STEPS)
 
            units = np.array([
                _predict_units(model, p, category, branch,
                               MID_YEAR_MONTH, feature_columns)
                for p in prices
            ])
 
            revenue = prices * units
            margin  = np.where(prices > unit_cost,
                               (prices - unit_cost) * units,
                               0.0)
 
            rev_opt_price    = float(prices[np.argmax(revenue)])
            margin_opt_price = float(prices[np.argmax(margin)])
 
            sweeps[(sku, branch)] = {
                "prices":            prices,
                "units":             units,
                "revenue":           revenue,
                "margin":            margin,
                "unit_cost":         unit_cost,
                "category":          category,
                "rev_opt_price":     rev_opt_price,
                "margin_opt_price":  margin_opt_price,
            }
 
    return sweeps
 
 
# 4. LOOKUP SCENARIO
 
def lookup_scenario(sweeps: dict, sku: str, branch: str,
                    price: float) -> dict:
    """
    Interpolate demand, revenue, and margin at any price from the
    precomputed sweep for a given SKU + Branch.
 
    This is what runs on every slider move — no model inference,
    just a numpy interpolation (~microseconds).
 
    Returns
    -------
    dict with keys:
        predicted_units, revenue, margin,
        unit_cost, rev_opt_price, margin_opt_price,
        prices, units, revenues, margins   ← full curves for charts
    """
    key = (sku, branch)
 
    if key not in sweeps:
        return _empty_scenario(price)
 
    s = sweeps[key]
 
    predicted_units = float(np.interp(price, s["prices"], s["units"]))
    predicted_units = max(predicted_units, 0.0)
    revenue         = price * predicted_units
    margin          = (price - s["unit_cost"]) * predicted_units if price > s["unit_cost"] else 0.0
 
    return {
        # — point values at the queried price —
        "predicted_units":   predicted_units,
        "revenue":           revenue,
        "margin":            margin,
        "unit_cost":         s["unit_cost"],
        # — optimal prices —
        "rev_opt_price":     s["rev_opt_price"],
        "margin_opt_price":  s["margin_opt_price"],
        # — full curves (for charts) —
        "prices":            s["prices"],
        "units":             s["units"],
        "revenues":          s["revenue"],
        "margins":           s["margin"],
    }
 
 
def _empty_scenario(price: float) -> dict:
    """Fallback returned if a SKU/Branch key is missing from sweeps."""
    return {
        "predicted_units":   0.0,
        "revenue":           0.0,
        "margin":            0.0,
        "unit_cost":         0.0,
        "rev_opt_price":     price,
        "margin_opt_price":  price,
        "prices":            np.array([price]),
        "units":             np.array([0.0]),
        "revenues":          np.array([0.0]),
        "margins":           np.array([0.0]),
    }
 
 
# HELPERS USED BY THE APP
 
def get_product_snapshot(df: pd.DataFrame, sku: str, branch: str) -> dict | None:
    """
    Return current price, cost, and category for a SKU + Branch.
    Uses the last month available (most recent observation).
    Month parameter removed — app no longer needs to pass it.
    """
    rows = df[(df["SKU"] == sku) & (df["Branch"] == branch)]
 
    if rows.empty:
        return None
 
    # Most recent month
    row = rows.sort_values("Month").iloc[-1]
 
    return {
        "sku":       row["SKU"],
        "branch":    row["Branch"],
        "price":     float(row["Price"]),
        "category":  row["Category"],
        "unit_cost": float(row["Unit_Cost"]),
    }
 
 
def sensitivity_label(demand_change_pct: float) -> str:
    """Classify price sensitivity from a percentage demand change."""
    if demand_change_pct <= -10:
        return "Price Sensitive"
    elif demand_change_pct <= -3:
        return "Moderately Sensitive"
    else:
        return "Price Resilient"
 
 
def get_portfolio_summary(df: pd.DataFrame, sweeps: dict,
                           branch_filter: str = "All") -> pd.DataFrame:
    """
    Build a portfolio-level summary table across all SKUs.
    Used by the Portfolio tab in the app.
 
    Parameters
    ----------
    branch_filter : "All" or a specific branch name
    """
    rows = []
 
    for sku in sorted(df["SKU"].dropna().unique()):
        sku_df = df[df["SKU"] == sku]
 
        branches = (
            sku_df["Branch"].unique()
            if branch_filter == "All"
            else [branch_filter]
        )
 
        for branch in sorted(branches):
            key = (sku, branch)
            if key not in sweeps:
                continue
 
            s         = sweeps[key]
            snap      = get_product_snapshot(df, sku, branch)
            if snap is None:
                continue
 
            current_price = snap["price"]
            unit_cost     = s["unit_cost"]
 
            # Current performance
            current = lookup_scenario(sweeps, sku, branch, current_price)
 
            # Optimal performance
            optimal = lookup_scenario(sweeps, sku, branch, s["rev_opt_price"])
 
            revenue_upside_pct = (
                (optimal["revenue"] - current["revenue"]) / current["revenue"] * 100
                if current["revenue"] > 0 else 0.0
            )
            price_gap_pct = (
                (s["rev_opt_price"] - current_price) / current_price * 100
                if current_price > 0 else 0.0
            )
 
            rows.append({
                "SKU":                   sku,
                "Branch":                branch,
                "Current Price":         round(current_price, 2),
                "Revenue-Optimal Price": round(s["rev_opt_price"], 2),
                "Margin-Optimal Price":  round(s["margin_opt_price"], 2),
                "Revenue Gap (%)":       round(price_gap_pct, 2),
                "Revenue Upside (%)":    round(revenue_upside_pct, 2),
                "Current Revenue":       round(current["revenue"], 2),
                "Optimal Revenue":       round(optimal["revenue"], 2),
                "Unit Cost":             round(unit_cost, 2),
                "Observations":          len(df[(df["SKU"] == sku) & (df["Branch"] == branch)])
            })
 
    return pd.DataFrame(rows)
 

"""
Pricing Elasticity Model
XGBoost demand model trained on PROJ_PELAST_TRAIN_v2, evaluated on PROJ_PELAST_TEST.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MACRO_FEATURES = [
    "FRGSHPUSM649NCIS", "TRUCKD11", "VMTD11", "HTRUCKSSAAR", "WPU141106"
]

NUMERIC_FEATURES = (
    ["unit_retail", "price_vs_comp", "discount_rate", "annual_spend"]
    + MACRO_FEATURES
    + ["month", "year"]
)

OHE_COLS = ["PartID", "CustID", "oem_flag"]


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def load_and_preprocess(path, is_train: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalise date column name (train uses '(No column name)', test uses 'invoice_date')
    if "(No column name)" in df.columns:
        df = df.rename(columns={"(No column name)": "invoice_date"})

    df["invoice_date"] = pd.to_datetime(df["invoice_date"], format="mixed", dayfirst=False)
    df["month"] = df["invoice_date"].dt.month
    df["year"] = df["invoice_date"].dt.year

    # Drop returns and null transactions
    df = df[(df["quantity"] > 0) & (df["retail_sales"] > 0)].copy()

    # Derived price features
    df["unit_retail"] = df["retail_sales"] / df["quantity"]
    df["unit_cogs"] = df["cogs"] / df["quantity"]
    df["price_vs_comp"] = np.where(
        df["comp_price"] == 0, np.nan, df["unit_retail"] - df["comp_price"]
    )

    # Discount rate
    if is_train:
        # actual_sales column present in train
        df["discount_rate"] = 1.0 - (df["actual_sales"] / df["retail_sales"])
    else:
        # Reconstruct actual_sales from posgp + cogs (posgp = actual_sales - cogs)
        df["discount_rate"] = 1.0 - ((df["posgp"] + df["cogs"]) / df["retail_sales"])

    # Drop columns not used as features
    drop_cols = ["natl_flag", "posgp", "invoice_date", "retail_sales", "cogs", "unit_cogs"]
    if is_train:
        drop_cols += ["actual_sales"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    cat_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode PartID, CustID, oem_flag and concatenate with numeric features.
    If cat_columns is provided (from training), align to that exact column set.
    """
    dummies = pd.get_dummies(df[OHE_COLS], columns=OHE_COLS, dtype=int)

    num_df = df[NUMERIC_FEATURES].reset_index(drop=True)
    dummies = dummies.reset_index(drop=True)
    X = pd.concat([num_df, dummies], axis=1)

    if cat_columns is not None:
        for col in cat_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[cat_columns]

    return X, list(X.columns)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(train_df: pd.DataFrame) -> Tuple[XGBRegressor, List[str]]:
    X, feature_cols = build_features(train_df)
    y = train_df["quantity"].values

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X, y, verbose=False)

    return model, feature_cols


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: XGBRegressor,
    feature_cols: List[str],
    test_df: pd.DataFrame,
) -> dict:
    X, _ = build_features(test_df, cat_columns=feature_cols)
    y_true = test_df["quantity"].values
    y_pred = np.maximum(model.predict(X), 0.0)

    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def predict_demand(
    model: XGBRegressor,
    feature_cols: List[str],
    row: pd.Series,
) -> float:
    X, _ = build_features(pd.DataFrame([row]), cat_columns=feature_cols)
    return float(max(model.predict(X)[0], 0.0))


def price_sweep(
    model: XGBRegressor,
    feature_cols: List[str],
    base_row: pd.Series,
    price_range: Tuple[float, float] = (0.5, 2.0),
    n_points: int = 120,
) -> pd.DataFrame:
    """
    Hold all features constant except unit_retail (and price_vs_comp if comp_price known).
    Returns a DataFrame with unit_retail, predicted_quantity, elasticity, elasticity_zone, revenue.
    """
    base_price = float(base_row["unit_retail"])
    comp_price = float(base_row.get("comp_price", 0))
    prices = np.linspace(base_price * price_range[0], base_price * price_range[1], n_points)

    records = []
    for p in prices:
        r = base_row.copy()
        r["unit_retail"] = p
        if comp_price != 0:
            r["price_vs_comp"] = p - comp_price
        X, _ = build_features(pd.DataFrame([r]), cat_columns=feature_cols)
        q = float(max(model.predict(X)[0], 0.0))
        records.append({"unit_retail": p, "predicted_quantity": q})

    result = pd.DataFrame(records)

    # Point elasticity via finite differences: ε = (dQ/dP) * (P/Q)
    dQ_dP = np.gradient(result["predicted_quantity"].values, result["unit_retail"].values)
    Q = result["predicted_quantity"].values
    P = result["unit_retail"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(Q > 0.001, dQ_dP * P / Q, 0.0)
    result["elasticity"] = e
    result["elasticity_zone"] = result["elasticity"].apply(
        lambda v: "Elastic" if v < -1 else ("Inelastic" if v > -1 else "Unitary Elastic")
    )
    result["revenue"] = result["unit_retail"] * result["predicted_quantity"]

    return result


def comp_price_sweep(
    model: XGBRegressor,
    feature_cols: List[str],
    base_row: pd.Series,
    comp_range: Tuple[float, float] = (0.5, 2.0),
    n_points: int = 100,
) -> pd.DataFrame:
    """
    Hold unit_retail constant. Vary comp_price to sweep price_vs_comp.
    Returns DataFrame with comp_price, price_vs_comp, predicted_quantity.
    """
    base_unit = float(base_row["unit_retail"])
    base_comp = float(base_row.get("comp_price", base_unit))
    if base_comp == 0:
        base_comp = base_unit  # fallback

    comp_prices = np.linspace(base_comp * comp_range[0], base_comp * comp_range[1], n_points)

    records = []
    for cp in comp_prices:
        r = base_row.copy()
        r["comp_price"] = cp
        r["price_vs_comp"] = base_unit - cp
        X, _ = build_features(pd.DataFrame([r]), cat_columns=feature_cols)
        q = float(max(model.predict(X)[0], 0.0))
        records.append({
            "comp_price": cp,
            "price_vs_comp": base_unit - cp,
            "predicted_quantity": q,
        })

    return pd.DataFrame(records)

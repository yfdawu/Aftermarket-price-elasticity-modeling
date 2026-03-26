import pandas as pd
import numpy as np
from xgboost import XGBRegressor


MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def load_data(file_path="data/synthetic_pricing_data.csv"):
    df = pd.read_csv(file_path)
    df["Month"] = pd.Categorical(df["Month"], categories=MONTH_ORDER, ordered=True)
    return df


def prepare_features(df):
    model_df = df.copy()
    X = pd.get_dummies(model_df[["Price", "Category", "Branch", "Month"]], drop_first=False)
    y = model_df["Units"]
    return X, y


def train_model(df):
    X, y = prepare_features(df)

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

    feature_columns = X.columns.tolist()
    return model, feature_columns


def build_prediction_row(price, category, branch, month, feature_columns):
    row = pd.DataFrame({
        "Price": [price],
        "Category": [category],
        "Branch": [branch],
        "Month": [month]
    })

    row_encoded = pd.get_dummies(row, drop_first=False)

    for col in feature_columns:
        if col not in row_encoded.columns:
            row_encoded[col] = 0

    row_encoded = row_encoded[feature_columns]
    return row_encoded


def predict_units(model, price, category, branch, month, feature_columns):
    pred_row = build_prediction_row(price, category, branch, month, feature_columns)
    pred_units = model.predict(pred_row)[0]
    return max(0, float(pred_units))


def optimize_price(model, category, branch, month, unit_cost, feature_columns,
                   price_min=50, price_max=400, step=1):
    prices = np.arange(price_min, price_max + step, step)

    results = []
    for p in prices:
        units = predict_units(model, p, category, branch, month, feature_columns)
        revenue = p * units
        profit = (p - unit_cost) * units
        results.append((p, units, revenue, profit))

    results_df = pd.DataFrame(results, columns=["Price", "Predicted_Units", "Revenue", "Profit"])

    revenue_best = results_df.loc[results_df["Revenue"].idxmax()]
    profit_best = results_df.loc[results_df["Profit"].idxmax()]

    return {
        "revenue_best_price": float(revenue_best["Price"]),
        "profit_best_price": float(profit_best["Price"]),
        "results_table": results_df
    }


def scenario_test(model, current_price, scenario_pct, category, branch, month, feature_columns):
    new_price = current_price * (1 + scenario_pct / 100)

    current_units = predict_units(model, current_price, category, branch, month, feature_columns)
    new_units = predict_units(model, new_price, category, branch, month, feature_columns)

    current_revenue = current_price * current_units
    new_revenue = new_price * new_units

    demand_change_pct = ((new_units - current_units) / current_units * 100) if current_units > 0 else 0
    revenue_change_pct = ((new_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0

    return {
        "scenario_price": float(new_price),
        "current_units": float(current_units),
        "scenario_units": float(new_units),
        "demand_change_pct": float(demand_change_pct),
        "revenue_change_pct": float(revenue_change_pct)
    }


def sensitivity_label(demand_change_pct):
    if demand_change_pct <= -10:
        return "Price Sensitive"
    elif demand_change_pct <= -3:
        return "Moderately Sensitive"
    else:
        return "Price Resilient"


def get_product_snapshot(df, sku, branch, month):
    row = df[(df["SKU"] == sku) & (df["Branch"] == branch) & (df["Month"] == month)]

    if row.empty:
        return None

    row = row.iloc[0]
    return {
        "sku": row["SKU"],
        "branch": row["Branch"],
        "month": row["Month"],
        "price": float(row["Price"]),
        "category": row["Category"],
        "unit_cost": float(row["Unit_Cost"])
    }

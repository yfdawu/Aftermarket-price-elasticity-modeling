from src.pricing_model import (
    load_data,
    train_model,
    get_product_snapshot,
    optimize_price,
    scenario_test,
    sensitivity_label
)

# Load data
df = load_data("data/synthetic_pricing_data.csv")

# Train model
model, feature_columns = train_model(df)

# Pick one example product/branch/month
sku = df["SKU"].iloc[0]
branch = df["Branch"].iloc[0]
month = df["Month"].iloc[0]

snapshot = get_product_snapshot(df, sku, branch, month)

if snapshot is None:
    print("No matching record found.")
else:
    current_price = snapshot["price"]
    category = snapshot["category"]
    unit_cost = snapshot["unit_cost"]

    # Optimize price
    opt = optimize_price(
        model=model,
        category=category,
        branch=branch,
        month=month,
        unit_cost=unit_cost,
        feature_columns=feature_columns,
        price_min=max(10, current_price * 0.7),
        price_max=current_price * 1.3,
        step=1
    )

    # Scenario test
    scenario = scenario_test(
        model=model,
        current_price=current_price,
        scenario_pct=5,
        category=category,
        branch=branch,
        month=month,
        feature_columns=feature_columns
    )

    label = sensitivity_label(scenario["demand_change_pct"])

    print("=== PRODUCT SNAPSHOT ===")
    print(snapshot)

    print("\n=== OPTIMIZATION ===")
    print(f"Current Price: {current_price:.2f}")
    print(f"Revenue-Optimized Price: {opt['revenue_best_price']:.2f}")
    print(f"Profit-Optimized Price: {opt['profit_best_price']:.2f}")

    print("\n=== SCENARIO TEST (+5%) ===")
    print(f"Scenario Price: {scenario['scenario_price']:.2f}")
    print(f"Demand Change %: {scenario['demand_change_pct']:.2f}")
    print(f"Revenue Change %: {scenario['revenue_change_pct']:.2f}")
    print(f"Pricing Response Signal: {label}")

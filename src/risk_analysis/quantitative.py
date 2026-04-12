# src/risk_analysis/quantitative.py
# Payoff table built entirely from real ML dollar losses

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# SCENARIO MULTIPLIERS
# High Risk: fraud is 2x more frequent and larger
# Moderate Risk: normal year, baseline
# Low Risk: quiet year, half the baseline
# ─────────────────────────────────────────────
SCENARIO_MULTIPLIERS = {
    "High_Risk":     2.0,
    "Moderate_Risk": 1.0,
    "Low_Risk":      0.5,
}

# Scenario probabilities
PROBABILITIES = {
    "High_Risk":     0.30,
    "Moderate_Risk": 0.50,
    "Low_Risk":      0.20,
}

# Annual deployment costs
DEPLOYMENT_COSTS = {
    "Advanced_ML": 800_000,
    "Standard_ML": 250_000,
    "Rule_Based":        0,
}

# Model name mapping
MODEL_MAP = {
    "XGBoost":            "Advanced_ML",
    "LogisticRegression": "Standard_ML",
    "RuleBased":          "Rule_Based",
}


def load_annual_losses():
    """
    Load real annual scaled losses from model_performance.csv.
    Returns dict: {alternative: annual_scaled_loss}
    """
    path = "results/tables/model_performance.csv"
    df   = pd.read_csv(path).set_index("Model")
    losses = {}
    for model, alt in MODEL_MAP.items():
        losses[alt] = df.loc[model, "annual_scaled_loss"]
    return losses


def build_sle_ale_table():
    """
    Builds full SLE/ALE table.
    SLE = annual_scaled_loss * scenario_multiplier (one occurrence)
    ALE = SLE * ARO (probability-weighted annual loss)
    Total Cost = ALE + deployment cost
    """
    losses = load_annual_losses()
    rows   = []

    for alt, base_loss in losses.items():
        for scenario, multiplier in SCENARIO_MULTIPLIERS.items():
            sle        = base_loss * multiplier
            aro        = PROBABILITIES[scenario]
            ale        = sle * aro
            deploy     = DEPLOYMENT_COSTS[alt]
            total_cost = ale + deploy

            rows.append({
                "Alternative":        alt,
                "Scenario":           scenario,
                "Base Annual Loss($)": round(base_loss, 2),
                "Scenario Multiplier": multiplier,
                "SLE ($)":            round(sle, 2),
                "ARO":                aro,
                "ALE ($)":            round(ale, 2),
                "Deploy Cost ($)":    deploy,
                "Total Cost ($)":     round(total_cost, 2),
            })

    return pd.DataFrame(rows)


def build_payoff_table():
    """
    Payoff table: rows = alternatives, columns = scenarios.
    Values = Total Annual Cost.
    """
    table  = build_sle_ale_table()
    payoff = table.pivot_table(
        index="Alternative",
        columns="Scenario",
        values="Total Cost ($)"
    )
    return payoff[["High_Risk", "Moderate_Risk", "Low_Risk"]]


if __name__ == "__main__":
    print("=" * 60)
    print("QUANTITATIVE RISK ANALYSIS")
    print("=" * 60)

    losses = load_annual_losses()
    print("\n--- BASE ANNUAL LOSSES (from real model performance) ---")
    for alt, loss in losses.items():
        print(f"  {alt:<15} ${loss:>15,.2f}")

    print("\n--- SLE / ALE TABLE ---")
    table = build_sle_ale_table()
    print(table.to_string(index=False))

    print("\n--- PAYOFF TABLE (Total Annual Cost $) ---")
    payoff = build_payoff_table()
    print(payoff.to_string())

    os.makedirs("results/tables", exist_ok=True)
    table.to_csv("results/tables/sle_ale_analysis.csv", index=False)
    payoff.to_csv("results/tables/payoff_table.csv")
    print("\n  Saved to results/tables/")

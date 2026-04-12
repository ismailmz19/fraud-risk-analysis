# src/risk_analysis/utility.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.risk_analysis.quantitative import build_payoff_table, PROBABILITIES

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

RISK_TOLERANCE = 2_000_000  # R = $2M

COLORS = {
    "Advanced_ML": "#2196F3",
    "Standard_ML": "#FF9800",
    "Rule_Based":  "#F44336",
}

def utility(cost, R=RISK_TOLERANCE):
    """
    Exponential utility for costs (risk-averse decision maker).
    U(cost) = -e^(cost/R)
    Lower cost = higher (less negative) utility.
    """
    return -np.exp(cost / R)

def inverse_utility(u, R=RISK_TOLERANCE):
    """
    Inverse: given utility value, return the equivalent cost.
    cost = R * ln(-u)
    """
    return R * np.log(-u)

def compute_utility_analysis(payoff, probabilities):
    probs = pd.Series(probabilities)
    rows  = []

    for alt in payoff.index:
        costs = payoff.loc[alt]

        # Utility of each outcome
        utils = costs.apply(lambda c: utility(c))

        # Expected Utility
        eu = (utils * probs).sum()

        # Certainty Equivalent: cost with same utility as EU
        ce = inverse_utility(eu)

        # EMV
        emv = (costs * probs).sum()

        # Risk Premium = CE - EMV
        rp = ce - emv

        rows.append({
            "Alternative":        alt,
            "EMV ($)":            round(emv, 2),
            "Expected Utility":   round(eu, 6),
            "Certainty Equiv ($)": round(ce, 2),
            "Risk Premium ($)":   round(rp, 2),
        })

    return pd.DataFrame(rows).set_index("Alternative")

def plot_utility_curves(payoff, probabilities):
    df = compute_utility_analysis(payoff, probabilities)

    # Cost range for curve
    max_cost = payoff.values.max()
    costs    = np.linspace(0, max_cost * 1.1, 500)
    utils    = utility(costs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: utility function curve with CE marked
    axes[0].plot(costs/1e6, utils, color="black", linewidth=2, label="Utility Function")
    for alt, row in df.iterrows():
        ce_cost = row["Certainty Equiv ($)"]
        ce_util = utility(ce_cost)
        axes[0].axvline(ce_cost/1e6, color=COLORS[alt], linestyle="--", alpha=0.7)
        axes[0].scatter([ce_cost/1e6], [ce_util], color=COLORS[alt], s=80, zorder=5,
                        label=f"{alt} CE=${ce_cost/1e6:.2f}M")
    axes[0].set_xlabel("Cost ($ Millions)")
    axes[0].set_ylabel("Utility")
    axes[0].set_title("Exponential Utility Function\nwith Certainty Equivalents", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(linestyle="--", alpha=0.4)

    # Right: EMV vs CE comparison
    x     = np.arange(len(df))
    width = 0.35
    axes[1].bar(x - width/2, df["EMV ($)"]/1e6, width, label="EMV",
                color=[COLORS[a] for a in df.index], alpha=0.6, edgecolor="black")
    axes[1].bar(x + width/2, df["Certainty Equiv ($)"]/1e6, width, label="Certainty Equivalent",
                color=[COLORS[a] for a in df.index], alpha=1.0, edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df.index)
    axes[1].set_ylabel("Cost ($ Millions)")
    axes[1].set_title("EMV vs Certainty Equivalent\n(Risk-Averse Decision Maker)", fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle(f"Risk Utility Analysis (R = ${RISK_TOLERANCE/1e6:.1f}M)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/utility_curves.png", dpi=150)
    plt.close()
    print("  Saved: utility_curves.png")

if __name__ == "__main__":
    print("=" * 60)
    print("RISK UTILITY ANALYSIS")
    print("=" * 60)
    print(f"  Risk Tolerance R = ${RISK_TOLERANCE:,.0f}")
    print(f"  Utility Function: U(cost) = -exp(cost / R)")

    payoff = build_payoff_table()
    df     = compute_utility_analysis(payoff, PROBABILITIES)

    print("\n--- UTILITY ANALYSIS RESULTS ---")
    print(df.to_string())

    best_eu  = df["Expected Utility"].idxmax()
    best_ce  = df["Certainty Equiv ($)"].idxmin()
    print(f"\n  Best strategy by Expected Utility: {best_eu}")
    print(f"  Best strategy by Certainty Equiv:  {best_ce}")

    print("\n--- RISK PREMIUMS ---")
    for alt, row in df.iterrows():
        print(f"  {alt:<15} Risk Premium = ${row['Risk Premium ($)']:>12,.2f}")

    plot_utility_curves(payoff, PROBABILITIES)
    df.to_csv("results/tables/utility_analysis.csv")
    print("  Saved: utility_analysis.csv")

# src/visualization/risk_profile.py
# Risk Profile — probability distribution of outcomes per strategy

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.risk_analysis.quantitative import build_payoff_table, PROBABILITIES

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

COLORS = {
    "Advanced_ML": "#1A6B9A",
    "Standard_ML": "#F5A623",
    "Rule_Based":  "#D94F3D",
}

def build_risk_profiles(payoff, probabilities):
    """
    For each strategy, list (cost, probability) pairs — one per scenario.
    Also compute EMV and variance.
    """
    profiles = {}
    scenarios = list(probabilities.keys())
    probs     = list(probabilities.values())

    for alt in payoff.index:
        costs = [payoff.loc[alt, s] for s in scenarios]
        emv   = sum(c * p for c, p in zip(costs, probs))
        var   = sum(p * (c - emv)**2 for c, p in zip(costs, probs))
        std   = var ** 0.5
        profiles[alt] = {
            "costs":     costs,
            "probs":     probs,
            "scenarios": scenarios,
            "emv":       emv,
            "std":       std,
            "var":       var,
        }
    return profiles


def plot_risk_profiles(profiles):
    """
    Individual risk profile bar chart per strategy,
    plus a combined overlay for comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
    fig.patch.set_facecolor("#F4F8FC")

    scenario_labels = ["High Risk\n(p=0.30)", "Moderate Risk\n(p=0.50)", "Low Risk\n(p=0.20)"]
    bar_width = 0.45

    for ax, (alt, profile) in zip(axes, profiles.items()):
        color  = COLORS[alt]
        costs  = [c / 1_000_000 for c in profile["costs"]]
        probs  = profile["probs"]
        emv    = profile["emv"] / 1_000_000
        std    = profile["std"] / 1_000_000

        bars = ax.bar(scenario_labels, probs, width=bar_width,
                      color=color, alpha=0.85, edgecolor="white", linewidth=1.5)

        # Annotate bars with cost values
        for bar, cost, prob in zip(bars, costs, probs):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"${cost:.2f}M\n(p={prob})",
                    ha="center", va="bottom", fontsize=9,
                    color="#0D1B2A", fontweight="bold")

        ax.set_ylim(0, 0.75)
        ax.set_ylabel("Probability", fontsize=10)
        ax.set_title(
            f"{alt.replace('_', ' ')}\nEMV = ${emv:.2f}M  |  Std = ${std:.2f}M",
            fontsize=11, fontweight="bold", color=color
        )
        ax.set_facecolor("white")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # EMV vertical line
        # Map EMV to x position (not directly possible on bar chart,
        # so we annotate it as a text box instead)
        ax.axhline(y=0, color="gray", linewidth=0.5)

    fig.suptitle(
        "Risk Profiles — Probability Distribution of Outcomes per Strategy\n"
        "Each bar shows the probability of that scenario occurring; cost is annotated above",
        fontsize=13, fontweight="bold", color="#0D1B2A", y=1.01
    )
    plt.tight_layout()
    plt.savefig("results/figures/risk_profile_individual.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: risk_profile_individual.png")


def plot_risk_profile_combined(profiles):
    """
    Combined risk profile: all strategies on one chart.
    X-axis = cost outcomes, Y-axis = probability.
    Grouped bars by scenario.
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#F4F8FC")
    ax.set_facecolor("white")

    alts      = list(profiles.keys())
    scenarios = profiles[alts[0]]["scenarios"]
    n_groups  = len(scenarios)
    n_bars    = len(alts)
    x         = np.arange(n_groups)
    width     = 0.25

    scenario_labels = ["High Risk (p=0.30)", "Moderate Risk (p=0.50)", "Low Risk (p=0.20)"]

    for i, alt in enumerate(alts):
        profile = profiles[alt]
        probs   = profile["probs"]
        costs   = [c / 1_000_000 for c in profile["costs"]]
        color   = COLORS[alt]

        bars = ax.bar(x + i * width, probs, width,
                      label=alt.replace("_", " "),
                      color=color, alpha=0.85, edgecolor="white", linewidth=1.2)

        for bar, cost in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"${cost:.2f}M",
                    ha="center", va="bottom", fontsize=7.5,
                    color=color, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(scenario_labels, fontsize=10)
    ax.set_ylabel("Probability of Scenario", fontsize=11)
    ax.set_xlabel("Risk Scenario", fontsize=11)
    ax.set_ylim(0, 0.72)
    ax.set_title(
        "Combined Risk Profile — All Strategies\n"
        "Dollar costs annotated above each bar; strategies with widely spread costs carry more risk",
        fontsize=12, fontweight="bold", color="#0D1B2A"
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/figures/risk_profile_combined.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: risk_profile_combined.png")


def print_risk_profile_summary(profiles):
    import pandas as pd
    rows = []
    for alt, p in profiles.items():
        for scenario, cost, prob in zip(p["scenarios"], p["costs"], p["probs"]):
            rows.append({
                "Strategy":    alt,
                "Scenario":    scenario,
                "Probability": prob,
                "Cost ($)":    round(cost, 2),
                "Prob x Cost": round(prob * cost, 2),
            })
    df = pd.DataFrame(rows)
    print("\n--- RISK PROFILE TABLE ---")
    print(df.to_string(index=False))

    print("\n--- EMV & STANDARD DEVIATION ---")
    for alt, p in profiles.items():
        print(f"  {alt:<15}  EMV = ${p['emv']:>12,.2f}  |  Std Dev = ${p['std']:>12,.2f}")

    df.to_csv("results/tables/risk_profiles.csv", index=False)
    print("\n  Saved: results/tables/risk_profiles.csv")


if __name__ == "__main__":
    print("=" * 60)
    print("RISK PROFILE ANALYSIS")
    print("=" * 60)

    payoff   = build_payoff_table()
    profiles = build_risk_profiles(payoff, PROBABILITIES)

    print_risk_profile_summary(profiles)
    plot_risk_profiles(profiles)
    plot_risk_profile_combined(profiles)

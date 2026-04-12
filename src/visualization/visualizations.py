# src/visualization/visualizations.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.risk_analysis.quantitative import build_sle_ale_table, build_payoff_table, PROBABILITIES
from src.decision_theory.decision_criteria import apply_all_criteria

os.makedirs("results/figures", exist_ok=True)

COLORS = {
    "Advanced_ML": "#2196F3",
    "Standard_ML": "#FF9800",
    "Rule_Based":  "#F44336",
}

def plot_ale_heatmap(table):
    pivot = table.pivot(index="Alternative", columns="Scenario", values="ALE ($)")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot/1_000_000, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "ALE ($ Millions)"})
    ax.set_title("Annual Loss Expectancy (ALE) Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Risk Scenario")
    ax.set_ylabel("Deployment Strategy")
    plt.tight_layout()
    plt.savefig("results/figures/ale_heatmap.png", dpi=150)
    plt.close()
    print("  Saved: ale_heatmap.png")

def plot_total_cost_grouped(table):
    scenarios    = ["High_Risk", "Moderate_Risk", "Low_Risk"]
    alternatives = table["Alternative"].unique()
    x     = np.arange(len(scenarios))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, alt in enumerate(alternatives):
        subset = table[table["Alternative"] == alt]
        costs  = [subset[subset["Scenario"] == s]["Total Cost ($)"].values[0]/1_000_000 for s in scenarios]
        ax.bar(x + i*width, costs, width, label=alt, color=COLORS[alt], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Total Annual Cost ($ Millions)")
    ax.set_title("Total Annual Cost by Strategy and Scenario", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/figures/total_cost_grouped.png", dpi=150)
    plt.close()
    print("  Saved: total_cost_grouped.png")

def plot_emv_comparison(payoff):
    emv  = payoff.mul(pd.Series(PROBABILITIES), axis=1).sum(axis=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(emv.index, emv.values/1_000_000,
                  color=[COLORS[a] for a in emv.index], alpha=0.85, edgecolor="black")
    for bar, val in zip(bars, emv.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"${val/1e6:.2f}M", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Expected Annual Cost ($ Millions)")
    ax.set_title("EMV Comparison Across Strategies", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/figures/emv_comparison.png", dpi=150)
    plt.close()
    print("  Saved: emv_comparison.png")

def plot_regret_heatmap(payoff):
    regret = payoff.apply(lambda col: col - col.min(), axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(regret/1_000_000, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Regret ($ Millions)"})
    ax.set_title("Regret Table", fontsize=14, fontweight="bold")
    ax.set_xlabel("Risk Scenario")
    ax.set_ylabel("Deployment Strategy")
    plt.tight_layout()
    plt.savefig("results/figures/regret_table.png", dpi=150)
    plt.close()
    print("  Saved: regret_table.png")

def plot_criteria_summary(payoff):
    results, emv, regret_table, max_regret, minimin, minimax = apply_all_criteria(payoff, PROBABILITIES)
    criteria        = list(results.keys())
    recommendations = list(results.values())
    cell_colors = []
    for rec in recommendations:
        if rec == "Advanced_ML":
            cell_colors.append(["#BBDEFB", "#BBDEFB"])
        elif rec == "Standard_ML":
            cell_colors.append(["#FFE0B2", "#FFE0B2"])
        else:
            cell_colors.append(["#FFCDD2", "#FFCDD2"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    tbl = ax.table(
        cellText=[[c, r] for c, r in zip(criteria, recommendations)],
        colLabels=["Criterion", "Recommended Strategy"],
        cellLoc="center", loc="center",
        cellColours=cell_colors
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2)
    ax.set_title("Decision Criteria Summary", fontsize=14, fontweight="bold", pad=20)
    patches = [
        mpatches.Patch(color="#BBDEFB", label="Advanced_ML"),
        mpatches.Patch(color="#FFE0B2", label="Standard_ML"),
        mpatches.Patch(color="#FFCDD2", label="Rule_Based"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig("results/figures/criteria_summary.png", dpi=150)
    plt.close()
    print("  Saved: criteria_summary.png")

def plot_fn_fp_comparison():
    df = pd.read_csv("results/tables/model_performance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    name_map = {"XGBoost": "Advanced_ML", "LogisticRegression": "Standard_ML", "RuleBased": "Rule_Based"}
    df["Alternative"] = df["Model"].map(name_map)

    # FN dollar loss
    axes[0].bar(df["Alternative"], df["FN_amount"],
                color=[COLORS[a] for a in df["Alternative"]], alpha=0.85, edgecolor="black")
    axes[0].set_title("Missed Fraud Loss (FN $)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Dollar Loss ($)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    for i, val in enumerate(df["FN_amount"]):
        axes[0].text(i, val + 50, f"${val:,.0f}", ha="center", fontsize=9)

    # FP dollar loss
    axes[1].bar(df["Alternative"], df["FP_amount"],
                color=[COLORS[a] for a in df["Alternative"]], alpha=0.85, edgecolor="black")
    axes[1].set_title("False Alarm Cost (FP $)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Dollar Loss ($)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for i, val in enumerate(df["FP_amount"]):
        axes[1].text(i, val + 50, f"${val:,.0f}", ha="center", fontsize=9)

    plt.suptitle("Real Model Loss Breakdown from Dataset", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/fn_fp_comparison.png", dpi=150)
    plt.close()
    print("  Saved: fn_fp_comparison.png")

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    table  = build_sle_ale_table()
    payoff = build_payoff_table()
    plot_ale_heatmap(table)
    plot_total_cost_grouped(table)
    plot_emv_comparison(payoff)
    plot_regret_heatmap(payoff)
    plot_criteria_summary(payoff)
    plot_fn_fp_comparison()
    print("\n  All figures saved to results/figures/")

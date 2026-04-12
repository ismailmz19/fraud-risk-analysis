# src/decision_theory/decision_tree.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.risk_analysis.quantitative import build_payoff_table, PROBABILITIES

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

COLORS = {
    "Advanced_ML": "#2196F3",
    "Standard_ML": "#FF9800",
    "Rule_Based":  "#F44336",
}

def solve_decision_tree(payoff, probabilities):
    """
    Manually solve the decision tree by rolling back expected values.
    For each alternative, compute EV at the chance node.
    The decision node picks the alternative with lowest EV (cost table).
    """
    probs = pd.Series(probabilities)
    rows  = []

    for alt in payoff.index:
        ev = 0
        branches = []
        for scenario in payoff.columns:
            cost = payoff.loc[alt, scenario]
            prob = probabilities[scenario]
            ev  += cost * prob
            branches.append({
                "Scenario":    scenario,
                "Probability": prob,
                "Cost ($)":    round(cost, 2),
                "EV contrib":  round(cost * prob, 2),
            })
        rows.append({
            "Alternative":  alt,
            "EV ($)":       round(ev, 2),
            "Branches":     branches,
        })

    best = min(rows, key=lambda x: x["EV ($)"])
    return rows, best


def plot_decision_tree(rows, best):
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")

    # Layout constants
    decision_x  = 0.05
    chance_x    = 0.35
    leaf_x      = 0.75
    n_alts      = len(rows)
    alt_ys      = np.linspace(0.85, 0.15, n_alts)
    scenarios   = ["High_Risk", "Moderate_Risk", "Low_Risk"]
    scene_offsets = [0.12, 0, -0.12]

    # Draw decision node (square)
    decision_box = mpatches.FancyBboxPatch(
        (decision_x - 0.03, 0.47), 0.06, 0.06,
        boxstyle="square,pad=0.01", linewidth=2,
        edgecolor="black", facecolor="#E3F2FD"
    )
    ax.add_patch(decision_box)
    ax.text(decision_x, 0.50, "DECISION", ha="center", va="center",
            fontsize=9, fontweight="bold")

    for i, row in enumerate(rows):
        alt   = row["Alternative"]
        alt_y = alt_ys[i]
        color = COLORS[alt]
        ev    = row["EV ($)"]
        is_best = alt == best["Alternative"]

        # Line from decision to chance node
        ax.annotate("", xy=(chance_x - 0.02, alt_y),
                    xytext=(decision_x + 0.03, 0.50),
                    arrowprops=dict(arrowstyle="-", color=color, lw=2))

        # Alternative label
        ax.text((decision_x + chance_x)/2 - 0.02, alt_y + 0.03,
                alt.replace("_", " "), ha="center", fontsize=9,
                color=color, fontweight="bold")

        # Chance node (circle)
        circle = plt.Circle((chance_x, alt_y), 0.025,
                             color=color, fill=True, alpha=0.3, linewidth=2,
                             zorder=5)
        ax.add_patch(circle)
        circle2 = plt.Circle((chance_x, alt_y), 0.025,
                              color=color, fill=False, linewidth=2, zorder=6)
        ax.add_patch(circle2)
        ax.text(chance_x, alt_y, f"EV\n${ev/1e6:.2f}M",
                ha="center", va="center", fontsize=7, fontweight="bold")

        # Best marker
        if is_best:
            ax.text(chance_x - 0.06, alt_y, "★ OPTIMAL",
                    ha="center", va="center", fontsize=8,
                    color="green", fontweight="bold")

        # Branches to leaf nodes
        for j, (scenario, offset) in enumerate(zip(scenarios, scene_offsets)):
            leaf_y = alt_y + offset
            cost   = row["Branches"][j]["Cost ($)"]
            prob   = row["Branches"][j]["Probability"]

            # Branch line
            ax.annotate("", xy=(leaf_x - 0.02, leaf_y),
                        xytext=(chance_x + 0.025, alt_y),
                        arrowprops=dict(arrowstyle="-", color=color, lw=1.2, alpha=0.7))

            # Probability label on branch
            mid_x = (chance_x + leaf_x) / 2
            mid_y = (alt_y + leaf_y) / 2
            ax.text(mid_x, mid_y + 0.015, f"p={prob}",
                    ha="center", fontsize=7, color="gray")

            # Leaf node (rectangle)
            leaf_box = mpatches.FancyBboxPatch(
                (leaf_x - 0.02, leaf_y - 0.025), 0.18, 0.05,
                boxstyle="round,pad=0.01", linewidth=1.5,
                edgecolor=color, facecolor="#FAFAFA"
            )
            ax.add_patch(leaf_box)
            ax.text(leaf_x + 0.07, leaf_y,
                    f"{scenario.replace('_',' ')}\n${cost/1e6:.2f}M",
                    ha="center", va="center", fontsize=7.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Decision Tree — Fraud Detection Deployment\nOptimal: {best['Alternative'].replace('_',' ')} (EV = ${best['EV ($)']/1e6:.2f}M)",
        fontsize=13, fontweight="bold", pad=15
    )

    patches = [mpatches.Patch(color=c, label=a.replace("_"," "))
               for a, c in COLORS.items()]
    ax.legend(handles=patches, loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig("results/figures/decision_tree.png", dpi=150)
    plt.close()
    print("  Saved: decision_tree.png")


if __name__ == "__main__":
    print("=" * 60)
    print("DECISION TREE ANALYSIS")
    print("=" * 60)

    payoff      = build_payoff_table()
    rows, best  = solve_decision_tree(payoff, PROBABILITIES)

    print("\n--- DECISION TREE ROLLBACK ---")
    for row in rows:
        marker = " <-- OPTIMAL" if row["Alternative"] == best["Alternative"] else ""
        print(f"\n  {row['Alternative']}{marker}")
        print(f"  Expected Value: ${row['EV ($)']:,.2f}")
        for b in row["Branches"]:
            print(f"    {b['Scenario']:<18} p={b['Probability']}  cost=${b['Cost ($)']:>12,.2f}  contrib=${b['EV contrib']:>12,.2f}")

    print(f"\n  OPTIMAL DECISION: {best['Alternative']} (EV = ${best['EV ($)']:,.2f})")

    plot_decision_tree(rows, best)

    # Save table
    flat_rows = []
    for row in rows:
        for b in row["Branches"]:
            flat_rows.append({
                "Alternative":  row["Alternative"],
                "EV ($)":       row["EV ($)"],
                "Scenario":     b["Scenario"],
                "Probability":  b["Probability"],
                "Cost ($)":     b["Cost ($)"],
                "EV Contrib":   b["EV contrib"],
            })
    pd.DataFrame(flat_rows).to_csv("results/tables/decision_tree.csv", index=False)
    print("  Saved: decision_tree.csv")

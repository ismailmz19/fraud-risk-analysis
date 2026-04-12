# src/decision_theory/sensitivity.py
# Sensitivity Analysis — breakeven and one-way analysis

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.risk_analysis.quantitative import (
    load_annual_losses, DEPLOYMENT_COSTS, SCENARIO_MULTIPLIERS, PROBABILITIES
)

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

COLORS = {
    "Advanced_ML": "#1A6B9A",
    "Standard_ML": "#F5A623",
    "Rule_Based":  "#D94F3D",
}


def compute_emv_given_probs(p_high, p_mod, p_low):
    """
    Compute EMV for all 3 strategies given explicit scenario probabilities.
    Returns dict {alt: emv}.
    """
    losses = load_annual_losses()
    probs  = {"High_Risk": p_high, "Moderate_Risk": p_mod, "Low_Risk": p_low}
    emvs   = {}
    for alt, base_loss in losses.items():
        emv = 0
        for scenario, multiplier in SCENARIO_MULTIPLIERS.items():
            sle   = base_loss * multiplier
            ale   = sle * probs[scenario]
            total = ale + DEPLOYMENT_COSTS[alt]
            emv  += total * probs[scenario]
        emvs[alt] = emv
    return emvs


def sensitivity_p_high():
    """
    One-way sensitivity: vary P(High Risk) from 0 to 1.
    P(Moderate) and P(Low) scale proportionally from their base ratio (0.50:0.20 = 5:2).
    Returns arrays for plotting.
    """
    p_highs = np.linspace(0.01, 0.95, 300)
    results = {alt: [] for alt in ["Advanced_ML", "Standard_ML", "Rule_Based"]}

    for p_h in p_highs:
        remaining = 1 - p_h
        # Keep Moderate:Low ratio = 5:2 (0.50:0.20)
        p_m = remaining * (0.50 / 0.70)
        p_l = remaining * (0.20 / 0.70)
        emvs = compute_emv_given_probs(p_h, p_m, p_l)
        for alt, emv in emvs.items():
            results[alt].append(emv)

    return p_highs, results


def find_breakeven(p_highs, results, alt_a, alt_b):
    """
    Find P(High Risk) where EMV(alt_a) == EMV(alt_b).
    Returns the breakeven probability or None.
    """
    diff = np.array(results[alt_a]) - np.array(results[alt_b])
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:
            # Linear interpolation
            p1, p2 = p_highs[i], p_highs[i+1]
            d1, d2 = diff[i], diff[i+1]
            breakeven = p1 - d1 * (p2 - p1) / (d2 - d1)
            return round(breakeven, 4)
    return None


def plot_sensitivity(p_highs, results):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#F4F8FC")
    ax.set_facecolor("white")

    for alt, emvs in results.items():
        ax.plot(p_highs, [e / 1_000_000 for e in emvs],
                label=alt.replace("_", " "),
                color=COLORS[alt], linewidth=2.5)

    # Mark current estimate
    ax.axvline(x=0.30, color="gray", linestyle="--", linewidth=1.5,
               label="Current P(High Risk) = 0.30")

    # Find and mark breakeven points
    pairs = [
        ("Advanced_ML", "Standard_ML"),
        ("Advanced_ML", "Rule_Based"),
        ("Standard_ML", "Rule_Based"),
    ]
    breakevens = {}
    for a, b in pairs:
        be = find_breakeven(p_highs, results, a, b)
        if be:
            breakevens[f"{a} vs {b}"] = be
            emv_at_be = results[a][np.argmin(np.abs(p_highs - be))] / 1_000_000
            ax.scatter([be], [emv_at_be], s=80, zorder=5,
                       color="black", marker="x", linewidths=2)
            ax.annotate(f"Breakeven\np={be:.2f}",
                        xy=(be, emv_at_be),
                        xytext=(be + 0.03, emv_at_be + 0.3),
                        fontsize=8, color="black",
                        arrowprops=dict(arrowstyle="->", color="black", lw=1))

    ax.set_xlabel("P(High Risk)", fontsize=11)
    ax.set_ylabel("EMV ($ Millions)", fontsize=11)
    ax.set_title(
        "Sensitivity Analysis — EMV as a Function of P(High Risk)\n"
        "P(Moderate) and P(Low) scale proportionally; vertical line = current estimate",
        fontsize=12, fontweight="bold", color="#0D1B2A"
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/figures/sensitivity_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: sensitivity_analysis.png")

    return breakevens


def one_way_table(p_highs, results):
    """
    Build a summary table at key probability values.
    """
    key_probs = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    rows = []
    for p in key_probs:
        idx = np.argmin(np.abs(p_highs - p))
        row = {"P(High Risk)": p}
        for alt in results:
            row[f"EMV_{alt} ($)"] = round(results[alt][idx], 2)
        # Find best at this probability
        best = min(results.keys(), key=lambda a: results[a][idx])
        row["Best Strategy"] = best
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/sensitivity_analysis.csv", index=False)
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)

    print("\n--- ONE-WAY SENSITIVITY: P(High Risk) ---")
    p_highs, results = sensitivity_p_high()

    # Summary table
    df = one_way_table(p_highs, results)
    print(df.to_string(index=False))

    # Breakeven analysis
    print("\n--- BREAKEVEN POINTS ---")
    breakevens = plot_sensitivity(p_highs, results)
    for pair, be in breakevens.items():
        print(f"  {pair}: breakeven at P(High Risk) = {be}")

    print("\n--- ROBUSTNESS CHECK ---")
    current_p = 0.30
    for pair, be in breakevens.items():
        distance = abs(current_p - be)
        robust   = "ROBUST" if distance > 0.10 else "SENSITIVE"
        print(f"  {pair}: distance from breakeven = {distance:.2f}  [{robust}]")

    print(f"\n  Current estimate P(High Risk) = {current_p}")
    print("  A decision is robust if it is far (>0.10) from any breakeven point.")
    print("\n  Saved: results/tables/sensitivity_analysis.csv")
    print("  Saved: results/figures/sensitivity_analysis.png")

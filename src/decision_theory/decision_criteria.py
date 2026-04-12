# src/decision_theory/decision_criteria.py
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.risk_analysis.quantitative import build_payoff_table, PROBABILITIES

def apply_all_criteria(payoff, probabilities, alpha=0.6):
    results = {}

    # 1. MINIMIN
    minimin = payoff.min(axis=1)
    results["Minimin (Optimistic)"] = minimin.idxmin()

    # 2. MINIMAX
    minimax = payoff.max(axis=1)
    results["Minimax (Conservative)"] = minimax.idxmin()

    # 3. MINIMAX REGRET
    regret_table = payoff.apply(lambda col: col - col.min(), axis=0)
    max_regret   = regret_table.max(axis=1)
    results["Minimax Regret"] = max_regret.idxmin()

    # 4. EMV
    probs = pd.Series(probabilities)
    emv   = payoff.mul(probs, axis=1).sum(axis=1)
    results["EMV (Expected Cost)"] = emv.idxmin()

    return results, emv, regret_table, max_regret, minimin, minimax


def print_full_analysis(payoff, probabilities):
    results, emv, regret_table, max_regret, minimin, minimax = apply_all_criteria(payoff, probabilities)

    print("=" * 60)
    print("DECISION CRITERIA ANALYSIS")
    print("=" * 60)

    print("\n--- PAYOFF TABLE (Total Annual Cost $) ---")
    print(payoff.applymap(lambda x: f"${x:,.0f}").to_string())

    print("\n--- MINIMIN (Best case per strategy) ---")
    print(minimin.apply(lambda x: f"${x:,.0f}").to_string())

    print("\n--- MINIMAX (Worst case per strategy) ---")
    print(minimax.apply(lambda x: f"${x:,.0f}").to_string())

    print("\n--- REGRET TABLE ---")
    print(regret_table.applymap(lambda x: f"${x:,.0f}").to_string())

    print("\n--- MAX REGRET per strategy ---")
    print(max_regret.apply(lambda x: f"${x:,.0f}").to_string())

    print(f"\n--- EMV (Probabilities: {probabilities}) ---")
    print(emv.apply(lambda x: f"${x:,.0f}").to_string())

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for criterion, recommendation in results.items():
        print(f"  {criterion:<30} -> {recommendation}")

    return results


if __name__ == "__main__":
    payoff  = build_payoff_table()
    results = print_full_analysis(payoff, PROBABILITIES)
    os.makedirs("results/tables", exist_ok=True)
    summary = pd.DataFrame(list(results.items()), columns=["Criterion", "Recommended Strategy"])
    summary.to_csv("results/tables/decision_criteria_summary.csv", index=False)
    print("\n  Saved: results/tables/decision_criteria_summary.csv")

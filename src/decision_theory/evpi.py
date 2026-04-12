# src/decision_theory/evpi.py
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.risk_analysis.quantitative import build_payoff_table, PROBABILITIES

def compute_emv(payoff, probabilities):
    probs = pd.Series(probabilities)
    emv   = payoff.mul(probs, axis=1).sum(axis=1)
    return emv, emv.idxmin(), emv.min()

def compute_evwpi(payoff, probabilities):
    evwpi = 0
    best_per_scenario = {}
    for scenario, prob in probabilities.items():
        best_cost = payoff[scenario].min()
        best_alt  = payoff[scenario].idxmin()
        best_per_scenario[scenario] = (best_alt, best_cost)
        evwpi += best_cost * prob
    return round(evwpi, 2), best_per_scenario

def compute_evpi(payoff, probabilities):
    emv, best_alt, best_emv = compute_emv(payoff, probabilities)
    evwpi, best_per_scenario = compute_evwpi(payoff, probabilities)
    evpi = best_emv - evwpi
    return round(evpi, 2), round(evwpi, 2), round(best_emv, 2), best_alt, best_per_scenario, emv

def compute_evsi(payoff, probabilities, posterior_probs, cost_of_info=50_000):
    _, _, best_emv_prior    = compute_emv(payoff, probabilities)
    _, _, best_emv_posterior = compute_emv(payoff, posterior_probs)
    evsi     = best_emv_prior - best_emv_posterior
    net_evsi = evsi - cost_of_info
    return {
        "EMV (Prior)":      round(best_emv_prior, 2),
        "EMV (Posterior)":  round(best_emv_posterior, 2),
        "EVSI":             round(evsi, 2),
        "Cost of Info":     cost_of_info,
        "Net EVSI":         round(net_evsi, 2),
        "Worth it?":        "YES" if net_evsi > 0 else "NO",
    }

if __name__ == "__main__":
    print("=" * 60)
    print("EVPI / EVwPI / EVSI ANALYSIS")
    print("=" * 60)

    payoff = build_payoff_table()

    emv, best_alt, best_emv = compute_emv(payoff, PROBABILITIES)
    print("\n--- EMV per Strategy ---")
    for alt, val in emv.items():
        marker = " <-- BEST" if alt == best_alt else ""
        print(f"  {alt:<15} ${val:>15,.2f}{marker}")

    evwpi, best_per_scenario = compute_evwpi(payoff, PROBABILITIES)
    print("\n--- Best Choice per Scenario (with Perfect Info) ---")
    for scenario, (alt, cost) in best_per_scenario.items():
        print(f"  {scenario:<18} -> {alt:<15} (${cost:,.2f})")
    print(f"\n  EVwPI = ${evwpi:,.2f}")

    evpi, evwpi, best_emv, best_alt, _, emv = compute_evpi(payoff, PROBABILITIES)
    print(f"\n--- EVPI ---")
    print(f"  EMV (best without info) = ${best_emv:,.2f}  [{best_alt}]")
    print(f"  EVwPI                   = ${evwpi:,.2f}")
    print(f"  EVPI                    = ${evpi:,.2f}")
    print(f"  Interpretation: Pay at most ${evpi:,.2f}/year for perfect threat intelligence.")

    posterior = {"High_Risk": 0.5217, "Moderate_Risk": 0.4348, "Low_Risk": 0.0435}
    print(f"\n--- EVSI (3-month Pilot Test, cost = $50,000) ---")
    evsi_result = compute_evsi(payoff, PROBABILITIES, posterior, cost_of_info=50_000)
    for k, v in evsi_result.items():
        print(f"  {k:<25} {v}")

    raw_evsi   = evsi_result["EVSI"]
    efficiency = (raw_evsi / evpi * 100) if evpi != 0 else 0
    print(f"\n  EVSI Efficiency = {efficiency:.1f}%")

    os.makedirs("results/tables", exist_ok=True)
    rows = [
        {"Metric": "EVwPI",      "Value ($)": evwpi},
        {"Metric": "EMV (best)", "Value ($)": best_emv},
        {"Metric": "EVPI",       "Value ($)": evpi},
        {"Metric": "EVSI",       "Value ($)": raw_evsi},
        {"Metric": "Net EVSI",   "Value ($)": evsi_result["Net EVSI"]},
        {"Metric": "Efficiency", "Value ($)": f"{efficiency:.1f}%"},
    ]
    pd.DataFrame(rows).to_csv("results/tables/evpi_evsi.csv", index=False)
    print("\n  Saved: results/tables/evpi_evsi.csv")

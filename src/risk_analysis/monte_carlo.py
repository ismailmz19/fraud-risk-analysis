import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('src')
from risk_analysis.quantitative import ASSETS, DEPLOYMENT_COSTS, calculate_sle, calculate_ale

TOTAL_ASSET_VALUE = sum(ASSETS.values())
ARO_BASE = {"High_Risk": 0.30, "Moderate_Risk": 0.50, "Low_Risk": 0.20}
EXPOSURE_FACTORS_BASE = {
    "Advanced_ML": {"High_Risk": 0.4889, "Moderate_Risk": 0.2444, "Low_Risk": 0.0978},
    "Standard_ML": {"High_Risk": 0.55, "Moderate_Risk": 0.264, "Low_Risk": 0.088},
    "Rule_Based":  {"High_Risk": 0.3361, "Moderate_Risk": 0.1833, "Low_Risk": 0.0611},
}
DEPLOYMENT_COSTS_BASE = DEPLOYMENT_COSTS.copy()

np.random.seed(42)
n_iterations = 10000

# Store results
costs = {alt: [] for alt in DEPLOYMENT_COSTS_BASE.keys()}

for i in range(n_iterations):
    # Sample ARO High Risk
    aro_high = np.random.triangular(0.1, 0.3, 0.6)
    aro = ARO_BASE.copy()
    aro['High_Risk'] = aro_high

    # Sample EF for each alt/scenario
    ef_sampled = {}
    for alt in EXPOSURE_FACTORS_BASE:
        ef_sampled[alt] = {}
        for scenario in EXPOSURE_FACTORS_BASE[alt]:
            mean_ef = EXPOSURE_FACTORS_BASE[alt][scenario]
            ef = np.random.normal(mean_ef, 0.02)
            ef = np.clip(ef, 0, 1)
            ef_sampled[alt][scenario] = ef

    # Sample deploy costs
    deploy_sampled = {}
    for alt in DEPLOYMENT_COSTS_BASE:
        base = DEPLOYMENT_COSTS_BASE[alt]
        deploy = np.random.uniform(base * 0.8, base * 1.2)
        deploy_sampled[alt] = deploy

    # Compute total cost for each alt
    for alt in DEPLOYMENT_COSTS_BASE:
        total_cost = 0
        for scenario in aro:
            sle = calculate_sle(TOTAL_ASSET_VALUE, ef_sampled[alt][scenario])
            ale = calculate_ale(sle, aro[scenario])
            total_cost += ale
        total_cost += deploy_sampled[alt]
        costs[alt].append(total_cost)

# Convert to arrays
for alt in costs:
    costs[alt] = np.array(costs[alt])

# Statistics
stats = {}
for alt in costs:
    arr = costs[alt]
    stats[alt] = {
        'Mean': np.mean(arr),
        'Std': np.std(arr),
        '5th_Percentile': np.percentile(arr, 5),
        '95th_Percentile': np.percentile(arr, 95),
        'VaR_95': np.percentile(arr, 5)  # Since higher cost is risk
    }

# Probability cheapest
cheapest_counts = {alt: 0 for alt in costs}
for i in range(n_iterations):
    iter_costs = {alt: costs[alt][i] for alt in costs}
    cheapest = min(iter_costs, key=iter_costs.get)
    cheapest_counts[cheapest] += 1

prob_cheapest = {alt: count / n_iterations for alt, count in cheapest_counts.items()}

# Save to CSV
rows = []
for alt in stats:
    row = {'Strategy': alt, **stats[alt], 'Prob_Cheapest': prob_cheapest[alt]}
    rows.append(row)
df = pd.DataFrame(rows)
os.makedirs('results/tables', exist_ok=True)
df.to_csv('results/tables/monte_carlo_results.csv', index=False)

# Plot distributions
fig, ax = plt.subplots(figsize=(10, 6))
for alt in costs:
    ax.hist(costs[alt], bins=50, alpha=0.5, label=alt)
ax.set_xlabel('Total Annual Cost ($)')
ax.set_ylabel('Frequency')
ax.set_title('Monte Carlo Cost Distributions')
ax.legend()
plt.savefig('results/figures/monte_carlo_distributions.png', dpi=300, bbox_inches='tight')

# Plot CDF
fig2, ax2 = plt.subplots(figsize=(10, 6))
for alt in costs:
    sorted_costs = np.sort(costs[alt])
    cdf = np.arange(1, len(sorted_costs)+1) / len(sorted_costs)
    ax2.plot(sorted_costs, cdf, label=alt)
ax2.set_xlabel('Total Annual Cost ($)')
ax2.set_ylabel('Cumulative Probability')
ax2.set_title('Risk Profiles (CDF)')
ax2.legend()
plt.savefig('results/figures/monte_carlo_cdf.png', dpi=300, bbox_inches='tight')

print("Monte Carlo simulation completed.")
print("Results saved to results/tables/monte_carlo_results.csv, results/figures/monte_carlo_distributions.png, results/figures/monte_carlo_cdf.png")
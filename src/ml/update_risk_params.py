import pandas as pd
import numpy as np
import os

# Load model performance
perf_df = pd.read_csv('results/tables/model_performance.csv')

# Calculate metrics for each model
metrics = []
for _, row in perf_df.iterrows():
    tp = row['TP']
    fp = row['FP']
    fn = row['FN']
    tn = row['TN']

    total_pos = tp + fn  # total fraud
    total_neg = fp + tn  # total legit

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Relative Risk: RR = P(detected | fraud) / P(detected | legit) = recall / (1 - specificity)
    rr = recall / (1 - specificity) if (1 - specificity) > 0 else np.inf

    # Odds Ratio: OR = (tp/fp) / (fn/tn) = (tp * tn) / (fp * fn)
    or_val = (tp * tn) / (fp * fn) if fp * fn > 0 else np.inf

    metrics.append({
        'Model': row['Model'],
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'FPR': fpr,
        'Specificity': specificity,
        'Relative_Risk': rr,
        'Odds_Ratio': or_val,
        'FN_rate': fn / (tp + fn) if (tp + fn) > 0 else 0
    })

metrics_df = pd.DataFrame(metrics)

# Compute max_FN_rate
max_fn_rate = metrics_df['FN_rate'].max()

# Base EF max = 0.55
base_ef_max = 0.55

# Compute new base EF for each model
metrics_df['Base_EF'] = (metrics_df['FN_rate'] / max_fn_rate) * base_ef_max

# Load current quantitative.py to get current ratios
# But since we can't modify it yet, hardcode the current ratios
current_ef = {
    "Advanced_ML": {"High_Risk": 0.10, "Moderate_Risk": 0.05, "Low_Risk": 0.02},
    "Standard_ML": {"High_Risk": 0.25, "Moderate_Risk": 0.12, "Low_Risk": 0.04},
    "Rule_Based":  {"High_Risk": 0.55, "Moderate_Risk": 0.30, "Low_Risk": 0.10},
}

# Compute ratios
ratios = {}
for alt in current_ef:
    high = current_ef[alt]["High_Risk"]
    ratios[alt] = {
        "High_Risk": high / high,
        "Moderate_Risk": current_ef[alt]["Moderate_Risk"] / high,
        "Low_Risk": current_ef[alt]["Low_Risk"] / high
    }

# Map models to alternatives
model_to_alt = {
    'XGBoost': 'Advanced_ML',
    'LogisticRegression': 'Standard_ML',
    'RuleBased': 'Rule_Based'
}

# Compute new EF
updated_ef = {}
for _, row in metrics_df.iterrows():
    model = row['Model']
    alt = model_to_alt[model]
    base_ef = row['Base_EF']
    updated_ef[alt] = {}
    for scenario in ['High_Risk', 'Moderate_Risk', 'Low_Risk']:
        updated_ef[alt][scenario] = base_ef * ratios[alt][scenario]

# Add to metrics_df
for _, row in metrics_df.iterrows():
    model = row['Model']
    alt = model_to_alt[model]
    for scenario in ['High_Risk', 'Moderate_Risk', 'Low_Risk']:
        metrics_df.loc[metrics_df['Model'] == model, f'EF_{scenario}'] = updated_ef[alt][scenario]

# Save to CSV
os.makedirs('results/tables', exist_ok=True)
metrics_df.to_csv('results/tables/updated_risk_params.csv', index=False)

print("Updated risk parameters saved to results/tables/updated_risk_params.csv")
print(metrics_df)
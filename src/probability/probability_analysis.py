# src/probability/probability_analysis.py
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PROBABILITIES = {"High_Risk": 0.30, "Moderate_Risk": 0.50, "Low_Risk": 0.20}

def load_model_metrics():
    df = pd.read_csv("results/tables/model_performance.csv").set_index("Model")
    metrics = {}
    for model in df.index:
        TP = df.loc[model, "TP"]
        FP = df.loc[model, "FP"]
        FN = df.loc[model, "FN"]
        TN = df.loc[model, "TN"]
        total       = TP + FP + FN + TN
        precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall      = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        accuracy    = (TP + TN) / total
        f1          = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
        fpr         = FP / (FP + TN) if (FP + TN) > 0 else 0
        metrics[model] = {
            "Precision":        round(precision, 4),
            "Recall":           round(recall, 4),
            "Specificity":      round(specificity, 4),
            "Accuracy":         round(accuracy, 6),
            "F1 Score":         round(f1, 4),
            "FPR":              round(fpr, 6),
            "P(Fraud|Flagged)": round(precision, 4),
            "P(Fraud|Cleared)": round(FN/(FN+TN), 6) if (FN+TN) > 0 else 0,
        }
    return pd.DataFrame(metrics).T

def compute_relative_risk(df_raw):
    pairs = [
        ("RuleBased",          "XGBoost"),
        ("LogisticRegression", "XGBoost"),
        ("RuleBased",          "LogisticRegression"),
    ]
    results = []
    for a, b in pairs:
        FN_a = df_raw.loc[a,"FN"]; TP_a = df_raw.loc[a,"TP"]
        FN_b = df_raw.loc[b,"FN"]; TP_b = df_raw.loc[b,"TP"]
        miss_a = FN_a / (TP_a + FN_a)
        miss_b = FN_b / (TP_b + FN_b)
        rr     = miss_a / miss_b if miss_b > 0 else float("inf")
        odds_a = FN_a / TP_a    if TP_a   > 0 else float("inf")
        odds_b = FN_b / TP_b    if TP_b   > 0 else float("inf")
        oratio = odds_a / odds_b if odds_b > 0 else float("inf")
        results.append({"Comparison": f"{a} vs {b}",
                        "RR": round(rr,4), "OR": round(oratio,4)})
    return pd.DataFrame(results)

def bayesian_update(prior, likelihoods):
    p_evidence = sum(likelihoods[s] * prior[s] for s in prior)
    posterior  = {s: round(likelihoods[s]*prior[s]/p_evidence, 4) for s in prior}
    return posterior, round(p_evidence, 4)

if __name__ == "__main__":
    print("=" * 60)
    print("PROBABILITY ANALYSIS")
    print("=" * 60)

    metrics = load_model_metrics()
    print("\n--- MODEL PERFORMANCE METRICS ---")
    print(metrics.to_string())

    df_raw = pd.read_csv("results/tables/model_performance.csv").set_index("Model")
    rr_df  = compute_relative_risk(df_raw)
    print("\n--- RELATIVE RISK & ODDS RATIO ---")
    print(rr_df.to_string(index=False))

    print("\n--- BAYESIAN UPDATE ---")
    likelihoods = {"High_Risk": 0.80, "Moderate_Risk": 0.40, "Low_Risk": 0.10}
    posterior, p_ev = bayesian_update(PROBABILITIES, likelihoods)
    print(f"  Prior:       {PROBABILITIES}")
    print(f"  P(Evidence): {p_ev}")
    print(f"  Posterior:   {posterior}")

    os.makedirs("results/tables", exist_ok=True)
    metrics.to_csv("results/tables/model_metrics.csv")
    rr_df.to_csv("results/tables/relative_risk.csv", index=False)
    print("\n  Saved: model_metrics.csv, relative_risk.csv")

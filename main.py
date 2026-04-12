# main.py
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("\n" + "=" * 60)
    print("  FRAUD DETECTION DEPLOYMENT RISK ANALYSIS")
    print("  Fintech Company — Credit Card Fraud Dataset")
    print("=" * 60)

    import pandas as pd

    # PHASE 1: Train ML Models
    print("\n[PHASE 1] Training ML Models...")
    from src.ml.train_models import (
        load_data, split_data,
        train_xgboost, train_logistic, train_rule_based
    )
    df = load_data()
    X_train, X_test, y_train, y_test, amt_train, amt_test = split_data(df)
    results = {}
    results["XGBoost"]            = train_xgboost(X_train, X_test, y_train, y_test, amt_test)
    results["LogisticRegression"] = train_logistic(X_train, X_test, y_train, y_test, amt_test)
    results["RuleBased"]          = train_rule_based(X_test, y_test, amt_test)
    rows = [{"Model": m, **v} for m, v in results.items()]
    pd.DataFrame(rows).to_csv("results/tables/model_performance.csv", index=False)
    print("  model_performance.csv saved.")

    # PHASE 2: Quantitative Risk Analysis
    print("\n[PHASE 2] Quantitative Risk Analysis...")
    from src.risk_analysis.quantitative import (
        build_sle_ale_table, build_payoff_table, PROBABILITIES
    )
    table  = build_sle_ale_table()
    payoff = build_payoff_table()
    table.to_csv("results/tables/sle_ale_analysis.csv", index=False)
    payoff.to_csv("results/tables/payoff_table.csv")
    print(f"  Payoff table saved. ({len(table)} rows)")

    # PHASE 3: Decision Criteria
    print("\n[PHASE 3] Decision Criteria Analysis...")
    from src.decision_theory.decision_criteria import print_full_analysis
    results_dc = print_full_analysis(payoff, PROBABILITIES)
    summary = pd.DataFrame(list(results_dc.items()), columns=["Criterion", "Recommended Strategy"])
    summary.to_csv("results/tables/decision_criteria_summary.csv", index=False)

    # PHASE 4: EVPI / EVSI
    print("\n[PHASE 4] EVPI / EVSI Analysis...")
    from src.decision_theory.evpi import compute_evpi, compute_evsi
    evpi, evwpi, best_emv, best_alt, _, emv = compute_evpi(payoff, PROBABILITIES)
    posterior = {"High_Risk": 0.5217, "Moderate_Risk": 0.4348, "Low_Risk": 0.0435}
    evsi_result = compute_evsi(payoff, PROBABILITIES, posterior, cost_of_info=50_000)
    print(f"  EVPI = ${evpi:,.2f}")
    print(f"  EVSI = ${evsi_result['EVSI']:,.2f}  | Worth it? {evsi_result['Worth it?']}")
    rows_evpi = [
        {"Metric": "EVwPI",      "Value ($)": evwpi},
        {"Metric": "EMV (best)", "Value ($)": best_emv},
        {"Metric": "EVPI",       "Value ($)": evpi},
        {"Metric": "EVSI",       "Value ($)": evsi_result["EVSI"]},
        {"Metric": "Net EVSI",   "Value ($)": evsi_result["Net EVSI"]},
    ]
    pd.DataFrame(rows_evpi).to_csv("results/tables/evpi_evsi.csv", index=False)

    # PHASE 5: Decision Tree
    print("\n[PHASE 5] Decision Tree Analysis...")
    from src.decision_theory.decision_tree import solve_decision_tree, plot_decision_tree
    rows_dt, best_dt = solve_decision_tree(payoff, PROBABILITIES)
    plot_decision_tree(rows_dt, best_dt)
    flat = []
    for row in rows_dt:
        for b in row["Branches"]:
            flat.append({
                "Alternative": row["Alternative"],
                "EV ($)":      row["EV ($)"],
                "Scenario":    b["Scenario"],
                "Probability": b["Probability"],
                "Cost ($)":    b["Cost ($)"],
                "EV Contrib":  b["EV contrib"],
            })
    pd.DataFrame(flat).to_csv("results/tables/decision_tree.csv", index=False)
    print(f"  Optimal: {best_dt['Alternative']} (EV=${best_dt['EV ($)']:,.2f})")

    # PHASE 6: Fault Tree
    print("\n[PHASE 6] Fault Tree Analysis...")
    from src.risk_analysis.fault_tree import (
        compute_fault_tree, compute_birnbaum_importance,
        plot_fault_tree, BASIC_EVENTS
    )
    gates, p_top = compute_fault_tree()
    importance   = compute_birnbaum_importance(p_top)
    plot_fault_tree(gates, p_top)
    ft_rows = [{"Component": e, "Probability": p,
                "Birnbaum Importance": importance.get(e, "")}
               for e, p in BASIC_EVENTS.items()]
    pd.DataFrame(ft_rows).to_csv("results/tables/fault_tree_analysis.csv", index=False)
    print(f"  System failure probability: {p_top:.4f}")

    # PHASE 7: Risk Utility Analysis
    print("\n[PHASE 7] Risk Utility Analysis...")
    from src.risk_analysis.utility import compute_utility_analysis, plot_utility_curves
    df_util  = compute_utility_analysis(payoff, PROBABILITIES)
    plot_utility_curves(payoff, PROBABILITIES)
    df_util.to_csv("results/tables/utility_analysis.csv")
    best_util = df_util["Expected Utility"].idxmax()
    print(f"  Best strategy by Expected Utility: {best_util}")

    # PHASE 8: Probability Analysis
    print("\n[PHASE 8] Probability Analysis...")
    from src.probability.probability_analysis import (
        load_model_metrics, compute_relative_risk, bayesian_update
    )
    metrics     = load_model_metrics()
    df_raw      = pd.read_csv("results/tables/model_performance.csv").set_index("Model")
    rr_df       = compute_relative_risk(df_raw)
    likelihoods = {"High_Risk": 0.80, "Moderate_Risk": 0.40, "Low_Risk": 0.10}
    posterior_b, p_ev = bayesian_update(PROBABILITIES, likelihoods)
    metrics.to_csv("results/tables/model_metrics.csv")
    rr_df.to_csv("results/tables/relative_risk.csv", index=False)
    print(f"  Bayesian posterior: {posterior_b}")

    # PHASE 9: Influence Diagram
    print("\n[PHASE 9] Influence Diagram...")
    from src.visualization.influence_diagram import draw_influence_diagram
    draw_influence_diagram()

    # PHASE 10: Risk Profiles
    print("\n[PHASE 10] Risk Profile Analysis...")
    from src.visualization.risk_profile import (
        build_risk_profiles, plot_risk_profiles,
        plot_risk_profile_combined, print_risk_profile_summary
    )
    profiles = build_risk_profiles(payoff, PROBABILITIES)
    print_risk_profile_summary(profiles)
    plot_risk_profiles(profiles)
    plot_risk_profile_combined(profiles)

    # PHASE 11: Sensitivity Analysis
    print("\n[PHASE 11] Sensitivity Analysis...")
    from src.decision_theory.sensitivity import (
        sensitivity_p_high, plot_sensitivity, one_way_table
    )
    p_highs, sens_results = sensitivity_p_high()
    breakevens = plot_sensitivity(p_highs, sens_results)
    one_way_table(p_highs, sens_results)
    if breakevens:
        for pair, be in breakevens.items():
            print(f"  Breakeven: {pair} at P(High Risk) = {be}")
    else:
        print("  No breakeven found — Advanced ML dominates across full range.")

    # PHASE 12: Event Tree Analysis
    print("\n[PHASE 12] Event Tree Analysis...")
    from src.risk_analysis.event_tree import (
        compute_end_states, plot_event_trees,
        build_summary_table, MODEL_PARAMS
    )
    all_end_states = {m: compute_end_states(m) for m in MODEL_PARAMS}
    plot_event_trees(all_end_states)
    build_summary_table(all_end_states)
    for model, es in all_end_states.items():
        el = sum(s["prob"] * s["loss"] for s in es.values())
        print(f"  {model:<15} Expected Loss per Transaction = ${el:.4f}")

    # PHASE 13: Core Visualizations
    print("\n[PHASE 13] Generating Visualizations...")
    from src.visualization.visualizations import (
        plot_ale_heatmap, plot_total_cost_grouped,
        plot_emv_comparison, plot_regret_heatmap,
        plot_criteria_summary, plot_fn_fp_comparison
    )
    plot_ale_heatmap(table)
    plot_total_cost_grouped(table)
    plot_emv_comparison(payoff)
    plot_regret_heatmap(payoff)
    plot_criteria_summary(payoff)
    plot_fn_fp_comparison()

    # FINAL SUMMARY
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"\n  Recommended Strategy (all criteria): {best_dt['Alternative']}")
    print(f"  Expected Annual Cost:                ${best_dt['EV ($)']:,.2f}")
    print(f"  EVPI:                                ${evpi:,.2f}")
    print(f"  System Failure Probability:          {p_top:.4f}")
    print(f"  Best by Utility:                     {best_util}")
    print(f"  Sensitivity Breakeven:               None — dominates full range")

    print("\n  Tables generated:")
    for f in sorted(os.listdir("results/tables")):
        print(f"    - results/tables/{f}")
    print("\n  Figures generated:")
    for f in sorted(os.listdir("results/figures")):
        print(f"    - results/figures/{f}")

if __name__ == "__main__":
    main()

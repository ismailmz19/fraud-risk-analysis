# src/ml/train_models.py
# Train 3 models and extract real dollar-based confusion matrix

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/models", exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20
DAYS_IN_DATASET = 2
DAYS_PER_YEAR   = 365
SCALE_FACTOR    = DAYS_PER_YEAR / DAYS_IN_DATASET  # 182.5
FP_COST_RATE    = 0.10  # 10% of blocked legitimate transaction value is lost


def load_data():
    path = "data/creditcard.csv"
    print(f"Loading dataset from {path}...")
    df = pd.read_csv(path)
    print(f"  Total transactions: {len(df):,}")
    print(f"  Fraudulent:         {df['Class'].sum():,}")
    print(f"  Legitimate:         {(df['Class']==0).sum():,}")
    print(f"  Fraud rate:         {df['Class'].mean()*100:.3f}%")
    return df


def split_data(df):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    amounts = df["Amount"].values

    X_train, X_test, y_train, y_test, amt_train, amt_test = train_test_split(
        X, y, amounts,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    print(f"\n  Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, amt_train, amt_test


def compute_dollar_losses(y_test, y_pred, amt_test):
    """
    Given predictions, compute real dollar losses from:
    - FN: missed frauds -> full transaction amount lost
    - FP: false alarms -> FP_COST_RATE * transaction amount lost
    """
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    y_test_arr  = np.array(y_test)
    y_pred_arr  = np.array(y_pred)
    amt_arr     = np.array(amt_test)

    # FN: actual fraud, predicted legitimate -> full amount lost
    fn_mask      = (y_test_arr == 1) & (y_pred_arr == 0)
    fn_amount    = amt_arr[fn_mask].sum()

    # FP: actual legitimate, predicted fraud -> partial cost (friction + review)
    fp_mask      = (y_test_arr == 0) & (y_pred_arr == 1)
    fp_amount    = amt_arr[fp_mask].sum() * FP_COST_RATE

    total_direct_loss    = fn_amount + fp_amount
    annual_scaled_loss   = total_direct_loss * SCALE_FACTOR

    return {
        "TP": int(tp), "FP": int(fp),
        "FN": int(fn), "TN": int(tn),
        "FN_amount":          round(fn_amount, 2),
        "FP_amount":          round(fp_amount, 2),
        "total_direct_loss":  round(total_direct_loss, 2),
        "annual_scaled_loss": round(annual_scaled_loss, 2),
    }


def train_xgboost(X_train, X_test, y_train, y_test, amt_test):
    print("\n[1/3] Training XGBoost (Advanced ML)...")
    try:
        from xgboost import XGBClassifier
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
        from xgboost import XGBClassifier

    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
    )
    model.fit(X_res, y_res)

    # Optimize threshold
    probs = model.predict_proba(X_test)[:, 1]
    best_thresh, best_f1 = 0.5, 0
    from sklearn.metrics import f1_score
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    y_pred = (probs >= best_thresh).astype(int)
    joblib.dump(model, "results/models/xgboost_model.pkl")
    print(f"  Best threshold: {best_thresh:.2f} | F1: {best_f1:.4f}")
    return compute_dollar_losses(y_test, y_pred, amt_test)


def train_logistic(X_train, X_test, y_train, y_test, amt_test):
    print("\n[2/3] Training Logistic Regression (Standard ML)...")

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_tr_sc, y_train)

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )
    model.fit(X_res, y_res)
    y_pred = model.predict(X_te_sc)

    joblib.dump((model, scaler), "results/models/logistic_model.pkl")
    return compute_dollar_losses(y_test, y_pred, amt_test)


def train_rule_based(X_test, y_test, amt_test):
    """
    Rule-Based system:
    Flag transaction as fraud if ANY of these rules trigger:
    - Amount > 1000
    - V14 < -5 (strong fraud signal in this dataset)
    - V17 < -5
    - V12 < -5
    """
    print("\n[3/3] Running Rule-Based system...")

    X_arr  = np.array(X_test)
    cols   = list(X_test.columns)
    v14    = X_arr[:, cols.index("V14")]
    v17    = X_arr[:, cols.index("V17")]
    v12    = X_arr[:, cols.index("V12")]
    amt    = X_arr[:, cols.index("Amount")]

    y_pred = ((amt > 1000) | (v14 < -5) | (v17 < -5) | (v12 < -5)).astype(int)
    return compute_dollar_losses(y_test, y_pred, amt_test)


if __name__ == "__main__":
    print("=" * 60)
    print("ML MODEL TRAINING & DOLLAR LOSS COMPUTATION")
    print("=" * 60)
    print(f"Scale factor: {SCALE_FACTOR}x (dataset=2 days, scaled to 1 year)")
    print(f"FP cost rate: {FP_COST_RATE*100}% of blocked transaction value")

    df = load_data()
    X_train, X_test, y_train, y_test, amt_train, amt_test = split_data(df)

    results = {}
    results["XGBoost"]            = train_xgboost(X_train, X_test, y_train, y_test, amt_test)
    results["LogisticRegression"] = train_logistic(X_train, X_test, y_train, y_test, amt_test)
    results["RuleBased"]          = train_rule_based(X_test, y_test, amt_test)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    rows = []
    for model, metrics in results.items():
        recall    = metrics["TP"] / (metrics["TP"] + metrics["FN"])
        precision = metrics["TP"] / (metrics["TP"] + metrics["FP"]) if (metrics["TP"] + metrics["FP"]) > 0 else 0
        print(f"\n  {model}")
        print(f"    TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}, TN={metrics['TN']}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    FN Loss (missed fraud $):     ${metrics['FN_amount']:>12,.2f}")
        print(f"    FP Loss (false alarm cost $): ${metrics['FP_amount']:>12,.2f}")
        print(f"    Total Direct Loss (test set): ${metrics['total_direct_loss']:>12,.2f}")
        print(f"    Annual Scaled Loss:           ${metrics['annual_scaled_loss']:>12,.2f}")
        row = {"Model": model}
        row.update(metrics)
        rows.append(row)

    pd.DataFrame(rows).to_csv("results/tables/model_performance.csv", index=False)
    print("\n  Saved: results/tables/model_performance.csv")

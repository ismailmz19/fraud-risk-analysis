# src/risk_analysis/event_tree.py
# Event Tree Analysis — fraud transaction submitted as initiating event

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# EVENT TREE STRUCTURE
#
# Initiating Event: Fraud Transaction Submitted
#
# Branch 1 — Detection: Does the model flag it?
#   YES (detected)  → Branch 2
#   NO  (missed)    → End State: Financial Loss (full amount)
#
# Branch 2 — Correct Classification: Is the flag a true positive?
#   YES (TP: real fraud correctly flagged) → Branch 3
#   NO  (FP: legitimate tx wrongly flagged) → End State: Operational Loss
#
# Branch 3 — Response Effectiveness: Is the fraud blocked in time?
#   YES → End State: Fraud Prevented (no loss)
#   NO  → End State: Partial Loss (fraud partially processed)
# ─────────────────────────────────────────────────────────────────

# Model-specific probabilities derived from real confusion matrices
MODEL_PARAMS = {
    "Advanced_ML": {
        "P_detect":      0.8367,   # recall = TP/(TP+FN)
        "P_correct":     0.8542,   # precision = TP/(TP+FP)
        "P_response":    0.95,     # probability of timely block given correct flag
    },
    "Standard_ML": {
        "P_detect":      0.9184,
        "P_correct":     0.0578,
        "P_response":    0.90,
    },
    "Rule_Based": {
        "P_detect":      0.7857,
        "P_correct":     0.1034,
        "P_response":    0.85,
    },
}

# Average fraud transaction amount from dataset context
AVG_FRAUD_AMOUNT = 122.21   # mean Amount for fraud transactions (EUR)
PARTIAL_LOSS_RATE = 0.30    # fraction of fraud amount lost even when partially blocked


def compute_end_states(model_name):
    """
    Compute probability and expected loss for each end state.

    End states:
    1. Fraud Prevented         — detected, correct, response effective
    2. Partial Loss            — detected, correct, response too slow
    3. Operational Loss (FP)   — detected, incorrect classification (false alarm)
    4. Full Financial Loss     — not detected (missed fraud)
    """
    p = MODEL_PARAMS[model_name]
    pd_ = p["P_detect"]
    pc  = p["P_correct"]
    pr  = p["P_response"]

    # Path probabilities
    p_prevented     = pd_ * pc * pr
    p_partial       = pd_ * pc * (1 - pr)
    p_operational   = pd_ * (1 - pc)
    p_missed        = 1 - pd_

    # Sanity check
    total = p_prevented + p_partial + p_operational + p_missed
    assert abs(total - 1.0) < 1e-6, f"Probabilities don't sum to 1: {total}"

    # Expected losses per end state
    loss_prevented   = 0
    loss_partial     = AVG_FRAUD_AMOUNT * PARTIAL_LOSS_RATE
    loss_operational = AVG_FRAUD_AMOUNT * 0.10   # 10% friction cost for false alarm
    loss_missed      = AVG_FRAUD_AMOUNT

    return {
        "Fraud Prevented":       {"prob": round(p_prevented, 4),   "loss": round(loss_prevented, 2),   "color": "#2D8653"},
        "Partial Loss":          {"prob": round(p_partial, 4),     "loss": round(loss_partial, 2),     "color": "#F5A623"},
        "Operational Loss (FP)": {"prob": round(p_operational, 4), "loss": round(loss_operational, 2), "color": "#E8903A"},
        "Full Financial Loss":   {"prob": round(p_missed, 4),      "loss": round(loss_missed, 2),      "color": "#D94F3D"},
    }


def draw_event_tree(model_name, end_states, ax):
    """
    Draw the event tree for one model on the given axes.
    """
    p = MODEL_PARAMS[model_name]
    pd_ = p["P_detect"]
    pc  = p["P_correct"]
    pr  = p["P_response"]

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor("#F9FBFD")

    def draw_node(x, y, label, color="#1A6B9A", radius=0.22):
        circle = plt.Circle((x, y), radius, color=color, zorder=4)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=7, color="white", fontweight="bold", zorder=5)

    def draw_line(x1, y1, x2, y2, color="#555555"):
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.5, zorder=2)

    def end_label(x, y, state, prob, loss, color):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.05, y - 0.28), 3.2, 0.56,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="white",
            linewidth=1, alpha=0.85, zorder=3
        ))
        ax.text(x + 1.55, y + 0.08,
                f"{state}",
                ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white", zorder=5)
        ax.text(x + 1.55, y - 0.14,
                f"p={prob:.4f}  |  Loss=${loss:.2f}",
                ha="center", va="center",
                fontsize=6.5, color="white", zorder=5)

    # ── Initiating event ──
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.1, 4.6), 1.6, 0.8,
        boxstyle="round,pad=0.05",
        facecolor="#0D1B2A", edgecolor="white", linewidth=1.5, zorder=3
    ))
    ax.text(0.9, 5.0, "Fraud Tx\nSubmitted",
            ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="white", zorder=5)

    # ── Branch 1: Detection ──
    draw_line(1.7, 5.0, 2.8, 5.0)
    ax.text(2.25, 5.2, "Detection", ha="center", fontsize=7, color="#333")
    draw_node(2.8, 5.0, "D")

    # YES branch (detected)
    draw_line(2.8, 5.0, 4.2, 7.2)
    ax.text(3.35, 6.35, f"YES\np={pd_:.3f}", ha="center", fontsize=6.5, color="#2D8653", fontweight="bold")

    # NO branch (missed)
    draw_line(2.8, 5.0, 4.2, 2.8)
    ax.text(3.35, 3.75, f"NO\np={1-pd_:.3f}", ha="center", fontsize=6.5, color="#D94F3D", fontweight="bold")

    # End state: Full Financial Loss (missed)
    end_label(4.3, 2.8,
              "Full Financial Loss",
              end_states["Full Financial Loss"]["prob"],
              end_states["Full Financial Loss"]["loss"],
              end_states["Full Financial Loss"]["color"])

    # ── Branch 2: Correct Classification ──
    draw_node(4.2, 7.2, "C")
    ax.text(4.2, 7.65, "Classification", ha="center", fontsize=7, color="#333")

    # YES (TP)
    draw_line(4.2, 7.2, 5.8, 8.5)
    ax.text(4.85, 8.05, f"TP\np={pc:.3f}", ha="center", fontsize=6.5, color="#2D8653", fontweight="bold")

    # NO (FP)
    draw_line(4.2, 7.2, 5.8, 5.9)
    ax.text(4.85, 6.85, f"FP\np={1-pc:.3f}", ha="center", fontsize=6.5, color="#E8903A", fontweight="bold")

    # End state: Operational Loss (FP)
    end_label(5.9, 5.9,
              "Operational Loss (FP)",
              end_states["Operational Loss (FP)"]["prob"],
              end_states["Operational Loss (FP)"]["loss"],
              end_states["Operational Loss (FP)"]["color"])

    # ── Branch 3: Response Effectiveness ──
    draw_node(5.8, 8.5, "R")
    ax.text(5.8, 8.95, "Response", ha="center", fontsize=7, color="#333")

    # YES (blocked in time)
    draw_line(5.8, 8.5, 7.0, 9.3)
    ax.text(6.28, 9.1, f"Effective\np={pr:.2f}", ha="center", fontsize=6.5, color="#2D8653", fontweight="bold")

    # NO (too slow)
    draw_line(5.8, 8.5, 7.0, 7.7)
    ax.text(6.28, 7.9, f"Delayed\np={1-pr:.2f}", ha="center", fontsize=6.5, color="#F5A623", fontweight="bold")

    # End states
    end_label(7.05, 9.3,
              "Fraud Prevented",
              end_states["Fraud Prevented"]["prob"],
              end_states["Fraud Prevented"]["loss"],
              end_states["Fraud Prevented"]["color"])

    end_label(7.05, 7.7,
              "Partial Loss",
              end_states["Partial Loss"]["prob"],
              end_states["Partial Loss"]["loss"],
              end_states["Partial Loss"]["color"])

    ax.set_title(
        f"{model_name.replace('_', ' ')}\n"
        f"Recall={pd_:.3f}  Precision={pc:.4f}  Response={pr:.2f}",
        fontsize=10, fontweight="bold",
        color={"Advanced_ML": "#1A6B9A", "Standard_ML": "#F5A623", "Rule_Based": "#D94F3D"}[model_name]
    )


def plot_event_trees(all_end_states):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.patch.set_facecolor("#F4F8FC")

    for ax, (model, end_states) in zip(axes, all_end_states.items()):
        draw_event_tree(model, end_states, ax)

    fig.suptitle(
        "Event Tree Analysis — Fraud Transaction Submitted (Initiating Event)\n"
        "D = Detection Node   C = Classification Node   R = Response Node",
        fontsize=13, fontweight="bold", color="#0D1B2A", y=1.01
    )
    plt.tight_layout()
    plt.savefig("results/figures/event_tree.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: results/figures/event_tree.png")


def build_summary_table(all_end_states):
    rows = []
    for model, end_states in all_end_states.items():
        expected_loss = sum(
            s["prob"] * s["loss"] for s in end_states.values()
        )
        for state, data in end_states.items():
            rows.append({
                "Model":       model,
                "End State":   state,
                "Probability": data["prob"],
                "Loss ($)":    data["loss"],
                "EL Contrib":  round(data["prob"] * data["loss"], 4),
            })
        rows.append({
            "Model":       model,
            "End State":   "--- EXPECTED LOSS PER TX ---",
            "Probability": "",
            "Loss ($)":    "",
            "EL Contrib":  round(expected_loss, 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/event_tree_analysis.csv", index=False)
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("EVENT TREE ANALYSIS")
    print("=" * 60)
    print(f"\nInitiating Event: Fraud Transaction Submitted")
    print(f"Average Fraud Transaction Amount: ${AVG_FRAUD_AMOUNT}")
    print(f"Partial Loss Rate: {PARTIAL_LOSS_RATE*100}%")

    all_end_states = {}
    for model in MODEL_PARAMS:
        all_end_states[model] = compute_end_states(model)

    print("\n--- END STATE PROBABILITIES & LOSSES ---")
    for model, end_states in all_end_states.items():
        print(f"\n  {model}")
        expected_loss = 0
        for state, data in end_states.items():
            el = data["prob"] * data["loss"]
            expected_loss += el
            print(f"    {state:<25}  p={data['prob']:.4f}  Loss=${data['loss']:>6.2f}  EL=${el:.4f}")
        print(f"    {'Expected Loss per Transaction':<25}  EL=${expected_loss:.4f}")

    df = build_summary_table(all_end_states)
    df.to_csv("results/tables/event_tree_analysis.csv", index=False)
    print("\n  Saved: results/tables/event_tree_analysis.csv")

    plot_event_trees(all_end_states)

# src/risk_analysis/fault_tree.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

BASIC_EVENTS = {
    "Feature Distribution Shift": 0.15,
    "Concept Drift":               0.20,
    "GPU Failure":                 0.05,
    "Memory Overflow":             0.08,
    "API Timeout":                 0.10,
    "Dependency Failure":          0.07,
}

def and_gate(*probs):
    result = 1.0
    for p in probs:
        result *= p
    return result

def or_gate(*probs):
    result = 1.0
    for p in probs:
        result *= (1 - p)
    return 1 - result

def compute_fault_tree():
    be = BASIC_EVENTS
    p_model_failure = and_gate(be["Feature Distribution Shift"], be["Concept Drift"])
    p_infra_failure = and_gate(be["GPU Failure"], be["Memory Overflow"],
                                be["API Timeout"], be["Dependency Failure"])
    p_top = or_gate(p_model_failure, p_infra_failure)
    gates = {
        "Model Failure (AND)":          round(p_model_failure, 6),
        "Infrastructure Failure (AND)": round(p_infra_failure, 6),
        "ML System Failure (OR)":       round(p_top, 6),
    }
    return gates, p_top

def compute_birnbaum_importance(p_top):
    be = BASIC_EVENTS
    importance = {}
    for event in be:
        be_high = dict(be); be_high[event] = 1.0
        be_low  = dict(be); be_low[event]  = 0.0
        def compute_top(b):
            pm = and_gate(b["Feature Distribution Shift"], b["Concept Drift"])
            pi = and_gate(b["GPU Failure"], b["Memory Overflow"],
                         b["API Timeout"], b["Dependency Failure"])
            return or_gate(pm, pi)
        importance[event] = round(compute_top(be_high) - compute_top(be_low), 6)
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

def draw_event_box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=9):
    box = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02",
        linewidth=1.5, edgecolor=edgecolor,
        facecolor=facecolor, zorder=4
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color="#1a1a1a", zorder=5, multialignment="center")

def draw_or_gate(ax, x, y, size=0.38):
    theta = np.linspace(np.pi, 0, 80)
    top_x = x + size * np.cos(theta) * 0.9
    top_y = y + size * np.sin(theta) * 0.55
    theta2 = np.linspace(np.pi, 0, 80)
    bot_x  = x + size * 1.1 * np.cos(theta2)
    bot_y  = y - size * 0.18 + size * 0.55 * np.abs(np.sin(theta2)) * 0.5
    xs = np.concatenate([top_x, bot_x[::-1]])
    ys = np.concatenate([top_y, bot_y[::-1]])
    gate = plt.Polygon(list(zip(xs, ys)), closed=True,
                       facecolor="#FFF9C4", edgecolor="#B8860B",
                       linewidth=1.8, zorder=5)
    ax.add_patch(gate)
    ax.text(x, y + 0.04, "OR", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#7B6000", zorder=6)

def draw_and_gate(ax, x, y, size=0.38):
    theta = np.linspace(0, np.pi, 80)
    arc_x = x + size * np.cos(theta)
    arc_y = y - size * 0.4 + size * 1.1 * np.sin(theta)
    xs = np.concatenate([[x - size], arc_x, [x + size]])
    ys = np.concatenate([[y - size * 0.4], arc_y, [y - size * 0.4]])
    gate = plt.Polygon(list(zip(xs, ys)), closed=True,
                       facecolor="#E8F5E9", edgecolor="#2D8653",
                       linewidth=1.8, zorder=5)
    ax.add_patch(gate)
    ax.text(x, y + 0.05, "AND", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#1B5E20", zorder=6)

def draw_line(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color="#444444", linewidth=1.6, zorder=2)

def plot_fault_tree(gates, p_top):
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_facecolor("#F9FBFD")
    fig.patch.set_facecolor("#F9FBFD")

    # Top event
    draw_event_box(ax, 8.0, 9.8, 3.0, 0.75,
                   f"ML System Failure\nP = {gates['ML System Failure (OR)']:.4f}",
                   "#FFCDD2", "#D94F3D", fontsize=10)
    draw_line(ax, 8.0, 9.43, 8.0, 8.9)
    draw_or_gate(ax, 8.0, 8.55)
    draw_line(ax, 8.0, 8.18, 8.0, 8.0)
    draw_line(ax, 3.5, 8.0, 12.5, 8.0)

    # Left: Model Failure
    draw_line(ax, 3.5, 8.0, 3.5, 7.58)
    draw_event_box(ax, 3.5, 7.2, 3.0, 0.7,
                   f"Model Failure\nP = {gates['Model Failure (AND)']:.4f}",
                   "#FFF9C4", "#E6A817", fontsize=9)
    draw_line(ax, 3.5, 6.85, 3.5, 6.38)
    draw_and_gate(ax, 3.5, 6.0)
    draw_line(ax, 3.5, 5.62, 3.5, 5.5)
    draw_line(ax, 1.8, 5.5, 5.2, 5.5)
    draw_line(ax, 1.8, 5.5, 1.8, 4.9)
    draw_line(ax, 5.2, 5.5, 5.2, 4.9)
    draw_event_box(ax, 1.8, 4.35, 2.8, 0.95,
                   "Feature Distribution\nShift\nP = 0.15",
                   "#E8F5E9", "#2D8653", fontsize=8.5)
    draw_event_box(ax, 5.2, 4.35, 2.4, 0.95,
                   "Concept Drift\nP = 0.20",
                   "#E8F5E9", "#2D8653", fontsize=8.5)

    # Right: Infra Failure
    draw_line(ax, 12.5, 8.0, 12.5, 7.58)
    draw_event_box(ax, 12.5, 7.2, 3.2, 0.7,
                   f"Infra Failure\nP = {gates['Infrastructure Failure (AND)']:.6f}",
                   "#FFF9C4", "#E6A817", fontsize=9)
    draw_line(ax, 12.5, 6.85, 12.5, 6.38)
    draw_and_gate(ax, 12.5, 6.0)
    draw_line(ax, 12.5, 5.62, 12.5, 5.5)
    draw_line(ax, 9.2, 5.5, 15.8, 5.5)
    infra_xs = [9.2, 11.1, 13.0, 14.9]
    for ix in infra_xs:
        draw_line(ax, ix, 5.5, ix, 4.9)
    infra_events = [
        ("GPU Failure\nP = 0.05", 9.2),
        ("Memory Overflow\nP = 0.08", 11.1),
        ("API Timeout\nP = 0.10", 13.0),
        ("Dependency\nFailure\nP = 0.07", 14.9),
    ]
    for label, ix in infra_events:
        draw_event_box(ax, ix, 4.2, 1.6, 1.2,
                       label, "#E8F5E9", "#2D8653", fontsize=8)

    legend_patches = [
        mpatches.Patch(facecolor="#FFCDD2", edgecolor="#D94F3D", label="Top Event"),
        mpatches.Patch(facecolor="#FFF9C4", edgecolor="#E6A817", label="Intermediate Event"),
        mpatches.Patch(facecolor="#E8F5E9", edgecolor="#2D8653", label="Basic Event"),
        mpatches.Patch(facecolor="#FFF9C4", edgecolor="#B8860B", label="OR Gate"),
        mpatches.Patch(facecolor="#E8F5E9", edgecolor="#2D8653", label="AND Gate"),
    ]
    ax.legend(handles=legend_patches, loc="lower left",
              fontsize=9, framealpha=0.9, ncol=5,
              bbox_to_anchor=(0.0, 0.0))

    ax.set_title(
        "Fault Tree Analysis — ML Fraud Detection System Failure\n"
        "OR gate: system fails if ANY path occurs  |  AND gate: fails only if ALL events occur simultaneously",
        fontsize=13, fontweight="bold", color="#0D1B2A", pad=15
    )
    plt.tight_layout()
    plt.savefig("results/figures/fault_tree.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: fault_tree.png")

if __name__ == "__main__":
    print("=" * 60)
    print("FAULT TREE ANALYSIS")
    print("=" * 60)
    gates, p_top = compute_fault_tree()
    print("\n--- GATE PROBABILITIES ---")
    for gate, prob in gates.items():
        print(f"  {gate:<35} P = {prob:.6f}")
    print("\n--- BASIC EVENT PROBABILITIES ---")
    for event, prob in BASIC_EVENTS.items():
        print(f"  {event:<35} P = {prob}")
    importance = compute_birnbaum_importance(p_top)
    print("\n--- BIRNBAUM IMPORTANCE (ranked) ---")
    for event, imp in importance.items():
        print(f"  {event:<35} I = {imp:.6f}")
    print("\n--- MINIMAL CUT SETS ---")
    print("  {Feature Distribution Shift, Concept Drift}  -> Model Failure")
    print("  {GPU Failure, Memory Overflow, API Timeout, Dependency Failure} -> Infra Failure")
    plot_fault_tree(gates, p_top)
    rows = []
    for event, prob in BASIC_EVENTS.items():
        rows.append({"Component": event, "Probability": prob,
                     "Birnbaum Importance": importance.get(event, 0)})
    for gate, prob in gates.items():
        rows.append({"Component": gate, "Probability": prob, "Birnbaum Importance": ""})
    pd.DataFrame(rows).to_csv("results/tables/fault_tree_analysis.csv", index=False)
    print("  Saved: fault_tree_analysis.csv")

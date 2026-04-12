# src/visualization/influence_diagram.py
# Influence Diagram for the fraud detection deployment decision

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

os.makedirs("results/figures", exist_ok=True)

def draw_influence_diagram():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor("#F4F8FC")
    fig.patch.set_facecolor("#F4F8FC")

    # ── Node definitions ──────────────────────────────────────────
    # Format: (x, y, label, shape, color, textcolor)
    nodes = {
        # Decision node (rectangle)
        "decision": (3.5, 4.5, "Deployment\nStrategy\nDecision", "rect", "#1A6B9A", "white"),

        # Chance nodes (ovals)
        "risk_env":   (7.0, 7.5, "Risk\nEnvironment\n(High/Mod/Low)", "oval", "#F5A623", "#0D1B2A"),
        "model_perf": (7.0, 1.5, "Model\nPerformance\n(TP/FP/FN/TN)", "oval", "#F5A623", "#0D1B2A"),
        "fraud_rate": (1.5, 7.5, "Annual\nFraud Rate\n(0.173%)", "oval", "#F5A623", "#0D1B2A"),
        "deploy_cost":  (1.5, 1.5, "Deployment\nCost\n($0–$800K)", "oval", "#F5A623", "#0D1B2A"),

        # Intermediate value nodes (rounded rect)
        "sle":  (7.0, 4.5, "Single Loss\nExpectancy\n(SLE)", "round", "#2D8653", "white"),

        # Value node (diamond)
        "value": (11.0, 4.5, "Total\nAnnual\nCost", "diamond", "#D94F3D", "white"),
    }

    def draw_node(ax, x, y, label, shape, color, textcolor, w=1.9, h=1.1):
        if shape == "rect":
            box = mpatches.FancyBboxPatch(
                (x - w/2, y - h/2), w, h,
                boxstyle="square,pad=0.05",
                linewidth=2.5, edgecolor="white",
                facecolor=color, zorder=3
            )
            ax.add_patch(box)
        elif shape == "oval":
            ellipse = mpatches.Ellipse(
                (x, y), w + 0.3, h + 0.2,
                linewidth=2, edgecolor="white",
                facecolor=color, zorder=3
            )
            ax.add_patch(ellipse)
        elif shape == "round":
            box = mpatches.FancyBboxPatch(
                (x - w/2, y - h/2), w, h,
                boxstyle="round,pad=0.12",
                linewidth=2, edgecolor="white",
                facecolor=color, zorder=3
            )
            ax.add_patch(box)
        elif shape == "diamond":
            dx, dy = w/2 + 0.1, h/2 + 0.2
            diamond = plt.Polygon(
                [[x, y + dy], [x + dx, y], [x, y - dy], [x - dx, y]],
                closed=True, linewidth=2.5,
                edgecolor="white", facecolor=color, zorder=3
            )
            ax.add_patch(diamond)

        ax.text(x, y, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold",
                color=textcolor, zorder=4,
                multialignment="center")

    # Draw all nodes
    for key, (x, y, label, shape, color, textcolor) in nodes.items():
        draw_node(ax, x, y, label, shape, color, textcolor)

    # ── Arrows ────────────────────────────────────────────────────
    arrows = [
        # Decision → SLE
        ("decision", "sle"),
        # Risk environment → SLE
        ("risk_env", "sle"),
        # Model performance → SLE
        ("model_perf", "sle"),
        # Fraud rate → risk_env
        ("fraud_rate", "risk_env"),
        # Deployment cost → value
        ("deploy_cost", "value"),
        # SLE → Value
        ("sle", "value"),
        # Decision → value (deployment cost channel)
        ("decision", "value"),
    ]

    def get_edge(x1, y1, x2, y2, margin=0.55):
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        ux, uy = dx/dist, dy/dist
        return (x1 + ux*margin, y1 + uy*margin,
                x2 - ux*margin, y2 - uy*margin)

    for src, dst in arrows:
        x1, y1 = nodes[src][0], nodes[src][1]
        x2, y2 = nodes[dst][0], nodes[dst][1]
        sx, sy, ex, ey = get_edge(x1, y1, x2, y2)
        ax.annotate("",
            xy=(ex, ey), xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#0D1B2A",
                lw=1.8,
                mutation_scale=18,
            ), zorder=2
        )

    # ── Legend ────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor="#1A6B9A", edgecolor="white", label="Decision Node (Rectangle)"),
        mpatches.Patch(facecolor="#F5A623", edgecolor="white", label="Chance Node (Oval)"),
        mpatches.Patch(facecolor="#2D8653", edgecolor="white", label="Intermediate Value (Rounded Rect)"),
        mpatches.Patch(facecolor="#D94F3D", edgecolor="white", label="Value Node (Diamond)"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              fontsize=9, framealpha=0.9, frameon=True)

    ax.set_title(
        "Influence Diagram — Fraud Detection Deployment Decision\n"
        "Arrows represent informational and probabilistic dependencies",
        fontsize=13, fontweight="bold", color="#0D1B2A", pad=15
    )

    plt.tight_layout()
    plt.savefig("results/figures/influence_diagram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: results/figures/influence_diagram.png")


if __name__ == "__main__":
    print("=" * 60)
    print("INFLUENCE DIAGRAM")
    print("=" * 60)
    draw_influence_diagram()
    print("\nNodes:")
    print("  Decision node:   Deployment Strategy Decision")
    print("  Chance nodes:    Risk Environment, Model Performance,")
    print("                   Annual Fraud Rate, Deployment Cost")
    print("  Intermediate:    Single Loss Expectancy (SLE)")
    print("  Value node:      Total Annual Cost")
    print("\nKey relationships:")
    print("  Decision + Risk Environment + Model Performance → SLE")
    print("  SLE + Deployment Cost → Total Annual Cost")

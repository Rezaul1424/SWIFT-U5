#!/usr/bin/env python3
"""
experiments/generate_workflow_diagram.py

Generates a clean, minimalistic, publication-ready workflow diagram
(`workflow_diagram.png`) with no background fills and concise descriptions.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def generate_workflow_diagram(output_paths: list[Path]) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    })

    steps = [
        ("1. Multi-Wave Cohort & Leakage Control", "BDHS 2004–2022 (0–59 mo); 2022 labels quarantined"),
        ("2. PSU-Grouped Feature Stability Selection", "Weighted LASSO & XGBoost permutation"),
        ("3. Module A: Recency & Covariate-Shift Weighting", "Survey × recency decay × clipped density ratio"),
        ("4. Survey-Aware Base Learner", "Survey-weighted XGBoost (2004–2014 dev)"),
        ("5. Module B: Division-Level EB Shrinkage", "Division parameters shrunk toward national prior"),
        ("6. Module C: Drift-Aware Recalibration", "Platt scaling on held-out 2017–18 validation"),
        ("7. Policy Translation & Equity Audit", "Capacity/yield prioritization & subgroup audit (2022 test)"),
    ]

    fig_w, fig_h = 8.5, 6.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    # Minimal header
    ax.text(
        5.0, 8.82,
        "SWIFT-U5 System Workflow",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color="#111827",
    )

    box_w = 8.2
    box_h = 0.74
    x_left = (10.0 - box_w) / 2.0
    y_top = 8.15
    y_step = 1.04

    for idx, (title, desc) in enumerate(steps):
        y = y_top - idx * y_step
        rect = FancyBboxPatch(
            (x_left, y - box_h),
            box_w,
            box_h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor="none",
            edgecolor="#1F2937",
            linewidth=1.15,
        )
        ax.add_patch(rect)

        ax.text(
            5.0, y - 0.25,
            title,
            ha="center", va="center",
            fontsize=10.0, fontweight="bold", color="#111827",
        )
        ax.text(
            5.0, y - 0.53,
            desc,
            ha="center", va="center",
            fontsize=8.3, color="#4B5563",
        )

        # Downward connector arrow to next box
        if idx < len(steps) - 1:
            y_arrow_start = y - box_h - 0.04
            y_arrow_end = y - y_step + 0.04
            ax.annotate(
                "",
                xy=(5.0, y_arrow_end),
                xytext=(5.0, y_arrow_start),
                arrowprops=dict(
                    arrowstyle="-|>,head_width=0.22,head_length=0.35",
                    color="#374151",
                    lw=1.1,
                    shrinkA=0,
                    shrinkB=0,
                ),
            )

    # Minimal footer note (no heavy box/fill)
    ax.text(
        5.0, 0.35,
        "Track A: 2004–2014 Development  →  2017–18 Validation  →  Locked 2022 Test",
        ha="center", va="center",
        fontsize=8.3, style="italic", color="#4B5563",
    )

    plt.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.02)

    for out_path in output_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.12,
            facecolor="white",
            transparent=False,
        )
        print(f"Saved {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    generate_workflow_diagram([
        root / "docs" / "figures" / "workflow_diagram.png",
        root / "outputs" / "figures" / "workflow_diagram.png",
    ])

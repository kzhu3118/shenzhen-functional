#!/usr/bin/env python3
"""Plot descriptive LST patterns by functional-ecological land-use type.

Panel (a): mean MODIS summer LST by land-use type from 2000 to 2020.
Panel (b): inter-class LST contrasts, highlighting thermal gradients.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
THERMAL_DIR = PROJECT / "data/thermal_joint_mlp_5class_modis"
OUT_DIR = PROJECT / "figures/part33_thermal_descriptive"

TABLE1 = THERMAL_DIR / "Table1_joint_mlp_5class_lst_by_year.csv"
TABLE2 = THERMAL_DIR / "Table2_joint_mlp_5class_pairwise_contrasts.csv"

CLASS_LABELS = {
    "production": "Production land",
    "living": "Living land",
    "green_space": "Green space",
    "water": "Water bodies",
    "farmland": "Farmland",
}

COLORS = {
    "production": "#D55E00",
    "living": "#0072B2",
    "green_space": "#009E73",
    "water": "#56B4E9",
    "farmland": "#E69F00",
}

CONTRAST_LABELS = {
    "production_minus_green_space": "Production - green space",
    "living_minus_green_space": "Living - green space",
    "production_minus_water": "Production - water bodies",
    "living_minus_water": "Living - water bodies",
    "production_minus_living": "Production - living",
    "farmland_minus_green_space": "Farmland - green space",
}

CONTRAST_ORDER = [
    "production_minus_green_space",
    "living_minus_green_space",
    "production_minus_water",
    "living_minus_water",
    "farmland_minus_green_space",
    "production_minus_living",
]

CONTRAST_COLORS = {
    "production_minus_green_space": "#D55E00",
    "living_minus_green_space": "#0072B2",
    "production_minus_water": "#CC79A7",
    "living_minus_water": "#56B4E9",
    "farmland_minus_green_space": "#E69F00",
    "production_minus_living": "#333333",
}


def read_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    t1 = pd.read_csv(TABLE1)
    t2 = pd.read_csv(TABLE2)
    return t1, t2


def plot_panel_a(ax: plt.Axes, t1: pd.DataFrame) -> None:
    for cls in ["production", "living", "farmland", "water", "green_space"]:
        sub = t1[t1["class"] == cls].sort_values("year")
        if sub.empty:
            continue
        ax.plot(
            sub["year"],
            sub["LST_mean"],
            marker="o",
            lw=2.2,
            color=COLORS[cls],
            label=CLASS_LABELS[cls],
        )
        ax.fill_between(
            sub["year"].to_numpy(),
            sub["LST_ci_lo"].to_numpy(),
            sub["LST_ci_hi"].to_numpy(),
            color=COLORS[cls],
            alpha=0.14,
            linewidth=0,
        )

    ax.set_title("(a) Mean summer LST by land-use type", loc="left", fontweight="bold")
    ax.set_ylabel("MODIS LST (°C)")
    ax.set_xlabel("Year")
    ax.set_xticks([2000, 2005, 2010, 2015, 2020])
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, ncol=1, loc="upper left")


def plot_panel_b(ax: plt.Axes, t2: pd.DataFrame) -> None:
    for contrast in CONTRAST_ORDER:
        sub = t2[t2["contrast"] == contrast].sort_values("year")
        if sub.empty:
            continue
        ax.plot(
            sub["year"],
            sub["dLST_mean"],
            marker="o",
            lw=2.1,
            color=CONTRAST_COLORS[contrast],
            label=CONTRAST_LABELS[contrast],
        )
        ax.fill_between(
            sub["year"].to_numpy(),
            sub["dLST_ci_lo"].to_numpy(),
            sub["dLST_ci_hi"].to_numpy(),
            color=CONTRAST_COLORS[contrast],
            alpha=0.12,
            linewidth=0,
        )

    ax.axhline(0, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_title("(b) Inter-class thermal gradients", loc="left", fontweight="bold")
    ax.set_ylabel("LST difference (°C)")
    ax.set_xlabel("Year")
    ax.set_xticks([2000, 2005, 2010, 2015, 2020])
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, fontsize=9, loc="upper left")


def export_compact_tables(t1: pd.DataFrame, t2: pd.DataFrame) -> None:
    lst_summary = t1.copy()
    lst_summary["class_label"] = lst_summary["class"].map(CLASS_LABELS)
    lst_summary = lst_summary[
        [
            "year",
            "class",
            "class_label",
            "n_grids",
            "LST_mean",
            "LST_ci_lo",
            "LST_ci_hi",
            "LST_median",
            "LST_q25",
            "LST_q75",
        ]
    ]
    lst_summary.to_csv(OUT_DIR / "Figure_thermal_descriptive_panel_a_data.csv", index=False, encoding="utf-8-sig")

    contrast_summary = t2[t2["contrast"].isin(CONTRAST_ORDER)].copy()
    contrast_summary["contrast_label"] = contrast_summary["contrast"].map(CONTRAST_LABELS)
    contrast_summary.to_csv(OUT_DIR / "Figure_thermal_descriptive_panel_b_data.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t1, t2 = read_tables()
    export_compact_tables(t1, t2)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    plot_panel_a(axes[0], t1)
    plot_panel_b(axes[1], t2)

    fig.savefig(OUT_DIR / "Figure_3_3_thermal_descriptive.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure_3_3_thermal_descriptive.pdf", bbox_inches="tight")
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()

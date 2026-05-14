#!/usr/bin/env python3
"""Plot transition-related thermal effects for joint MLP five-class results.

Panels:
  (a) sample sizes for key transition types
  (b) stage-level summary of ecological-to-built transition effects
  (c) background-normalized dLST forest plot
  (d) PSM ATT forest plot
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
THERMAL_DIR = PROJECT / "data/thermal_joint_mlp_5class_modis"
OUT_DIR = PROJECT / "figures/part33_transition_effects"

TABLE3 = THERMAL_DIR / "Table3_joint_mlp_5class_conversion_dLST.csv"
TABLE4 = THERMAL_DIR / "Table4_joint_mlp_5class_background_normalized_dLST.csv"
TABLE5 = THERMAL_DIR / "Table5_joint_mlp_5class_key_transition_psm.csv"

STAGE_ORDER = ["2000_2005", "2005_2010", "2010_2015", "2015_2020"]
STAGE_LABELS = {
    "2000_2005": "2000-2005",
    "2005_2010": "2005-2010",
    "2010_2015": "2010-2015",
    "2015_2020": "2015-2020",
}

KEY_CONVERSIONS = [
    "green_space_to_living",
    "green_space_to_production",
    "farmland_to_living",
    "farmland_to_production",
    "water_to_living",
    "water_to_production",
    "living_to_production",
    "production_to_living",
]

CONVERSION_LABELS = {
    "green_space_to_living": "Green Space -> Living",
    "green_space_to_production": "Green Space -> Production",
    "farmland_to_living": "Farmland -> Living",
    "farmland_to_production": "Farmland -> Production",
    "water_to_living": "Water Bodies -> Living",
    "water_to_production": "Water Bodies -> Production",
    "living_to_production": "Living -> Production",
    "production_to_living": "Production -> Living",
}

ECO_TO_BUILT = [
    "green_space_to_living",
    "green_space_to_production",
    "farmland_to_living",
    "farmland_to_production",
    "water_to_living",
    "water_to_production",
]

STAGE_COLORS = {
    "2000_2005": "#b2182b",
    "2005_2010": "#ef8a62",
    "2010_2015": "#67a9cf",
    "2015_2020": "#2166ac",
}


def read_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conv = pd.read_csv(TABLE3)
    bg = pd.read_csv(TABLE4)
    psm = pd.read_csv(TABLE5)
    return conv, bg, psm


def key_sample_matrix(conv: pd.DataFrame) -> pd.DataFrame:
    sub = conv[conv["conversion"].isin(KEY_CONVERSIONS)].copy()
    mat = sub.pivot_table(index="conversion", columns="stage", values="n_grids", aggfunc="sum", fill_value=0)
    mat = mat.reindex(KEY_CONVERSIONS)[STAGE_ORDER]
    return mat


def plot_sample_heatmap(ax: plt.Axes, mat: pd.DataFrame) -> None:
    data = mat.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels([STAGE_LABELS[s] for s in STAGE_ORDER], rotation=35, ha="right")
    ax.set_yticks(range(len(KEY_CONVERSIONS)))
    ax.set_yticklabels([CONVERSION_LABELS[c] for c in KEY_CONVERSIONS])
    ax.set_title("(a) Sample Sizes of Key Transitions", loc="left", fontweight="bold")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = int(data[i, j])
            if val > 0:
                ax.text(j, i, f"{val}", ha="center", va="center", fontsize=16, color="black")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Number of Grids")


def forest_subset_bg(bg: pd.DataFrame) -> pd.DataFrame:
    sub = bg[bg["conversion"].isin(KEY_CONVERSIONS)].copy()
    sub["conversion_label"] = sub["conversion"].map(CONVERSION_LABELS)
    sub["stage_label"] = sub["stage"].map(STAGE_LABELS)
    return sub


def forest_subset_psm(psm: pd.DataFrame) -> pd.DataFrame:
    sub = psm[psm["treated"].isin(KEY_CONVERSIONS) & (psm["status"] == "ok")].copy()
    sub["conversion"] = sub["treated"]
    sub["conversion_label"] = sub["conversion"].map(CONVERSION_LABELS)
    sub["stage_label"] = sub["stage"].map(STAGE_LABELS)
    return sub


def plot_forest(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    lo_col: str,
    hi_col: str,
    title: str,
    xlabel: str,
) -> None:
    rows = []
    for conv in KEY_CONVERSIONS:
        for stage in STAGE_ORDER:
            r = df[(df["conversion"] == conv) & (df["stage"] == stage)]
            if r.empty:
                continue
            rows.append(r.iloc[0])
    if not rows:
        ax.set_axis_off()
        return
    plot_df = pd.DataFrame(rows)
    y_base = np.arange(len(KEY_CONVERSIONS))
    offsets = {
        "2000_2005": -0.27,
        "2005_2010": -0.09,
        "2010_2015": 0.09,
        "2015_2020": 0.27,
    }
    for stage in STAGE_ORDER:
        s = plot_df[plot_df["stage"] == stage]
        if s.empty:
            continue
        y = np.array([KEY_CONVERSIONS.index(c) for c in s["conversion"]], dtype=float) + offsets[stage]
        x = s[value_col].astype(float).to_numpy()
        xerr = np.vstack([x - s[lo_col].astype(float).to_numpy(), s[hi_col].astype(float).to_numpy() - x])
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt="o",
            color=STAGE_COLORS[stage],
            ecolor=STAGE_COLORS[stage],
            elinewidth=1.4,
            capsize=2.5,
            markersize=4.5,
            label=STAGE_LABELS[stage],
        )
    ax.axvline(0, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_yticks(y_base)
    ax.set_yticklabels([CONVERSION_LABELS[c] for c in KEY_CONVERSIONS])
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", color="#dddddd", linewidth=0.7)


def stage_summary(bg: pd.DataFrame, psm: pd.DataFrame) -> pd.DataFrame:
    bg_sub = bg[bg["conversion"].isin(ECO_TO_BUILT)].copy()
    bg_stage = (
        bg_sub.groupby("stage")
        .apply(lambda x: np.average(x["attributed_dLST"], weights=x["n_grids"]))
        .rename("background_normalized_mean")
        .reset_index()
    )
    bg_stage["background_normalized_n"] = bg_sub.groupby("stage")["n_grids"].sum().reindex(bg_stage["stage"]).to_numpy()

    psm_sub = psm[(psm["treated"].isin(ECO_TO_BUILT)) & (psm["status"] == "ok")].copy()
    psm_sub["n_matched"] = pd.to_numeric(psm_sub["n_matched"], errors="coerce")
    psm_stage = (
        psm_sub.groupby("stage")
        .apply(lambda x: np.average(x["att_dLST"], weights=x["n_matched"]))
        .rename("psm_att_mean")
        .reset_index()
    )
    psm_stage["psm_n_matched"] = psm_sub.groupby("stage")["n_matched"].sum().reindex(psm_stage["stage"]).to_numpy()

    out = pd.DataFrame({"stage": STAGE_ORDER})
    out = out.merge(bg_stage, on="stage", how="left").merge(psm_stage, on="stage", how="left")
    out["stage_label"] = out["stage"].map(STAGE_LABELS)
    out["period"] = np.where(out["stage"].isin(["2000_2005", "2005_2010"]), "Early Expansion", "Later Transition")
    return out


def plot_stage_summary(ax: plt.Axes, summary: pd.DataFrame) -> None:
    x = np.arange(len(summary))
    ax.plot(
        x,
        summary["background_normalized_mean"],
        marker="o",
        lw=4.4,
        color="#b2182b",
        label="Background-normalized dLST",
    )
    ax.plot(
        x,
        summary["psm_att_mean"],
        marker="s",
        lw=4.4,
        color="#2166ac",
        label="PSM ATT",
    )
    ax.axhline(0, color="#555555", linestyle="--", linewidth=1.8)
    ax.axvspan(-0.4, 1.5, color="#b2182b", alpha=0.08, linewidth=0)
    ax.axvspan(1.5, 3.4, color="#2166ac", alpha=0.08, linewidth=0)
    y_top = max(
        float(summary["background_normalized_mean"].max(skipna=True)),
        float(summary["psm_att_mean"].max(skipna=True)),
    )
    ax.set_ylim(-0.05, y_top * 1.22)
    ax.text(0.5, y_top * 1.08, "Early Expansion", ha="center", va="center", color="#8c3a00")
    ax.text(2.5, y_top * 1.08, "Later Transition", ha="center", va="center", color="#00527f")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["stage_label"], rotation=35, ha="right")
    ax.set_ylabel("Mean effect (°C)")
    ax.set_title("(b) Stage-Level Effect of Ecological-to-Built Transitions", loc="left", fontweight="bold")
    ax.grid(True, axis="y", color="#dddddd", linewidth=1.4)
    ax.legend(frameon=False, loc="lower right")


def export_plot_data(mat: pd.DataFrame, bg_forest: pd.DataFrame, psm_forest: pd.DataFrame, summary: pd.DataFrame) -> None:
    mat.reset_index().rename(columns={"conversion": "transition"}).to_csv(
        OUT_DIR / "Figure_transition_panel_a_sample_sizes.csv", index=False, encoding="utf-8-sig"
    )
    bg_forest.to_csv(OUT_DIR / "Figure_transition_panel_b_background_normalized.csv", index=False, encoding="utf-8-sig")
    psm_forest.to_csv(OUT_DIR / "Figure_transition_panel_c_psm_att.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "Figure_transition_panel_d_stage_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conv, bg, psm = read_tables()

    mat = key_sample_matrix(conv)
    bg_forest = forest_subset_bg(bg)
    psm_forest = forest_subset_psm(psm)
    summary = stage_summary(bg, psm)
    export_plot_data(mat, bg_forest, psm_forest, summary)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 18,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "legend.fontsize": 17,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 0,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 18), constrained_layout=False)
    plot_sample_heatmap(axes[0, 0], mat)
    plot_stage_summary(axes[0, 1], summary)
    plot_forest(
        axes[1, 0],
        bg_forest,
        value_col="attributed_dLST",
        lo_col="attributed_ci_lo",
        hi_col="attributed_ci_hi",
        title="(c) Background-Normalized dLST",
        xlabel="Attributed dLST (°C)",
    )
    plot_forest(
        axes[1, 1],
        psm_forest,
        value_col="att_dLST",
        lo_col="att_ci_lo",
        hi_col="att_ci_hi",
        title="(d) PSM-Based Average Treatment Effect",
        xlabel="ATT of dLST (°C)",
    )

    handles, labels = axes[1, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.035))

    fig.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.18, wspace=0.28, hspace=0.38)
    fig.savefig(OUT_DIR / "Figure_3_3_transition_effects.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure_3_3_transition_effects.pdf", bbox_inches="tight")
    print(f"Saved outputs to: {OUT_DIR}")
    print("\nStage summary:")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

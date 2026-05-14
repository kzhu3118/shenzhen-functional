#!/usr/bin/env python3
"""Create manuscript Table/Figure for Results 3.1 model validation.

The figure has four panels:
  a) grouped variable importance/sensitivity for RF and MLP
  b) production-area trajectory compared with TCULU and official industrial area
  c) living-area trajectory compared with TCULU and official residential/commercial proxy
  d) built-up proxy trajectory compared with TCULU and official built-up area
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, f1_score


PROJECT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT / "data/multisource_labels_2020_5class_weight_sweep/tculuProd_1p5/model_test_mlp"
FEATURES_2020 = PROJECT / "data/GEE/shenzhen_features_2000_2020/shenzhen_features_2020.csv"
AREA_COMP_DIR = (
    PROJECT
    / "data/mlp_multiyear_5class_joint_tculuWeak_manual2005_gaia_thr10/area_comparison_tculu_official"
)
OUT_DIR = PROJECT / "figures/part31_model_validation"

CLASS_ORDER = ["production", "living", "green_space", "water", "farmland"]
MODEL_LABELS = {"rf": "RF", "mlp": "MLP"}
GROUP_ORDER = ["Landsat bands", "Spectral indices", "LST", "NTL", "Terrain"]
CLASS_TO_CODE = {cls: i for i, cls in enumerate(["farmland", "green_space", "living", "production", "water"])}

mpl.rcParams.update(
    {
        # Keep PDF text editable in Adobe Illustrator by embedding TrueType fonts
        # instead of converting text to Type 3 paths.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "axes.unicode_minus": False,
    }
)


def feature_group(name: str) -> str:
    if name.startswith("SR_B"):
        return "Landsat bands"
    if name.startswith(("NDVI", "NDBI", "MNDWI", "NBI")):
        return "Spectral indices"
    if name.startswith("LST"):
        return "LST"
    if name.startswith("NTL"):
        return "NTL"
    if name.startswith(("DEM", "slope")):
        return "Terrain"
    return "Other"


def load_feature_matrix() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    features = pd.read_csv(FEATURES_2020)
    heldout = pd.read_csv(MODEL_DIR / "heldout_predictions.csv")
    feature_cols = [c for c in features.columns if c != "grid_id"]
    df = heldout[["grid_id", "fused_label"]].merge(features, on="grid_id", how="inner")
    df = df.dropna(subset=feature_cols + ["fused_label"]).copy()
    y = df["fused_label"].map(CLASS_TO_CODE)
    df = df[y.notna()].copy()
    y = y[y.notna()].astype(int)
    return df[feature_cols], y, feature_cols


def build_performance_table() -> pd.DataFrame:
    rows = []
    reports = {}
    for model in ["rf", "mlp"]:
        rep = pd.read_csv(MODEL_DIR / f"classification_report_{model}.csv", index_col=0)
        reports[model] = rep

    for cls in CLASS_ORDER:
        row = {"class": cls}
        for model in ["rf", "mlp"]:
            rep = reports[model]
            if cls in rep.index:
                row[f"{model}_precision"] = rep.loc[cls, "precision"]
                row[f"{model}_recall"] = rep.loc[cls, "recall"]
                row[f"{model}_f1"] = rep.loc[cls, "f1-score"]
                row["support"] = rep.loc[cls, "support"]
        rows.append(row)

    for summary_row in ["accuracy", "macro avg", "weighted avg"]:
        row = {"class": summary_row}
        total_support = reports["rf"].loc["weighted avg", "support"] if "weighted avg" in reports["rf"].index else np.nan
        for model in ["rf", "mlp"]:
            rep = reports[model]
            if summary_row in rep.index:
                if summary_row == "accuracy":
                    row[f"{model}_precision"] = np.nan
                    row[f"{model}_recall"] = np.nan
                    row[f"{model}_f1"] = rep.loc[summary_row, "f1-score"]
                    row["support"] = total_support
                else:
                    row[f"{model}_precision"] = rep.loc[summary_row, "precision"]
                    row[f"{model}_recall"] = rep.loc[summary_row, "recall"]
                    row[f"{model}_f1"] = rep.loc[summary_row, "f1-score"]
                    row["support"] = rep.loc[summary_row, "support"]
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.round(3)


def build_importance_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    X, y, feature_cols = load_feature_matrix()
    X_np = X.to_numpy()
    parts = []

    rf_path = MODEL_DIR / "rf_multisource_2020.joblib"
    if rf_path.exists():
        rf = joblib.load(rf_path)
        parts.append(
            pd.DataFrame(
                {
                    "feature": feature_cols,
                    "model": "RF",
                    "importance": rf.feature_importances_,
                    "importance_type": "gini",
                }
            )
        )
    else:
        print(f"Skipping RF importance because the large model artifact is not included: {rf_path}")

    mlp_path = MODEL_DIR / "mlp_multisource_2020.joblib"
    if mlp_path.exists():
        mlp = joblib.load(mlp_path)
        perm = permutation_importance(
            mlp,
            X_np,
            y,
            scoring=make_scorer(f1_score, average="weighted", zero_division=0),
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )
        parts.append(
            pd.DataFrame(
                {
                    "feature": feature_cols,
                    "model": "MLP",
                    "importance": np.maximum(perm.importances_mean, 0),
                    "importance_std": perm.importances_std,
                    "importance_type": "permutation_f1_weighted",
                }
            )
        )
    else:
        print(f"Skipping MLP importance because the model artifact is missing: {mlp_path}")

    if not parts:
        raise FileNotFoundError("No model artifacts are available for importance calculation.")

    detail = pd.concat(parts, ignore_index=True)
    detail["group"] = detail["feature"].map(feature_group)

    grouped = detail.groupby(["model", "group"], as_index=False)["importance"].sum()
    grouped = grouped[grouped["group"].isin(GROUP_ORDER)].copy()
    grouped["importance_percent"] = grouped.groupby("model")["importance"].transform(lambda s: s / s.sum() * 100)
    return detail.sort_values(["model", "importance"], ascending=[True, False]), grouped


def load_area_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    five = pd.read_csv(AREA_COMP_DIR / "fiveclass_area_wide_joint_tculu.csv")
    built = pd.read_csv(AREA_COMP_DIR / "built_area_comparison_joint_tculu_official.csv")
    class_official = pd.read_csv(AREA_COMP_DIR / "production_living_vs_official.csv")
    return five, built, class_official


def plot_area_panel(ax: plt.Axes, data: pd.DataFrame, title: str, ylabel: str = "Area (km$^2$)") -> None:
    styles = {
        "joint_mlp": dict(color="#1f77b4", marker="o", lw=2.2, label="Joint MLP"),
        "joint_rf": dict(color="#ff7f0e", marker="s", lw=1.8, label="Joint RF"),
        "TCULU_5class": dict(color="#2ca02c", marker="^", lw=1.8, label="TCULU"),
        "official": dict(color="#222222", marker="D", lw=1.8, ls="--", label="Official/proxy"),
    }
    for col, style in styles.items():
        if col in data.columns and data[col].notna().any():
            ax.plot(data["year"], data[col], **style)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year")
    ax.set_xticks([2000, 2005, 2010, 2015, 2020])
    ax.grid(True, axis="y", color="#dddddd", lw=0.6)


def plot_figure(grouped: pd.DataFrame) -> None:
    five, built, class_official = load_area_tables()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.22, wspace=0.32, hspace=0.46)
    ax = axes[0, 0]

    pivot = (
        grouped.pivot_table(index="group", columns="model", values="importance_percent", aggfunc="sum")
        .reindex(GROUP_ORDER)
        .fillna(0)
    )
    y = np.arange(len(pivot.index))
    h = 0.36
    ax.barh(y - h / 2, pivot.get("RF", 0), height=h, color="#ff7f0e", label="RF")
    ax.barh(y + h / 2, pivot.get("MLP", 0), height=h, color="#1f77b4", label="MLP")
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.invert_yaxis()
    ax.set_xlabel("Grouped importance / sensitivity (%)")
    ax.set_title("(a) Variable-group importance", loc="left", fontweight="bold")
    ax.grid(True, axis="x", color="#dddddd", lw=0.6)
    ax.legend(frameon=False, loc="lower right")

    prod = five[five["class"] == "production"][["year", "joint_mlp", "joint_rf", "TCULU_5class"]].copy()
    official_prod = class_official[class_official["class"] == "production"][["year", "official_industrial"]]
    prod = prod.merge(official_prod, on="year", how="left").rename(columns={"official_industrial": "official"})
    plot_area_panel(axes[0, 1], prod, "(b) Production land")

    living = five[five["class"] == "living"][["year", "joint_mlp", "joint_rf", "TCULU_5class"]].copy()
    official_liv = class_official[class_official["class"] == "living"][
        ["year", "official_residential", "official_commercial"]
    ].copy()
    official_liv["official"] = official_liv["official_residential"].fillna(0) + official_liv["official_commercial"].fillna(0)
    official_liv.loc[official_liv["official"].eq(0), "official"] = np.nan
    living = living.merge(official_liv[["year", "official"]], on="year", how="left")
    plot_area_panel(axes[1, 0], living, "(c) Living land")

    built_plot = built[
        [
            "year",
            "joint_mlp_built_prod_living",
            "joint_rf_built_prod_living",
            "tculu_built_with_transport",
            "official_built_up",
        ]
    ].rename(
        columns={
            "joint_mlp_built_prod_living": "joint_mlp",
            "joint_rf_built_prod_living": "joint_rf",
            "tculu_built_with_transport": "TCULU_5class",
            "official_built_up": "official",
        }
    )
    plot_area_panel(axes[1, 1], built_plot, "(d) Built-up proxy")

    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.1))
    fig.savefig(OUT_DIR / "Figure_3_1_model_validation.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure_3_1_model_validation.pdf", bbox_inches="tight")
    # plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    perf = build_performance_table()
    perf.to_csv(OUT_DIR / "Table_3_1_RF_MLP_2020_classification_metrics.csv", index=False, encoding="utf-8-sig")
    perf.to_excel(OUT_DIR / "Table_3_1_RF_MLP_2020_classification_metrics.xlsx", index=False)

    detail, grouped = build_importance_tables()
    detail.to_csv(OUT_DIR / "feature_importance_detail_RF_MLP_2020.csv", index=False, encoding="utf-8-sig")
    grouped.to_csv(OUT_DIR / "feature_importance_grouped_RF_MLP_2020.csv", index=False, encoding="utf-8-sig")

    plot_figure(grouped)
    print("Saved outputs to:", OUT_DIR)
    print("\nTable 3.1:")
    print(perf.to_string(index=False))
    print("\nGrouped importance:")
    print(grouped.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Joint multi-year training and prediction for five-class land use.

Training sources:
  - 2020 multi-source strong labels (TCULU production weight = 1.5)
  - 2000/2005/2010/2015 TCULU historical weak labels
  - 2005 manual production/living samples

The model is trained on stacked year-specific feature rows with an added
`year_norm` feature, then predicts 2000-2020 for all grids. The MLP path uses
weighted resampling because sklearn MLPClassifier does not support sample
weights in many sklearn versions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data/GEE/shenzhen_features_2000_2020"
GRID_GPKG = PROJECT_ROOT / "data/poi_2020/poi_clean/grid_scored_2020_FD.gpkg"
TCULU_GRID_GPKG = PROJECT_ROOT / "data/tculu_shenzhen/validate_rf_ples_thr10/rf_ples_vs_tculu_grid_validation.gpkg"
STRONG_LABEL_CSV = (
    PROJECT_ROOT
    / "data/multisource_labels_2020_5class_weight_sweep/tculuProd_1p5/multisource_labels_2020_5class.csv"
)
MANUAL_GRID = PROJECT_ROOT / "data/manual_samples_2005/manual_2005_grid_labels.gpkg"
GAIA_GRID = PROJECT_ROOT / "data/GAIA_processed/ratio_thr10/grid_gaia_raw_masks_ratio_thr10.gpkg"
OUT_DIR = PROJECT_ROOT / "data/mlp_multiyear_5class_joint_tculuWeak_manual2005_gaia_thr10"

YEARS = [2000, 2005, 2010, 2015, 2020]
HISTORICAL_WEAK_YEARS = [2000, 2005, 2010, 2015]
CLASSES = ["farmland", "green_space", "living", "production", "water"]
BUILT_CLASSES = {"production", "living"}
NATURAL_CLASSES = ["green_space", "water", "farmland"]
RANDOM_STATE = 42

TCULU_TO_5CLASS = {
    "industrial": "production",
    "residential": "living",
    "commercial": "living",
    "institutional": "living",
    "green_space": "green_space",
    "water": "water",
    "farmland": "farmland",
    "bare_land": "farmland",
}

MANUAL_TO_5CLASS = {
    "production": "production",
    "living": "living",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--grid-gpkg", type=Path, default=GRID_GPKG)
    parser.add_argument("--grid-layer", default="grid_scored")
    parser.add_argument("--tculu-grid-gpkg", type=Path, default=TCULU_GRID_GPKG)
    parser.add_argument("--tculu-layer", default="rf_ples_vs_tculu")
    parser.add_argument("--strong-label-csv", type=Path, default=STRONG_LABEL_CSV)
    parser.add_argument("--manual-grid-gpkg", type=Path, default=MANUAL_GRID)
    parser.add_argument("--manual-grid-layer", default="manual_2005_grid_labels")
    parser.add_argument("--gaia-grid-gpkg", type=Path, default=GAIA_GRID)
    parser.add_argument("--gaia-grid-layer", default="grid_gaia_masks")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--years", default="2000,2005,2010,2015,2020")
    parser.add_argument("--historical-weak-years", default="2000,2005,2010,2015")
    parser.add_argument("--drop-ntl", action="store_true")
    parser.add_argument("--strong-min-confidence", type=float, default=0.40)
    parser.add_argument("--historical-min-dom-frac", type=float, default=0.55)
    parser.add_argument("--historical-min-valid-frac", type=float, default=0.20)
    parser.add_argument("--historical-weight", type=float, default=0.35)
    parser.add_argument("--historical-production-weight", type=float, default=0.60)
    parser.add_argument("--manual-weight", type=float, default=2.0)
    parser.add_argument("--mlp-hidden", default="128,64")
    parser.add_argument("--mlp-max-iter", type=int, default=500)
    parser.add_argument("--mlp-resample-multiplier", type=float, default=1.5)
    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument("--no-gaia-constraint", action="store_true")
    return parser.parse_args()


def parse_years(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_hidden(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def feature_path(feature_dir: Path, year: int) -> Path:
    if year == 2020:
        return feature_dir / "shenzhen_features_2020.csv"
    return feature_dir / f"shenzhen_features_{year}_noNTL.csv"


def common_feature_cols(feature_dir: Path, years: list[int], drop_ntl: bool) -> list[str]:
    common: set[str] | None = None
    for year in years:
        cols = set(pd.read_csv(feature_path(feature_dir, year), nrows=0).columns)
        cols.discard("grid_id")
        common = cols if common is None else common & cols
    out = sorted(common or [])
    if drop_ntl:
        out = [c for c in out if not c.startswith("NTL_")]
    return out


def load_features(feature_dir: Path, year: int, feature_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(feature_path(feature_dir, year), usecols=["grid_id"] + feature_cols)
    df["year"] = year
    df["year_norm"] = (year - 2000) / 20.0
    return df


def build_2020_strong(args: argparse.Namespace, feature_cols: list[str]) -> pd.DataFrame:
    labels = pd.read_csv(args.strong_label_csv)
    labels = labels[
        labels["fused_label"].isin(CLASSES)
        & (labels["fused_confidence"] >= args.strong_min_confidence)
        & (labels["sample_weight"] > 0)
    ].copy()
    feat = load_features(args.feature_dir, 2020, feature_cols)
    df = feat.merge(labels[["grid_id", "fused_label", "sample_weight", "fused_confidence"]], on="grid_id", how="inner")
    df = df.dropna(subset=feature_cols).copy()
    df["y"] = df["fused_label"]
    df["sample_weight"] = df["sample_weight"].astype(float)
    df["label_confidence"] = df["fused_confidence"].astype(float)
    df["source"] = "strong_multisource_2020"
    return df


def build_historical_tculu(args: argparse.Namespace, feature_cols: list[str], weak_years: list[int]) -> pd.DataFrame:
    tculu = gpd.read_file(args.tculu_grid_gpkg, layer=args.tculu_layer)
    all_rows = []
    for year in weak_years:
        label_col = f"tculu_dom_class_{year}"
        frac_col = f"tculu_dom_frac_{year}"
        valid_col = f"tculu_valid_frac_{year}"
        tmp = pd.DataFrame(
            {
                "grid_id": tculu["grid_id"].astype(int),
                "y": tculu[label_col].map(TCULU_TO_5CLASS),
                "label_confidence": pd.to_numeric(tculu[frac_col], errors="coerce").fillna(0.0),
                "valid_frac": pd.to_numeric(tculu[valid_col], errors="coerce").fillna(0.0),
            }
        )
        tmp = tmp[
            tmp["y"].isin(CLASSES)
            & (tmp["label_confidence"] >= args.historical_min_dom_frac)
            & (tmp["valid_frac"] >= args.historical_min_valid_frac)
        ].copy()
        tmp["sample_weight"] = args.historical_weight * tmp["label_confidence"]
        tmp.loc[tmp["y"] == "production", "sample_weight"] = args.historical_production_weight * tmp["label_confidence"]
        feat = load_features(args.feature_dir, year, feature_cols)
        joined = feat.merge(tmp[["grid_id", "y", "sample_weight", "label_confidence"]], on="grid_id", how="inner")
        joined = joined.dropna(subset=feature_cols).copy()
        joined["source"] = f"tculu_weak_{year}"
        all_rows.append(joined)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def build_manual_2005(args: argparse.Namespace, feature_cols: list[str]) -> pd.DataFrame:
    manual = gpd.read_file(args.manual_grid_gpkg, layer=args.manual_grid_layer)
    manual["y"] = manual["manual_class"].map(MANUAL_TO_5CLASS)
    manual = manual[manual["y"].isin(CLASSES)].copy()
    feat = load_features(args.feature_dir, 2005, feature_cols)
    df = feat.merge(manual[["grid_id", "y", "overlap_share"]], on="grid_id", how="inner")
    df = df.dropna(subset=feature_cols).copy()
    df["sample_weight"] = args.manual_weight
    df["label_confidence"] = df["overlap_share"].fillna(1.0).clip(lower=0, upper=1)
    df["source"] = "manual_2005"
    return df


def build_training(args: argparse.Namespace, feature_cols: list[str], weak_years: list[int]) -> pd.DataFrame:
    parts = [
        build_2020_strong(args, feature_cols),
        build_historical_tculu(args, feature_cols, weak_years),
        build_manual_2005(args, feature_cols),
    ]
    train = pd.concat(parts, ignore_index=True)
    train = train[train["y"].isin(CLASSES)].dropna(subset=feature_cols + ["year_norm"]).copy()
    return train


def weighted_resample(X: np.ndarray, y: np.ndarray, weights: np.ndarray, multiplier: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    n = max(len(y), int(len(y) * multiplier))
    p = weights.astype(float)
    p = p / p.sum()
    idx = rng.choice(np.arange(len(y)), size=n, replace=True, p=p)
    return X[idx], y[idx]


def train_models(args: argparse.Namespace, train: pd.DataFrame, model_features: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    X = train[model_features].to_numpy()
    y = train["y"].astype(str).to_numpy()
    sw = train["sample_weight"].astype(float).to_numpy()
    idx = train.index.to_numpy()

    X_train, X_test, y_train, y_test, sw_train, sw_test, idx_train, idx_test = train_test_split(
        X, y, sw, idx, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    models: dict[str, Any] = {}
    rf = RandomForestClassifier(
        n_estimators=args.rf_trees,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train, sample_weight=sw_train)
    models["rf"] = rf

    X_mlp, y_mlp = weighted_resample(X_train, y_train, sw_train, args.mlp_resample_multiplier)
    mlp = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=parse_hidden(args.mlp_hidden),
                    activation="relu",
                    solver="adam",
                    batch_size=256,
                    learning_rate_init=1e-3,
                    early_stopping=False,
                    n_iter_no_change=25,
                    max_iter=args.mlp_max_iter,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    mlp.fit(X_mlp, y_mlp)
    models["mlp"] = mlp

    rows = []
    heldout = train.loc[idx_test, ["grid_id", "year", "y", "source", "sample_weight"]].copy()
    for name, model in models.items():
        pred = model.predict(X_test)
        heldout[f"pred_{name}"] = pred
        report = classification_report(y_test, pred, labels=CLASSES, output_dict=True, zero_division=0)
        cm = pd.DataFrame(confusion_matrix(y_test, pred, labels=CLASSES), index=CLASSES, columns=CLASSES)
        rows.append(
            {
                "model": name,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "accuracy": accuracy_score(y_test, pred),
                "f1_macro": f1_score(y_test, pred, labels=CLASSES, average="macro"),
                "f1_weighted": f1_score(y_test, pred, labels=CLASSES, average="weighted"),
                "production_f1": report["production"]["f1-score"],
                "production_precision": report["production"]["precision"],
                "production_recall": report["production"]["recall"],
            }
        )
        pd.DataFrame(report).T.to_csv(args.out_dir / f"classification_report_joint_{name}.csv", encoding="utf-8-sig")
        cm.to_csv(args.out_dir / f"confusion_matrix_joint_{name}.csv", encoding="utf-8-sig")
        joblib.dump(model, args.out_dir / f"{name}_joint_5class_multiyear.joblib")

    return models, pd.DataFrame(rows), heldout


def apply_gaia_constraint(pred: np.ndarray, proba: np.ndarray, classes: list[str], built: np.ndarray, no_constraint: bool) -> tuple[np.ndarray, np.ndarray]:
    final = pred.astype(object).copy()
    adjusted = np.zeros(len(final), dtype=bool)
    if no_constraint:
        return final, adjusted
    need = (built == 0) & pd.Series(final).isin(BUILT_CLASSES).to_numpy()
    if need.any():
        natural_indices = [classes.index(c) for c in NATURAL_CLASSES if c in classes]
        natural_names = np.asarray([classes[i] for i in natural_indices], dtype=object)
        best_natural = natural_names[proba[:, natural_indices].argmax(axis=1)]
        final[need] = best_natural[need]
        adjusted = need
    return final, adjusted


def predict_all_years(
    args: argparse.Namespace,
    model: Any,
    model_name: str,
    model_features: list[str],
    feature_cols: list[str],
    years: list[int],
    grid: gpd.GeoDataFrame,
    gaia: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    base = gpd.GeoDataFrame(grid[["grid_id", "geometry"]].copy(), geometry="geometry", crs=grid.crs)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["mlp"].classes_)
    for year in years:
        feat = load_features(args.feature_dir, year, feature_cols)
        df = grid[["grid_id", "geometry"]].merge(feat[["grid_id"] + model_features], on="grid_id", how="inner")
        df = df.dropna(subset=model_features).copy()
        pred = model.predict(df[model_features].to_numpy())
        proba = model.predict_proba(df[model_features].to_numpy())
        conf = proba.max(axis=1)

        built_col = f"gaia_built_{year}"
        ratio_col = f"gaia_built_ratio_{year}"
        eco_col = f"gaia_ecological_{year}"
        g = gaia[["grid_id", built_col, ratio_col, eco_col]].set_index("grid_id").reindex(df["grid_id"])
        final, adjusted = apply_gaia_constraint(pred, proba, classes, g[built_col].fillna(0).to_numpy(), args.no_gaia_constraint)

        out = pd.DataFrame(
            {
                "grid_id": df["grid_id"].to_numpy(),
                f"{model_name}_raw_{year}": pred,
                f"{model_name}_conf_{year}": conf,
                f"{model_name}_pred5_{year}": final,
                f"{model_name}_gaia_adjusted_{year}": adjusted.astype(int),
                built_col: g[built_col].to_numpy(),
                ratio_col: g[ratio_col].to_numpy(),
                eco_col: g[eco_col].to_numpy(),
            }
        )
        aligned = out.set_index("grid_id").reindex(base["grid_id"])
        for col in out.columns:
            if col == "grid_id":
                continue
            base[col] = aligned[col].to_numpy()
    return base


def summarize_area(gdf: gpd.GeoDataFrame, model_name: str, years: list[int]) -> pd.DataFrame:
    area = gdf.geometry.area / 1e6
    rows = []
    for year in years:
        col = f"{model_name}_pred5_{year}"
        conf_col = f"{model_name}_conf_{year}"
        adj_col = f"{model_name}_gaia_adjusted_{year}"
        for cls, grp in gdf.groupby(col, dropna=False):
            idx = grp.index
            rows.append(
                {
                    "model": model_name,
                    "year": year,
                    "class": cls if pd.notna(cls) else "missing",
                    "n_grids": len(grp),
                    "area_km2": float(area.loc[idx].sum()),
                    "mean_confidence": float(gdf.loc[idx, conf_col].mean()) if conf_col in gdf else np.nan,
                    "gaia_adjusted_n": int(gdf.loc[idx, adj_col].sum()) if adj_col in gdf else 0,
                }
            )
    df = pd.DataFrame(rows)
    df["area_share_percent"] = df["area_km2"] / df.groupby(["model", "year"])["area_km2"].transform("sum") * 100
    return df.sort_values(["model", "year", "class"])


def write_transitions(gdf: gpd.GeoDataFrame, model_name: str, years: list[int], out_dir: Path) -> None:
    for y0, y1 in zip(years[:-1], years[1:]):
        pd.crosstab(gdf[f"{model_name}_pred5_{y0}"], gdf[f"{model_name}_pred5_{y1}"], dropna=False).to_csv(
            out_dir / f"transition_{model_name}_{y0}_{y1}_5class.csv", encoding="utf-8-sig"
        )
    pd.crosstab(gdf[f"{model_name}_pred5_{years[0]}"], gdf[f"{model_name}_pred5_{years[-1]}"], dropna=False).to_csv(
        out_dir / f"transition_{model_name}_{years[0]}_{years[-1]}_5class.csv", encoding="utf-8-sig"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    years = parse_years(args.years)
    weak_years = parse_years(args.historical_weak_years)
    feature_cols = common_feature_cols(args.feature_dir, years, args.drop_ntl)
    model_features = feature_cols + ["year_norm"]

    print("Feature count:", len(feature_cols), "+ year_norm")
    train = build_training(args, feature_cols, weak_years)
    train.to_csv(args.out_dir / "joint_training_samples_5class.csv", index=False, encoding="utf-8-sig")
    sample_summary = train.groupby(["source", "year", "y"]).size().reset_index(name="n")
    sample_summary.to_csv(args.out_dir / "joint_training_sample_summary.csv", index=False, encoding="utf-8-sig")
    print("Training sample summary:")
    print(sample_summary.to_string(index=False))

    models, summary, heldout = train_models(args, train, model_features)
    summary.to_csv(args.out_dir / "joint_model_summary.csv", index=False, encoding="utf-8-sig")
    heldout.to_csv(args.out_dir / "joint_heldout_predictions.csv", index=False, encoding="utf-8-sig")
    print("\nModel summary:")
    print(summary.round(4).to_string(index=False))

    grid = gpd.read_file(args.grid_gpkg, layer=args.grid_layer)
    grid["grid_id"] = grid["grid_id"].astype(int)
    gaia = gpd.read_file(args.gaia_grid_gpkg, layer=args.gaia_grid_layer)

    area_tables = []
    for model_name, model in models.items():
        print(f"Predicting all years with {model_name}...")
        pred_gdf = predict_all_years(args, model, model_name, model_features, feature_cols, years, grid, gaia)
        gpkg = args.out_dir / f"shenzhen_5class_joint_{model_name}_multiyear.gpkg"
        if gpkg.exists():
            gpkg.unlink()
        pred_gdf.to_file(gpkg, layer=f"pred5_joint_{model_name}", driver="GPKG")
        area = summarize_area(pred_gdf, model_name, years)
        area.to_csv(args.out_dir / f"area_by_class_joint_{model_name}.csv", index=False, encoding="utf-8-sig")
        write_transitions(pred_gdf, model_name, years, args.out_dir)
        area_tables.append(area)

    all_area = pd.concat(area_tables, ignore_index=True)
    all_area.to_csv(args.out_dir / "area_by_class_joint_all_models.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "training_sources": [
            "2020 multi-source strong labels",
            "2000/2005/2010/2015 TCULU historical weak labels",
            "2005 manual production/living samples",
        ],
        "classes": CLASSES,
        "years": years,
        "weak_years": weak_years,
        "feature_count": len(feature_cols),
        "model_features_include_year_norm": True,
        "args": vars(args),
    }
    with open(args.out_dir / "metadata_joint_5class_multiyear.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

    print("\nArea summary:")
    print(all_area.round(3).to_string(index=False))
    print(f"\nSaved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()

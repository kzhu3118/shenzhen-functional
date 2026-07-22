#!/usr/bin/env python3
"""Prepare final-class uncertainty data for Part 3.2 land-use maps.

This script does not modify the original classification GPKG. It reloads the
saved joint MLP model, recomputes five-class probabilities for each year, and
then extracts the probability assigned to the final GAIA-constrained class
shown in the land-use maps.

Main uncertainty metric:
    uncertainty_YEAR = 1 - P(final GAIA-constrained class)
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
from sklearn.isotonic import IsotonicRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "data/mlp_multiyear_5class_joint_tculuWeak_manual2005_gaia_thr10"
METADATA = RESULT_DIR / "metadata_joint_5class_multiyear.json"
MODEL_PATH = RESULT_DIR / "mlp_joint_5class_multiyear.joblib"
HELDOUT_CSV = RESULT_DIR / "joint_heldout_predictions.csv"
FINAL_GPKG = RESULT_DIR / "shenzhen_5class_joint_mlp_multiyear.gpkg"
FINAL_LAYER = "pred5_joint_mlp"
OUT_DIR = RESULT_DIR / "uncertainty"

CLASSES = ["farmland", "green_space", "living", "production", "water"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--heldout-csv", type=Path, default=HELDOUT_CSV)
    parser.add_argument("--final-gpkg", type=Path, default=FINAL_GPKG)
    parser.add_argument("--final-layer", default=FINAL_LAYER)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--out-layer", default="landuse_uncertainty_finalclass")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_metadata(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def feature_path(feature_dir: Path, year: int) -> Path:
    if year == 2020:
        return feature_dir / "shenzhen_features_2020.csv"
    return feature_dir / f"shenzhen_features_{year}_noNTL.csv"


def common_feature_cols(feature_dir: Path, years: list[int], drop_ntl: bool) -> list[str]:
    common: set[str] | None = None
    for year in years:
        path = feature_path(feature_dir, year)
        cols = set(pd.read_csv(path, nrows=0).columns)
        cols.discard("grid_id")
        common = cols if common is None else common & cols
    out = sorted(common or [])
    if drop_ntl:
        out = [c for c in out if not c.startswith("NTL_")]
    return out


def load_features(feature_dir: Path, year: int, feature_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(feature_path(feature_dir, year), usecols=["grid_id"] + feature_cols)
    df["year_norm"] = (year - 2000) / 20.0
    return df


def model_classes(model: Any) -> list[str]:
    if hasattr(model, "classes_"):
        return list(model.classes_)
    if hasattr(model, "named_steps") and "mlp" in model.named_steps:
        return list(model.named_steps["mlp"].classes_)
    raise ValueError("Cannot determine class order from the saved model.")


def normalized_entropy(proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(proba, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1) / np.log(proba.shape[1])


def margin_uncertainty(proba: np.ndarray) -> np.ndarray:
    sorted_proba = np.sort(proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    return 1.0 - margin


def probability_of_labels(proba: np.ndarray, labels: pd.Series, classes: list[str]) -> np.ndarray:
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    out = np.full(len(labels), np.nan, dtype=float)
    for cls, idx in class_to_idx.items():
        mask = labels.to_numpy() == cls
        out[mask] = proba[mask, idx]
    return out


def build_calibration_rows(
    heldout_csv: Path,
    model: Any,
    classes: list[str],
    feature_dir: Path,
    feature_cols: list[str],
    model_features: list[str],
) -> pd.DataFrame:
    heldout = pd.read_csv(heldout_csv)
    parts = []
    for year, grp in heldout.groupby("year"):
        year = int(year)
        feat = load_features(feature_dir, year, feature_cols)
        joined = grp.merge(feat[["grid_id"] + model_features], on="grid_id", how="left")
        valid = joined[model_features].notna().all(axis=1)
        proba = np.full((len(joined), len(classes)), np.nan, dtype=float)
        proba[valid.to_numpy(), :] = model.predict_proba(joined.loc[valid, model_features].to_numpy())

        rows = joined[["grid_id", "year", "y", "source"]].copy()
        for cls in classes:
            rows[f"prob_{cls}"] = proba[:, classes.index(cls)]
            rows[f"is_{cls}"] = (rows["y"].astype(str) == cls).astype(int)
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def build_isotonic_calibrators(cal_rows: pd.DataFrame, classes: list[str]) -> dict[str, IsotonicRegression]:
    calibrators: dict[str, IsotonicRegression] = {}
    for cls in classes:
        x = cal_rows[f"prob_{cls}"].to_numpy()
        y = cal_rows[f"is_{cls}"].to_numpy()
        ok = np.isfinite(x)
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(x[ok], y[ok])
        calibrators[cls] = calibrator
    return calibrators


def apply_calibrators(proba: np.ndarray, classes: list[str], calibrators: dict[str, IsotonicRegression]) -> np.ndarray:
    calibrated = np.full_like(proba, np.nan, dtype=float)
    for cls in classes:
        idx = classes.index(cls)
        x = proba[:, idx]
        ok = np.isfinite(x)
        calibrated[ok, idx] = calibrators[cls].predict(x[ok])
    return calibrated


def calibration_diagnostics(
    cal_rows: pd.DataFrame,
    classes: list[str],
    calibrators: dict[str, IsotonicRegression],
) -> pd.DataFrame:
    rows = []
    bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0000001])
    for cls in classes:
        raw = cal_rows[f"prob_{cls}"].to_numpy()
        obs = cal_rows[f"is_{cls}"].to_numpy()
        ok = np.isfinite(raw)
        raw = raw[ok]
        obs = obs[ok]
        cal = calibrators[cls].predict(raw)
        bin_id = np.digitize(raw, bins, right=False) - 1
        for i in range(len(bins) - 1):
            m = bin_id == i
            if not m.any():
                continue
            rows.append(
                {
                    "class": cls,
                    "raw_prob_bin": f"[{bins[i]:.2f}, {bins[i + 1]:.2f})",
                    "n": int(m.sum()),
                    "mean_raw_prob": float(raw[m].mean()),
                    "mean_calibrated_prob": float(cal[m].mean()),
                    "observed_frequency": float(obs[m].mean()),
                }
            )
    return pd.DataFrame(rows)


def area_km2(gdf: gpd.GeoDataFrame) -> pd.Series:
    if gdf.crs is not None and gdf.crs.is_geographic:
        geom = gdf.to_crs(epsg=32649).geometry
    else:
        geom = gdf.geometry
    return geom.area / 1e6


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def summarize_by_year(
    gdf: gpd.GeoDataFrame,
    years: list[int],
    prob_prefix: str = "final_prob",
    uncertainty_prefix: str = "uncertainty",
) -> pd.DataFrame:
    rows = []
    area = area_km2(gdf)
    for year in years:
        unc = gdf[f"{uncertainty_prefix}_{year}"]
        final_prob = gdf[f"{prob_prefix}_{year}"]
        adj = gdf[f"mlp_gaia_adjusted_{year}"]
        rows.append(
            {
                "year": year,
                "n_grids": int(unc.notna().sum()),
                "area_km2": float(area.loc[unc.notna()].sum()),
                "mean_final_prob": float(final_prob.mean()),
                "mean_uncertainty": float(unc.mean()),
                "median_uncertainty": float(unc.median()),
                "p75_uncertainty": float(unc.quantile(0.75)),
                "p90_uncertainty": float(unc.quantile(0.90)),
                "high_uncertainty_gt_0p2_pct": float((unc > 0.2).mean() * 100),
                "high_uncertainty_gt_0p3_pct": float((unc > 0.3).mean() * 100),
                "gaia_adjusted_pct": float((adj == 1).mean() * 100),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_class(
    gdf: gpd.GeoDataFrame,
    years: list[int],
    prob_prefix: str = "final_prob",
    uncertainty_prefix: str = "uncertainty",
) -> pd.DataFrame:
    rows = []
    area = area_km2(gdf)
    for year in years:
        cls_col = f"mlp_pred5_{year}"
        for cls, grp in gdf.groupby(cls_col, dropna=False):
            idx = grp.index
            unc = gdf.loc[idx, f"{uncertainty_prefix}_{year}"]
            valid_idx = unc[unc.notna()].index
            final_prob = gdf.loc[idx, f"{prob_prefix}_{year}"]
            adj = gdf.loc[idx, f"mlp_gaia_adjusted_{year}"]
            rows.append(
                {
                    "year": year,
                    "class": cls if pd.notna(cls) else "missing",
                    "n_grids": int(unc.notna().sum()),
                    "area_km2": float(area.loc[valid_idx].sum()),
                    "mean_final_prob": float(final_prob.mean()),
                    "mean_uncertainty": float(unc.mean()),
                    "median_uncertainty": float(unc.median()),
                    "p90_uncertainty": float(unc.quantile(0.90)),
                    "gaia_adjusted_pct": float((adj == 1).mean() * 100),
                }
            )
    return pd.DataFrame(rows).sort_values(["year", "class"])


def main() -> None:
    args = parse_args()
    meta = read_metadata(args.metadata)
    meta_args = meta["args"]
    years = [int(y) for y in meta["years"]]
    feature_dir = Path(meta_args["feature_dir"])
    if not feature_dir.is_absolute():
        feature_dir = PROJECT_ROOT / feature_dir
    drop_ntl = bool(meta_args.get("drop_ntl", False))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_gpkg = args.out_dir / "shenzhen_5class_joint_mlp_uncertainty_finalclass.gpkg"
    if out_gpkg.exists() and not args.overwrite:
        raise FileExistsError(f"{out_gpkg} exists. Use --overwrite to replace it.")
    if out_gpkg.exists() and args.overwrite:
        out_gpkg.unlink()

    print("[1] Load final GAIA-constrained classification...")
    final_gdf = gpd.read_file(args.final_gpkg, layer=args.final_layer)
    final_gdf["grid_id"] = final_gdf["grid_id"].astype(int)

    print("[2] Load saved MLP model...")
    model = joblib.load(args.model_path)
    classes = model_classes(model)
    missing_classes = sorted(set(CLASSES) - set(classes))
    if missing_classes:
        raise ValueError(f"Saved model is missing expected classes: {missing_classes}")

    print("[3] Rebuild feature column order...")
    feature_cols = common_feature_cols(feature_dir, years, drop_ntl)
    model_features = feature_cols + ["year_norm"]
    print(f"    Feature count: {len(feature_cols)} + year_norm")

    print("[4] Fit one-vs-rest isotonic probability calibrators from held-out samples...")
    cal_rows = build_calibration_rows(args.heldout_csv, model, classes, feature_dir, feature_cols, model_features)
    calibrators = build_isotonic_calibrators(cal_rows, classes)
    calibration_diagnostics(cal_rows, classes, calibrators).to_csv(
        args.out_dir / "landuse_uncertainty_calibration_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    out_gdf = final_gdf[["grid_id", "geometry"]].copy()
    for year in years:
        print(f"[5] Predict raw/calibrated probabilities and uncertainty for {year}...")
        feat = load_features(feature_dir, year, feature_cols)
        joined = final_gdf[["grid_id", f"mlp_pred5_{year}", f"mlp_gaia_adjusted_{year}"]].merge(
            feat[["grid_id"] + model_features],
            on="grid_id",
            how="left",
        )

        valid = joined[model_features].notna().all(axis=1)
        proba = np.full((len(joined), len(classes)), np.nan, dtype=float)
        proba[valid.to_numpy(), :] = model.predict_proba(joined.loc[valid, model_features].to_numpy())
        cal_proba = apply_calibrators(proba, classes, calibrators)

        for cls in CLASSES:
            if cls in classes:
                out_gdf[f"prob_{cls}_{year}"] = proba[:, classes.index(cls)]
                out_gdf[f"cal_prob_{cls}_{year}"] = cal_proba[:, classes.index(cls)]

        final_labels = joined[f"mlp_pred5_{year}"].astype(object)
        final_prob = probability_of_labels(proba, final_labels, classes)
        cal_final_prob = probability_of_labels(cal_proba, final_labels, classes)
        out_gdf[f"mlp_pred5_{year}"] = joined[f"mlp_pred5_{year}"].to_numpy()
        out_gdf[f"mlp_gaia_adjusted_{year}"] = joined[f"mlp_gaia_adjusted_{year}"].to_numpy()
        out_gdf[f"final_prob_{year}"] = final_prob
        out_gdf[f"uncertainty_{year}"] = 1.0 - final_prob
        out_gdf[f"cal_final_prob_{year}"] = cal_final_prob
        out_gdf[f"cal_uncertainty_{year}"] = 1.0 - cal_final_prob
        out_gdf[f"entropy_{year}"] = normalized_entropy(proba)
        out_gdf[f"margin_uncertainty_{year}"] = margin_uncertainty(proba)

    print("[6] Write uncertainty GPKG and summary tables...")
    out_gdf.to_file(out_gpkg, layer=args.out_layer, driver="GPKG")
    summarize_by_year(out_gdf, years).to_csv(
        args.out_dir / "landuse_uncertainty_summary_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summarize_by_class(out_gdf, years).to_csv(
        args.out_dir / "landuse_uncertainty_summary_by_class.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summarize_by_year(out_gdf, years, "cal_final_prob", "cal_uncertainty").to_csv(
        args.out_dir / "landuse_calibrated_uncertainty_summary_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summarize_by_class(out_gdf, years, "cal_final_prob", "cal_uncertainty").to_csv(
        args.out_dir / "landuse_calibrated_uncertainty_summary_by_class.csv",
        index=False,
        encoding="utf-8-sig",
    )

    run_meta = {
        "source_final_gpkg": project_relative(args.final_gpkg),
        "source_final_layer": args.final_layer,
        "source_model": project_relative(args.model_path),
        "source_heldout": project_relative(args.heldout_csv),
        "metadata": project_relative(args.metadata),
        "years": years,
        "model_classes_order": classes,
        "feature_count": len(feature_cols),
        "metric_definition": "uncertainty_YEAR = 1 - P(final GAIA-constrained class)",
        "calibrated_metric_definition": (
            "cal_uncertainty_YEAR = 1 - calibrated P(final GAIA-constrained class); "
            "calibration uses one-vs-rest isotonic regression fitted on held-out samples."
        ),
        "note": (
            "final_prob_YEAR is the MLP probability assigned to mlp_pred5_YEAR, "
            "the final post-GAIA-constraint class used in the land-use maps. "
            "cal_final_prob_YEAR is the corresponding one-vs-rest isotonic calibrated probability."
        ),
    }
    with open(args.out_dir / "metadata_landuse_uncertainty_finalclass.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved uncertainty data to: {out_gpkg}")
    print(f"Saved summaries to: {args.out_dir}")


if __name__ == "__main__":
    main()

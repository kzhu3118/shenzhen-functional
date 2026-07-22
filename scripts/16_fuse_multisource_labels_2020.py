#!/usr/bin/env python3
"""Fuse POI-FD, EULUC, and TCULU labels into five 2020 classes.

Five target classes:
  production, living, green_space, water, farmland

This version is designed for the "functional-ecological land-use types"
experiment. It keeps green space, water, and farmland separate while merging
residential/commercial/public service into living and industrial into production.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_GPKG = PROJECT_ROOT / "data/poi_2020/poi_clean/grid_scored_2020_FD.gpkg"
EULUC_SHP = PROJECT_ROOT / "data/validation/EULUC_Shenzhen.shp"
TCULU_GRID_GPKG = PROJECT_ROOT / "data/tculu_shenzhen/validate_rf_ples_thr10/rf_ples_vs_tculu_grid_validation.gpkg"
OUT_DIR = PROJECT_ROOT / "data/multisource_labels_2020_5class"

TARGET_CLASSES = ["production", "living", "green_space", "water", "farmland"]

POI_TO_5CLASS = {
    "industrial": "production",
    "residential": "living",
    "commercial": "living",
    "public_service": "living",
}

EULUC_TO_5CLASS = {
    0: "living",
    1: "living",
    2: "living",
    3: "production",
    # 4 road and 5 airport excluded.
    6: "living",
    7: "living",
    8: "living",
    9: "living",
    10: "green_space",
}

TCULU_TO_5CLASS = {
    "industrial": "production",
    "residential": "living",
    "commercial": "living",
    "institutional": "living",
    "green_space": "green_space",
    "water": "water",
    "farmland": "farmland",
    "bare_land": "farmland",
    # transport excluded.
}

PURE_SUBCLASSES = {"industrial", "residential", "commercial", "public_service"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-gpkg", type=Path, default=GRID_GPKG)
    parser.add_argument("--grid-layer", default="grid_scored")
    parser.add_argument("--euluc-shp", type=Path, default=EULUC_SHP)
    parser.add_argument("--tculu-grid-gpkg", type=Path, default=TCULU_GRID_GPKG)
    parser.add_argument("--tculu-layer", default="rf_ples_vs_tculu")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--poi-pure-weight", type=float, default=1.0)
    parser.add_argument("--poi-mixed-weight", type=float, default=0.6)
    parser.add_argument("--euluc-weight", type=float, default=0.8)
    parser.add_argument("--tculu-weight", type=float, default=0.8)
    parser.add_argument(
        "--tculu-production-weight",
        type=float,
        default=None,
        help="Optional separate source weight for TCULU industrial->production labels.",
    )
    parser.add_argument("--tculu-min-dom-frac", type=float, default=0.50)
    parser.add_argument("--tculu-min-valid-frac", type=float, default=0.20)
    parser.add_argument("--min-fused-confidence", type=float, default=0.34)
    return parser.parse_args()


def load_grid(args: argparse.Namespace) -> gpd.GeoDataFrame:
    grid = gpd.read_file(args.grid_gpkg, layer=args.grid_layer)
    grid["grid_id"] = grid["grid_id"].astype(int)
    for col in ["ples_subclass", "dominant"]:
        grid[col] = grid[col].astype("string").str.strip()
    return grid


def build_poi_labels(grid: gpd.GeoDataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = grid[["grid_id", "ples_subclass", "dominant", "label_confidence"]].copy()
    is_pure = df["ples_subclass"].isin(PURE_SUBCLASSES)
    is_dominated = df["ples_subclass"].str.endswith("_dominated_mixed", na=False)
    valid = (is_pure | is_dominated) & df["dominant"].isin(PURE_SUBCLASSES)

    df["poi_label"] = pd.NA
    df.loc[valid, "poi_label"] = df.loc[valid, "dominant"].map(POI_TO_5CLASS)
    df["poi_source_weight"] = 0.0
    df.loc[is_pure & valid, "poi_source_weight"] = args.poi_pure_weight
    df.loc[is_dominated & valid, "poi_source_weight"] = args.poi_mixed_weight
    df["poi_label_confidence"] = df["label_confidence"].fillna(0.0).astype(float)
    return df[["grid_id", "poi_label", "poi_source_weight", "poi_label_confidence"]]


def build_euluc_labels(grid: gpd.GeoDataFrame, args: argparse.Namespace) -> pd.DataFrame:
    euluc = gpd.read_file(args.euluc_shp)
    euluc["Class"] = pd.to_numeric(euluc["Class"], errors="coerce").astype("Int64")
    euluc["euluc_label_raw"] = euluc["Class"].map(EULUC_TO_5CLASS)
    euluc = euluc[euluc["euluc_label_raw"].notna()].copy().to_crs(grid.crs)
    euluc_pts = euluc[["euluc_label_raw", "geometry"]].copy()
    euluc_pts["geometry"] = euluc_pts.geometry.centroid
    joined = gpd.sjoin(euluc_pts, grid[["grid_id", "geometry"]], predicate="within", how="inner")
    if joined.empty:
        return pd.DataFrame(columns=["grid_id", "euluc_label", "euluc_source_weight", "euluc_label_confidence"])

    counts = joined.groupby(["grid_id", "euluc_label_raw"]).size().reset_index(name="n")
    counts["frac"] = counts["n"] / counts.groupby("grid_id")["n"].transform("sum")
    dom = counts.loc[counts.groupby("grid_id")["n"].idxmax()].copy()
    dom = dom.rename(columns={"euluc_label_raw": "euluc_label", "frac": "euluc_label_confidence"})
    dom["euluc_source_weight"] = args.euluc_weight * dom["euluc_label_confidence"]
    return dom[["grid_id", "euluc_label", "euluc_source_weight", "euluc_label_confidence"]]


def build_tculu_labels(args: argparse.Namespace) -> pd.DataFrame:
    tculu = gpd.read_file(args.tculu_grid_gpkg, layer=args.tculu_layer)
    df = pd.DataFrame(tculu.drop(columns="geometry"))
    df["grid_id"] = df["grid_id"].astype(int)
    df["tculu_label"] = df["tculu_dom_class_2020"].map(TCULU_TO_5CLASS)
    ok = (
        df["tculu_label"].notna()
        & (df["tculu_dom_frac_2020"] >= args.tculu_min_dom_frac)
        & (df["tculu_valid_frac_2020"] >= args.tculu_min_valid_frac)
    )
    df.loc[~ok, "tculu_label"] = pd.NA
    df["tculu_label_confidence"] = df["tculu_dom_frac_2020"].where(ok, 0.0).fillna(0.0)
    base_weight = args.tculu_weight
    production_weight = args.tculu_production_weight if args.tculu_production_weight is not None else base_weight
    df["tculu_base_weight"] = base_weight
    df.loc[df["tculu_label"] == "production", "tculu_base_weight"] = production_weight
    df["tculu_source_weight"] = df["tculu_base_weight"] * df["tculu_label_confidence"]
    return df[["grid_id", "tculu_label", "tculu_source_weight", "tculu_label_confidence"]]


def fuse_row(row: pd.Series) -> pd.Series:
    scores = {cls: 0.0 for cls in TARGET_CLASSES}
    votes = {cls: 0 for cls in TARGET_CLASSES}
    for prefix in ["poi", "euluc", "tculu"]:
        label = row.get(f"{prefix}_label")
        weight = row.get(f"{prefix}_source_weight", 0.0)
        if pd.notna(label) and label in scores and weight > 0:
            scores[label] += float(weight)
            votes[label] += 1
    total = sum(scores.values())
    if total <= 0:
        return pd.Series(
            {
                "fused_label": pd.NA,
                "fused_confidence": 0.0,
                "source_count": 0,
                "agreement_count": 0,
                "sample_weight": 0.0,
                "fusion_status": "no_label",
            }
        )
    best = max(scores, key=scores.get)
    ordered = sorted(scores.values(), reverse=True)
    agreement = votes[best]
    status = "consensus" if agreement >= 2 else "single_source"
    if ordered[0] == ordered[1]:
        status = "tie_or_conflict"
    sample_weight = scores[best] * (1.0 + 0.15 * max(0, agreement - 1))
    return pd.Series(
        {
            "fused_label": best,
            "fused_confidence": scores[best] / total,
            "source_count": sum(1 for p in ["poi", "euluc", "tculu"] if pd.notna(row.get(f"{p}_label"))),
            "agreement_count": agreement,
            "sample_weight": sample_weight,
            "fusion_status": status,
        }
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    grid = load_grid(args)
    labels = grid[["grid_id", "geometry"]].copy()
    labels = labels.merge(build_poi_labels(grid, args), on="grid_id", how="left")
    labels = labels.merge(build_euluc_labels(grid, args), on="grid_id", how="left")
    labels = labels.merge(build_tculu_labels(args), on="grid_id", how="left")
    labels = pd.concat([labels, labels.apply(fuse_row, axis=1)], axis=1)
    labels.loc[labels["fused_confidence"] < args.min_fused_confidence, "fused_label"] = pd.NA
    labels.loc[labels["fused_label"].isna(), "sample_weight"] = 0.0

    out = gpd.GeoDataFrame(labels, geometry="geometry", crs=grid.crs)
    out_gpkg = args.out_dir / "multisource_labels_2020_5class.gpkg"
    if out_gpkg.exists():
        out_gpkg.unlink()
    out.to_file(out_gpkg, layer="labels", driver="GPKG")
    out[[c for c in out.columns if c != "geometry"]].to_csv(
        args.out_dir / "multisource_labels_2020_5class.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "target_classes": TARGET_CLASSES,
        "source_counts": {
            "poi": int(out["poi_label"].notna().sum()),
            "euluc": int(out["euluc_label"].notna().sum()),
            "tculu": int(out["tculu_label"].notna().sum()),
            "fused": int(out["fused_label"].notna().sum()),
        },
        "fused_distribution": out["fused_label"].value_counts(dropna=False).astype(int).to_dict(),
        "fusion_status_distribution": out["fusion_status"].value_counts(dropna=False).astype(int).to_dict(),
        "args": vars(args),
    }
    with open(args.out_dir / "multisource_label_fusion_metadata_5class.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Saved: {out_gpkg}")


if __name__ == "__main__":
    main()

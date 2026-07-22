#!/usr/bin/env python3
"""Validate Shenzhen RF-GAIA PLES maps against clipped TCULU land-use rasters.

Outputs:
  - per-grid GPKG with RF PLES, TCULU dominant fine class, TCULU PLES, and agreement
  - yearly confusion matrices and metric summary
  - area comparison tables

TCULU fine classes are preserved. For PLES comparison, transport and bare_land
are treated as ambiguous and excluded from the strict PLES reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize


CLASS_NAMES = {
    1: "water",
    2: "green_space",
    3: "farmland",
    4: "bare_land",
    5: "residential",
    6: "commercial",
    7: "institutional",
    8: "industrial",
    9: "transport",
}

TCULU_TO_PLES_STRICT = {
    1: "ecological",  # water
    2: "ecological",  # green space
    3: "ecological",  # farmland
    5: "living",      # residential
    6: "living",      # commercial
    7: "living",      # institutional
    8: "production",  # industrial
    # 4 bare land and 9 transport are intentionally ambiguous.
}

PLES_ORDER = ["ecological", "living", "production"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rf-gpkg",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/rf_multiyear/noNTL_manual2005_gaia_ratio_thr10/shenzhen_ples_multiyear_noNTL_manual2005_gaia_ratio.gpkg",
        help="RF-GAIA multi-year PLES GeoPackage.",
    )
    parser.add_argument(
        "--tculu-clip-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/tculu_shenzhen/clips",
        help="Directory containing shenzhen_tculu_YYYY.tif clips.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/tculu_shenzhen/validate_rf_ples_thr10",
        help="Output directory.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2000, 2005, 2010, 2015, 2020],
        help="Years to compare. RF-GAIA result currently contains 2000-2020.",
    )
    parser.add_argument(
        "--min-valid-frac",
        type=float,
        default=0.20,
        help="Minimum TCULU valid pixel fraction in a grid to assign TCULU labels.",
    )
    return parser.parse_args()


def rasterize_grid(grid_3857: gpd.GeoDataFrame, template) -> np.ndarray:
    shapes = ((geom, int(gid)) for geom, gid in zip(grid_3857.geometry, grid_3857["grid_id"]))
    return rasterize(
        shapes,
        out_shape=(template.height, template.width),
        transform=template.transform,
        fill=-1,
        dtype="int32",
        all_touched=False,
    )


def class_counts_by_grid(arr: np.ndarray, grid_ids: np.ndarray, nodata) -> pd.DataFrame:
    valid = (grid_ids >= 0) & (arr != 0)
    if nodata is not None:
        valid &= arr != nodata

    gid = grid_ids[valid].astype(np.int64)
    cls = arr[valid].astype(np.int64)
    if len(gid) == 0:
        return pd.DataFrame(columns=["grid_id", "class_code", "pixel_count"])

    # Composite bincount is much faster than per-grid loops.
    max_class = 10
    comp = gid * max_class + cls
    counts = np.bincount(comp)
    idx = np.nonzero(counts)[0]
    out = pd.DataFrame(
        {
            "grid_id": idx // max_class,
            "class_code": idx % max_class,
            "pixel_count": counts[idx],
        }
    )
    return out[out["class_code"].isin(CLASS_NAMES)]


def build_year_grid_labels(
    counts: pd.DataFrame,
    all_grid_ids: pd.Series,
    year: int,
    total_grid_pixels: pd.Series,
    min_valid_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = (
        counts.pivot_table(index="grid_id", columns="class_code", values="pixel_count", aggfunc="sum", fill_value=0)
        .reindex(all_grid_ids, fill_value=0)
        .sort_index()
    )
    for code in CLASS_NAMES:
        if code not in pivot.columns:
            pivot[code] = 0
    pivot = pivot[sorted(CLASS_NAMES)]

    valid_pixels = pivot.sum(axis=1)
    valid_frac = valid_pixels / total_grid_pixels.reindex(pivot.index)

    dominant_code = pivot.idxmax(axis=1).where(valid_pixels > 0)
    dominant_count = pivot.max(axis=1)
    dominant_frac = dominant_count / valid_pixels.replace(0, np.nan)
    dominant_name = dominant_code.map(CLASS_NAMES)

    ples_counts = pd.DataFrame(index=pivot.index)
    for ples in PLES_ORDER:
        codes = [c for c, p in TCULU_TO_PLES_STRICT.items() if p == ples]
        ples_counts[ples] = pivot[codes].sum(axis=1)

    strict_valid_pixels = ples_counts.sum(axis=1)
    tculu_ples = ples_counts.idxmax(axis=1).where(strict_valid_pixels > 0)
    tculu_ples_frac = ples_counts.max(axis=1) / strict_valid_pixels.replace(0, np.nan)

    low_coverage = valid_frac < min_valid_frac
    dominant_name = dominant_name.mask(low_coverage)
    dominant_code = dominant_code.mask(low_coverage)
    dominant_frac = dominant_frac.mask(low_coverage)
    tculu_ples = tculu_ples.mask(low_coverage | (strict_valid_pixels == 0))
    tculu_ples_frac = tculu_ples_frac.mask(low_coverage | (strict_valid_pixels == 0))

    labels = pd.DataFrame(
        {
            "grid_id": pivot.index.astype(int),
            f"tculu_valid_frac_{year}": valid_frac.values,
            f"tculu_dom_code_{year}": dominant_code.values,
            f"tculu_dom_class_{year}": dominant_name.values,
            f"tculu_dom_frac_{year}": dominant_frac.values,
            f"tculu_ples_{year}": tculu_ples.values,
            f"tculu_ples_frac_{year}": tculu_ples_frac.values,
        }
    )

    area_rows = []
    # Area for TCULU clipped rasters already works on 10 m projected pixels.
    # Here we use counts for relative comparison on the same grid, not official
    # area accounting. The area table from extract_tculu_shenzhen.py remains the
    # corrected source for absolute TCULU areas.
    for code, name in CLASS_NAMES.items():
        area_rows.append(
            {
                "year": year,
                "source": "TCULU_fine_grid_count_raw",
                "class": name,
                "pixel_count": int(pivot[code].sum()),
            }
        )
    for ples in PLES_ORDER:
        area_rows.append(
            {
                "year": year,
                "source": "TCULU_PLES_strict_grid_count_raw",
                "class": ples,
                "pixel_count": int(ples_counts[ples].sum()),
            }
        )
    area = pd.DataFrame(area_rows)
    return labels, area


def confusion_and_metrics(df: pd.DataFrame, pred_col: str, ref_col: str, year: int) -> tuple[pd.DataFrame, dict]:
    valid = df[pred_col].isin(PLES_ORDER) & df[ref_col].isin(PLES_ORDER)
    sub = df.loc[valid, [pred_col, ref_col]].copy()
    cm = pd.crosstab(sub[ref_col], sub[pred_col], rownames=["tculu_ref"], colnames=["rf_pred"])
    cm = cm.reindex(index=PLES_ORDER, columns=PLES_ORDER, fill_value=0)

    total = int(cm.values.sum())
    correct = int(np.trace(cm.values))
    accuracy = correct / total if total else np.nan

    f1s = []
    recalls = []
    precisions = []
    for cls in PLES_ORDER:
        tp = cm.loc[cls, cls]
        fp = cm[cls].sum() - tp
        fn = cm.loc[cls].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = 2 * precision * recall / (precision + recall) if precision == precision and recall == recall and (precision + recall) else np.nan
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    metrics = {
        "year": year,
        "n_compared_grids": total,
        "accuracy": accuracy,
        "macro_precision": float(np.nanmean(precisions)),
        "macro_recall": float(np.nanmean(recalls)),
        "macro_f1": float(np.nanmean(f1s)),
    }
    for cls, precision, recall, f1 in zip(PLES_ORDER, precisions, recalls, f1s):
        metrics[f"{cls}_precision"] = precision
        metrics[f"{cls}_recall"] = recall
        metrics[f"{cls}_f1"] = f1
        metrics[f"{cls}_support_tculu"] = int(cm.loc[cls].sum())

    cm = cm.reset_index()
    cm.insert(0, "year", year)
    return cm, metrics


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rf = gpd.read_file(args.rf_gpkg)
    if "grid_id" not in rf.columns:
        raise ValueError("RF-GAIA GeoPackage must contain grid_id.")
    rf["grid_id"] = rf["grid_id"].astype(int)
    rf = rf.sort_values("grid_id").reset_index(drop=True)

    first_raster = args.tculu_clip_dir / f"shenzhen_tculu_{args.years[0]}.tif"
    with rasterio.open(first_raster) as src:
        grid_3857 = rf[["grid_id", "geometry"]].to_crs(CRS.from_epsg(3857))
        grid_raster = rasterize_grid(grid_3857, src)
        total_grid_pixels = pd.Series(np.bincount(grid_raster[grid_raster >= 0].ravel()), name="total_grid_pixels")

    per_year_labels = []
    area_rows = []
    cms = []
    metric_rows = []
    agreement_cols = []

    for year in args.years:
        raster_path = args.tculu_clip_dir / f"shenzhen_tculu_{year}.tif"
        if not raster_path.exists():
            raise FileNotFoundError(raster_path)
        rf_col = f"ples_{year}"
        if rf_col not in rf.columns:
            raise ValueError(f"Missing RF column: {rf_col}")

        with rasterio.open(raster_path) as src:
            arr = src.read(1)
            if arr.shape != grid_raster.shape:
                raise ValueError(f"Raster shape mismatch for {year}: {arr.shape} vs {grid_raster.shape}")
            counts = class_counts_by_grid(arr, grid_raster, src.nodata)

        labels, area = build_year_grid_labels(
            counts,
            rf["grid_id"],
            year,
            total_grid_pixels,
            args.min_valid_frac,
        )
        per_year_labels.append(labels)
        area_rows.append(area)

        tmp = rf[["grid_id", rf_col]].merge(labels[["grid_id", f"tculu_ples_{year}"]], on="grid_id", how="left")
        cm, metrics = confusion_and_metrics(tmp, rf_col, f"tculu_ples_{year}", year)
        cms.append(cm)
        metric_rows.append(metrics)

        agree_col = f"agree_ples_{year}"
        labels[agree_col] = tmp[rf_col].values == tmp[f"tculu_ples_{year}"].values
        agreement_cols.append(agree_col)

    out = rf.copy()
    for labels in per_year_labels:
        out = out.merge(labels, on="grid_id", how="left")

    out_gpkg = args.out_dir / "rf_ples_vs_tculu_grid_validation.gpkg"
    if out_gpkg.exists():
        out_gpkg.unlink()
    out.to_file(out_gpkg, layer="rf_ples_vs_tculu", driver="GPKG")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.out_dir / "metrics_by_year.csv", index=False, encoding="utf-8-sig")

    cm_df = pd.concat(cms, ignore_index=True)
    cm_df.to_csv(args.out_dir / "confusion_matrices_by_year.csv", index=False, encoding="utf-8-sig")

    area_df = pd.concat(area_rows, ignore_index=True)
    area_df.to_csv(args.out_dir / "tculu_grid_count_area_reference.csv", index=False, encoding="utf-8-sig")

    # Compare RF-GAIA class grid counts with strict TCULU PLES grid counts.
    compare_rows = []
    for year in args.years:
        rf_counts = out[f"ples_{year}"].value_counts(dropna=False)
        tculu_counts = out[f"tculu_ples_{year}"].value_counts(dropna=False)
        for cls in PLES_ORDER:
            compare_rows.append(
                {
                    "year": year,
                    "class": cls,
                    "rf_grid_count": int(rf_counts.get(cls, 0)),
                    "tculu_ref_grid_count": int(tculu_counts.get(cls, 0)),
                }
            )
        compare_rows.append(
            {
                "year": year,
                "class": "tculu_unassigned_or_ambiguous",
                "rf_grid_count": np.nan,
                "tculu_ref_grid_count": int(out[f"tculu_ples_{year}"].isna().sum()),
            }
        )
    pd.DataFrame(compare_rows).to_csv(args.out_dir / "rf_vs_tculu_grid_count_summary.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "rf_gpkg": str(args.rf_gpkg),
        "tculu_clip_dir": str(args.tculu_clip_dir),
        "out_dir": str(args.out_dir),
        "years": args.years,
        "min_valid_frac": args.min_valid_frac,
        "class_names": CLASS_NAMES,
        "strict_ples_mapping": TCULU_TO_PLES_STRICT,
        "ambiguous_tculu_classes_excluded_from_strict_ples": {
            4: "bare_land",
            9: "transport",
        },
    }
    with open(args.out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Metrics:")
    print(metrics_df.round(4).to_string(index=False))
    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()

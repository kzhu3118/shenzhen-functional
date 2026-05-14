"""
Step 4a-ratio: Derive year-specific GAIA masks by built-up pixel ratio.

Why this version:
  The centroid-sampling version can mark a grid as ecological when its centroid
  falls on a non-built pixel, even if a meaningful part of the grid was already
  impervious. This is especially strict for early years when urban patches were
  smaller and fragmented.

Expected original GAIA coding:
  value 1  = first impervious in 1985
  value 2  = first impervious in 1986
  ...
  value 40 = first impervious in 2024
  value 0 or nodata = non-impervious by 2024 / background

For target year Y:
  built pixels are 1 <= gaia_value <= (Y - 1984)
  grid is built if built_pixel_count / grid_pixel_count >= threshold
  otherwise ecological.

Default thresholds:
  0.05,0.10,0.20

Outputs for each threshold tag, e.g. thr10:
  data/GAIA_processed/ratio_thr10/grid_gaia_raw_masks_ratio_thr10.gpkg
  data/GAIA_processed/ratio_thr10/gaia_year_mask_distribution_ratio_thr10.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID = PROJECT_ROOT / "data/poi_2020/poi_clean/grid_scored_2020_FD.gpkg"
DEFAULT_GAIA = PROJECT_ROOT / "data/GEE/GAIA/gaia_shenzhen/gaia_shenzhen.tif"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data/GAIA_processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create GAIA grid masks from built-up pixel ratios.")
    parser.add_argument("--grid-gpkg", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--grid-layer", default="grid_scored")
    parser.add_argument("--gaia-raster", type=Path, default=DEFAULT_GAIA)
    parser.add_argument("--years", default="2000,2005,2010,2015,2020")
    parser.add_argument(
        "--thresholds",
        default="0.05,0.10,0.20",
        help="Comma-separated built ratio thresholds. Example: 0.05,0.10,0.20",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Rasterize all pixels touched by a grid polygon. Default uses pixel centers.",
    )
    return parser.parse_args()


def parse_years(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_thresholds(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def threshold_for_year(year: int) -> int:
    if year < 1985 or year > 2024:
        raise ValueError("GAIA year should be between 1985 and 2024.")
    return year - 1984


def threshold_tag(thr: float) -> str:
    return f"thr{int(round(thr * 100)):02d}"


def main() -> None:
    args = parse_args()
    years = parse_years(args.years)
    thresholds = parse_thresholds(args.thresholds)

    if not args.grid_gpkg.exists():
        raise FileNotFoundError(f"Grid GPKG not found: {args.grid_gpkg}")
    if not args.gaia_raster.exists():
        raise FileNotFoundError(f"GAIA raster not found: {args.gaia_raster}")

    grid = gpd.read_file(args.grid_gpkg, layer=args.grid_layer)
    if "grid_id" not in grid.columns:
        raise ValueError("Grid must contain grid_id.")

    with rasterio.open(args.gaia_raster) as src:
        grid_raster_crs = grid.to_crs(src.crs) if grid.crs != src.crs else grid.copy()
        raw = src.read(1, masked=True)
        raw_data = raw.astype("float32").filled(np.nan)
        height, width = src.height, src.width
        transform = src.transform

    # Rasterize sequential feature ids rather than grid_id values, which may be sparse.
    seq_ids = np.arange(1, len(grid_raster_crs) + 1, dtype=np.int32)
    shapes = zip(grid_raster_crs.geometry, seq_ids)
    grid_id_raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=args.all_touched,
    )

    inside = grid_id_raster > 0
    seq_for_pixels = grid_id_raster[inside]
    raw_for_pixels = raw_data[inside]

    total_counts = np.bincount(seq_for_pixels, minlength=len(grid_raster_crs) + 1)
    valid_raw = np.isfinite(raw_for_pixels)
    raw_valid_values = raw_for_pixels[valid_raw]

    out_base = grid[["grid_id", "geometry"]].copy()
    out_base = gpd.GeoDataFrame(out_base, geometry="geometry", crs=grid.crs)
    out_base["gaia_pixel_count"] = total_counts[1:]
    out_base["gaia_valid_pixel_count"] = np.bincount(
        seq_for_pixels[valid_raw],
        minlength=len(grid_raster_crs) + 1,
    )[1:]

    raw_dist = (
        pd.Series(raw_valid_values)
        .value_counts(dropna=False)
        .rename_axis("gaia_raw_value")
        .reset_index(name="n_pixels")
        .sort_values("gaia_raw_value", na_position="last")
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dist.to_csv(args.out_dir / "gaia_raw_pixel_value_distribution.csv", index=False, encoding="utf-8-sig")

    for thr in thresholds:
        tag = threshold_tag(thr)
        out_dir = args.out_dir / f"ratio_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_base.copy()
        rows = []

        for year in years:
            year_threshold = threshold_for_year(year)
            built_pixel_mask = (
                np.isfinite(raw_for_pixels)
                & (raw_for_pixels >= 1)
                & (raw_for_pixels <= year_threshold)
            )
            built_counts = np.bincount(
                seq_for_pixels[built_pixel_mask],
                minlength=len(grid_raster_crs) + 1,
            )[1:]
            ratio = np.divide(
                built_counts,
                out["gaia_pixel_count"].to_numpy(),
                out=np.zeros(len(out), dtype="float64"),
                where=out["gaia_pixel_count"].to_numpy() > 0,
            )
            built = ratio >= thr
            out[f"gaia_built_pixels_{year}"] = built_counts
            out[f"gaia_built_ratio_{year}"] = ratio
            out[f"gaia_built_{year}"] = built.astype(int)
            out[f"gaia_ecological_{year}"] = (~built).astype(int)
            rows.append(
                {
                    "year": year,
                    "gaia_year_threshold": year_threshold,
                    "built_ratio_threshold": thr,
                    "built_grids": int(built.sum()),
                    "ecological_grids": int((~built).sum()),
                    "total_grids": int(len(out)),
                    "built_share": float(built.mean()),
                    "ecological_share": float((~built).mean()),
                    "mean_built_ratio": float(np.nanmean(ratio)),
                    "median_built_ratio": float(np.nanmedian(ratio)),
                }
            )

        gpkg_path = out_dir / f"grid_gaia_raw_masks_ratio_{tag}.gpkg"
        if gpkg_path.exists():
            gpkg_path.unlink()
        out.to_file(gpkg_path, layer="grid_gaia_masks", driver="GPKG")
        dist = pd.DataFrame(rows)
        dist.to_csv(
            out_dir / f"gaia_year_mask_distribution_ratio_{tag}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        (out_dir / f"gaia_mask_metadata_ratio_{tag}.json").write_text(
            json.dumps(
                {
                    "gaia_raster": str(args.gaia_raster),
                    "grid_gpkg": str(args.grid_gpkg),
                    "coding": "value 1=1985, value 40=2024; built if built-pixel ratio >= threshold",
                    "years": years,
                    "built_ratio_threshold": thr,
                    "all_touched": args.all_touched,
                    "output_gpkg": str(gpkg_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("=" * 70)
        print(f"Threshold {thr:.2%} ({tag})")
        print(dist.to_string(index=False))
        print(f"Saved: {gpkg_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Clip TCULU China urban land-use rasters to Shenzhen and summarize areas.

The TCULU rasters are distributed in a Web-Mercator-like 10 m grid. Pixel
counts are therefore not directly equal-area. This script reports both raw
projected pixel area and latitude-corrected ground area; use the corrected
area for analysis.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--landuse-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/external/china_urban_landuse",
        help="Directory containing china_land_useYYYY.tif files.",
    )
    parser.add_argument(
        "--boundary",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/boundary/Hexagon_clip.shp",
        help="Shenzhen boundary/grid shapefile.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/tculu_shenzhen",
        help="Output directory.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2000, 2005, 2010, 2015, 2020, 2024],
        help="Years to process.",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Include all pixels touched by the boundary, not only pixel centers.",
    )
    return parser.parse_args()


def mercator_lat_from_y(y: np.ndarray) -> np.ndarray:
    """Convert Web Mercator y coordinates to latitude in radians."""
    radius = 6378137.0
    return 2.0 * np.arctan(np.exp(y / radius)) - math.pi / 2.0


def summarize_area(arr: np.ma.MaskedArray, transform, nodata: float | int | None) -> pd.DataFrame:
    """Summarize class areas using raw and latitude-corrected pixel areas."""
    data = arr.filled(0)
    valid = ~np.ma.getmaskarray(arr)
    if nodata is not None:
        valid &= data != nodata
    valid &= data != 0

    rows = np.arange(data.shape[0])
    y_centers = transform.f + (rows + 0.5) * transform.e
    lat_rad = mercator_lat_from_y(y_centers)
    # Web Mercator scale is sec(lat), so ground area = projected area * cos(lat)^2.
    raw_pixel_area_m2 = abs(transform.a * transform.e)
    corrected_row_area_m2 = raw_pixel_area_m2 * (np.cos(lat_rad) ** 2)

    records = []
    for code in sorted(CLASS_NAMES):
        class_mask = valid & (data == code)
        if not class_mask.any():
            pixel_count = 0
            raw_area_m2 = 0.0
            corrected_area_m2 = 0.0
        else:
            row_counts = class_mask.sum(axis=1)
            pixel_count = int(row_counts.sum())
            raw_area_m2 = float(pixel_count * raw_pixel_area_m2)
            corrected_area_m2 = float((row_counts * corrected_row_area_m2).sum())
        records.append(
            {
                "class_code": code,
                "class_name": CLASS_NAMES[code],
                "pixel_count": pixel_count,
                "area_raw_km2": raw_area_m2 / 1e6,
                "area_km2": corrected_area_m2 / 1e6,
            }
        )

    df = pd.DataFrame(records)
    total = df["area_km2"].sum()
    df["area_share_percent"] = np.where(total > 0, df["area_km2"] / total * 100.0, np.nan)
    return df


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    clip_dir = args.out_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    metadata = {
        "landuse_dir": str(args.landuse_dir),
        "boundary": str(args.boundary),
        "out_dir": str(args.out_dir),
        "years": args.years,
        "class_mapping": CLASS_NAMES,
        "area_method": "Web-Mercator latitude correction: projected pixel area * cos(latitude)^2",
        "all_touched": args.all_touched,
    }

    boundary = gpd.read_file(args.boundary)

    for year in args.years:
        raster_path = args.landuse_dir / f"china_land_use{year}.tif"
        if not raster_path.exists():
            raise FileNotFoundError(raster_path)

        with rasterio.open(raster_path) as src:
            # The source CRS is stored as LOCAL_CS but coordinates match EPSG:3857.
            dst_crs = CRS.from_epsg(3857)
            geom_gdf = boundary.to_crs(dst_crs)
            geometry = geom_gdf.geometry.union_all().__geo_interface__
            clipped, clipped_transform = mask(
                src,
                [geometry],
                crop=True,
                filled=False,
                all_touched=args.all_touched,
            )

            arr = clipped[0]
            profile = src.profile.copy()
            profile.update(
                {
                    "height": arr.shape[0],
                    "width": arr.shape[1],
                    "transform": clipped_transform,
                    "compress": "lzw",
                }
            )

            out_tif = clip_dir / f"shenzhen_tculu_{year}.tif"
            with rasterio.open(out_tif, "w", **profile) as dst:
                dst.write(arr.filled(src.nodata or 0).astype(profile["dtype"]), 1)

            df = summarize_area(arr, clipped_transform, src.nodata)
            df.insert(0, "year", year)
            rows.append(df)
            print(f"{year}: total corrected area = {df['area_km2'].sum():.3f} km2")

    area = pd.concat(rows, ignore_index=True)
    area.to_csv(args.out_dir / "shenzhen_tculu_area_by_class.csv", index=False, encoding="utf-8-sig")

    wide = area.pivot(index="year", columns="class_name", values="area_km2").reset_index()
    wide.to_csv(args.out_dir / "shenzhen_tculu_area_by_class_wide.csv", index=False, encoding="utf-8-sig")

    share = area.pivot(index="year", columns="class_name", values="area_share_percent").reset_index()
    share.to_csv(args.out_dir / "shenzhen_tculu_area_share_wide.csv", index=False, encoding="utf-8-sig")

    with open(args.out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()

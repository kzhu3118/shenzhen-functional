"""
GEE feature extraction for Shenzhen UFS backcasting years: 2000, 2005, 2010, 2015.

Goal:
  Export grid-level features with the SAME column names as the 2020 RF input:
    grid_id + 14 bands/features * 5 stats = 71 columns

Important consistency decisions:
  - Landsat 5/7/8 are harmonized to L8-like names:
      SR_B2=blue, SR_B3=green, SR_B4=red, SR_B5=NIR,
      SR_B6=SWIR1, SR_B7=SWIR2, LST=thermal K
  - 2000/2005/2010 use DMSP-OLS annual stable lights as NTL because VIIRS
    monthly data are unavailable before 2012.
  - 2015 uses VIIRS monthly median, matching the 2020 script.

Free-account quota strategy:
  - Submit one year per run by default.
  - Use CSV table export only, no raster export.
  - Increase tileScale if reduceRegions hits memory errors.

Usage examples:
  python gee_extract_features_2000_2015.py --year 2000
  python gee_extract_features_2000_2015.py --year 2005
  python gee_extract_features_2000_2015.py --year 2010
  python gee_extract_features_2000_2015.py --year 2015
  python gee_extract_features_2000_2015.py --all
"""

import argparse
import sys
import time

import ee


GRID_ASSET = "projects/ee-kzhu0908/assets/Shenzhen_Grid_2020"
GEE_PROJECT = "project-number-123456"  # Replace with your project number
EXPORT_DRIVE_FOLDER = "GEE_Shenzhen_UFS"
CLOUD_COVER_MAX = 30
TILE_SCALE = 4
YEARS = [2000, 2005, 2010, 2015]

CANONICAL_BANDS = [
    "SR_B2",
    "SR_B3",
    "SR_B4",
    "SR_B5",
    "SR_B6",
    "SR_B7",
    "NDVI",
    "NDBI",
    "MNDWI",
    "NBI",
    "LST",
    "NTL",
    "DEM",
    "slope",
]

STATS = ["mean", "stdDev", "min", "max", "median"]


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Shenzhen UFS features for 2000-2015.")
    parser.add_argument("--year", type=int, choices=YEARS, help="Single year to export.")
    parser.add_argument("--all", action="store_true", help="Submit all years one by one.")
    parser.add_argument("--project", default=GEE_PROJECT, help="GEE project for ee.Initialize.")
    parser.add_argument("--grid-asset", default=GRID_ASSET)
    parser.add_argument("--folder", default=EXPORT_DRIVE_FOLDER)
    parser.add_argument("--cloud-cover", type=float, default=CLOUD_COVER_MAX)
    parser.add_argument("--tile-scale", type=int, default=TILE_SCALE)
    parser.add_argument(
        "--ntl",
        choices=["mixed", "none"],
        default="mixed",
        help=(
            "mixed: DMSP for <=2010 and VIIRS for 2015; "
            "none: export NTL as a constant 0 band for consistency tests."
        ),
    )
    args = parser.parse_args()
    if not args.all and args.year is None:
        parser.error("Use --year YEAR or --all.")
    return args


def init_ee(project):
    print("[1/6] Initialize GEE...")
    try:
        ee.Initialize(project=project)
    except Exception as e:
        print(f"Failed to initialize GEE: {e}")
        sys.exit(1)
    print("  OK")


def mask_landsat_c2_l2(image, thermal_band):
    qa = image.select("QA_PIXEL")
    # C2 L2 QA_PIXEL bits:
    # bit 1 dilated cloud, bit 3 cloud, bit 4 cloud shadow, bit 5 snow.
    mask = (
        qa.bitwiseAnd(1 << 1)
        .eq(0)
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    )
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal = image.select(thermal_band).multiply(0.00341802).add(149.0)
    return image.addBands(optical, overwrite=True).addBands(thermal, overwrite=True).updateMask(mask)


def rename_l57_to_l8_like(image):
    old = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "ST_B6"]
    new = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10"]
    return image.select(old, new)


def rename_l8(image):
    return image.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10"])


def landsat_collection(year, roi, cloud_cover):
    start = ee.Date(f"{year}-01-01")
    end = ee.Date(f"{year}-12-31")

    if year <= 2011:
        # Landsat 5 is preferred for 2000/2005/2010 to avoid L7 SLC-off gaps.
        l5 = (
            ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
            .filterBounds(roi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", cloud_cover))
            .map(lambda img: mask_landsat_c2_l2(img, "ST_B6"))
            .map(rename_l57_to_l8_like)
        )
        return l5

    # 2015 uses Landsat 8, same source family as 2020.
    l8 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_cover))
        .map(lambda img: mask_landsat_c2_l2(img, "ST_B10"))
        .map(rename_l8)
    )
    return l8


def get_ntl(year, roi, mode):
    if mode == "none":
        return ee.Image.constant(0).rename("NTL").clip(roi)

    if year <= 2010:
        # DMSP-OLS stable lights are annual, digital number range roughly 0-63.
        # This is not directly comparable to VIIRS radiance, so downstream
        # sensitivity should be checked or a no-NTL RF model should be trained.
        dmsp_id = f"NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F{18 if year == 2010 else 15}{year}"
        dmsp = ee.Image(dmsp_id).select("stable_lights").rename("NTL").clip(roi)
        return dmsp

    start = ee.Date(f"{year}-01-01")
    end = ee.Date(f"{year}-12-31")
    viirs = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG")
        .filterDate(start, end)
        .select("avg_rad")
        .median()
        .rename("NTL")
        .clip(roi)
    )
    return viirs


def build_feature_image(year, roi, cloud_cover, ntl_mode):
    collection = landsat_collection(year, roi, cloud_cover)
    count = collection.size()
    print(f"  Landsat scenes for {year}:", count.getInfo())

    median = collection.median().clip(roi)
    optical = median.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])

    ndvi = median.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    ndbi = median.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")
    mndwi = median.normalizedDifference(["SR_B3", "SR_B6"]).rename("MNDWI")
    nbi = (
        median.select("SR_B4")
        .multiply(median.select("SR_B6"))
        .divide(median.select("SR_B5").add(0.0001))
        .rename("NBI")
    )
    lst = median.select("ST_B10").rename("LST")
    ntl = get_ntl(year, roi, ntl_mode)
    dem = ee.Image("USGS/SRTMGL1_003").clip(roi).rename("DEM")
    slope = ee.Terrain.slope(dem).rename("slope")

    image = ee.Image.cat([optical, ndvi, ndbi, mndwi, nbi, lst, ntl, dem, slope])
    image = image.select(CANONICAL_BANDS)
    return image


def export_year(year, grid, roi, args):
    print("=" * 70)
    print(f"Build and export features for {year}")
    print("=" * 70)

    image = build_feature_image(year, roi, args.cloud_cover, args.ntl)
    bands = image.bandNames().getInfo()
    print(f"  Feature bands ({len(bands)}): {bands}")

    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.min(), sharedInputs=True)
        .combine(ee.Reducer.max(), sharedInputs=True)
        .combine(ee.Reducer.median(), sharedInputs=True)
    )

    sampled = image.reduceRegions(
        collection=grid,
        reducer=reducer,
        scale=30,
        tileScale=args.tile_scale,
    )

    selectors = ["grid_id"]
    for band in CANONICAL_BANDS:
        for stat in STATS:
            selectors.append(f"{band}_{stat}")

    suffix = "noNTL" if args.ntl == "none" else "mixedNTL"
    file_name = f"shenzhen_features_{year}_{suffix}"
    task = ee.batch.Export.table.toDrive(
        collection=sampled,
        description=file_name,
        folder=args.folder,
        fileNamePrefix=file_name,
        fileFormat="CSV",
        selectors=selectors,
    )
    task.start()
    print(f"  Submitted task: {file_name}")
    print(f"  Task ID: {task.id}")
    print(f"  Output: Drive/{args.folder}/{file_name}.csv")


def main():
    args = parse_args()
    init_ee(args.project)

    print("[2/6] Load grid...")
    grid = ee.FeatureCollection(args.grid_asset)
    n_grid = grid.size().getInfo()
    print(f"  Grid count: {n_grid}")
    roi = grid.geometry().bounds()

    years = YEARS if args.all else [args.year]
    print(f"[3/6] Years to submit: {years}")
    print(f"  NTL mode: {args.ntl}")
    print(f"  Cloud cover max: {args.cloud_cover}")
    print(f"  tileScale: {args.tile_scale}")

    for i, year in enumerate(years, start=1):
        export_year(year, grid, roi, args)
        # Avoid submitting many tasks in the same second.
        if i < len(years):
            time.sleep(5)

    print("=" * 70)
    print("All requested export tasks have been submitted.")
    print("For a free GEE account, let 1-2 tasks finish before submitting many more.")
    print("Check progress in the GEE Code Editor Tasks tab or with earthengine task list.")
    print("=" * 70)


if __name__ == "__main__":
    main()

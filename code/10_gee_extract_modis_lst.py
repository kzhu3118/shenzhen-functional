"""GEE MODIS LST extraction for five summer periods.

Uses Terra/Aqua MODIS 8-day LST composites to export summer LST statistics for
2000, 2005, 2010, 2015, and 2020.
"""

import argparse
import ee
import sys


GEE_PROJECT = 'project-number-123456'  # Replace with your project number
GRID_ASSET_DEFAULT = 'projects/ee-kzhu0908/assets/Shenzhen_Grid_2020'
EXPORT_FOLDER_DEFAULT = 'GEE_Shenzhen_UFS'
YEARS_DEFAULT = [2000, 2005, 2010, 2015, 2020]


def parse_args():
    parser = argparse.ArgumentParser(description='Extract summer MODIS LST statistics from GEE.')
    parser.add_argument('--project', default=GEE_PROJECT)
    parser.add_argument('--grid-asset', default=GRID_ASSET_DEFAULT)
    parser.add_argument('--folder', default=EXPORT_FOLDER_DEFAULT)
    parser.add_argument('--years', default=','.join(str(y) for y in YEARS_DEFAULT))
    parser.add_argument('--tile-scale', type=int, default=2)
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Build images and print sample output properties without submitting exports.',
    )
    return parser.parse_args()


def parse_years(text):
    return [int(x.strip()) for x in text.split(',') if x.strip()]

# ============================================================
# 1. Initialize
# ============================================================
print("=" * 70)
print("MODIS LST extraction for five summer periods")
print("=" * 70)

args = parse_args()
try:
    ee.Initialize(project=args.project)
except Exception as e:
    print(f"Initializing GEE failed: {e}")
    sys.exit(1)


# ============================================================
# 2. Configuration
# ============================================================
GRID_ASSET = args.grid_asset
EXPORT_FOLDER = args.folder
YEARS = parse_years(args.years)
TILE_SCALE = args.tile_scale

LST_SELECTORS = [
    'grid_id',
    'LST_mean',
    'LST_stdDev',
    'LST_min',
    'LST_max',
    'LST_median',
]

# Summer window for Shenzhen hot season
SUMMER_START_MMDD = '-06-01'
SUMMER_END_MMDD = '-09-30'

# QA threshold for LST_Day_1km quality bits.
# Accept good-quality MODIS LST pixels.


# ============================================================
# 3. Read grid and ROI
# ============================================================
grid = ee.FeatureCollection(GRID_ASSET)
n_grid = grid.size().getInfo()
print(f"\nGrid count: {n_grid}")
roi = grid.geometry().bounds()


# ============================================================
# 4. MODIS LST processing functions
# ============================================================
def mask_modis_lst(image):
    """
    MODIS 8-day LST composites (MOD11A2 / MYD11A2):
        - LST_Day_1km: 8-day daytime LST, unit 0.02 K
        - QC_Day:      QA flags
        - Keep QA bits 0-1 equal to 00 or 01.

    Returns LST in Kelvin as a single band named 'LST'.
    """
    qc = image.select('QC_Day')
    # Bits 0-1 encode LST quality: 00 good, 01 other quality, 10 cloud, 11 other missing.
    qa_mask = qc.bitwiseAnd(3).lte(1)  # Accept 00 and 01

    lst_kelvin = image.select('LST_Day_1km').multiply(0.02).updateMask(qa_mask)
    return lst_kelvin.rename('LST').copyProperties(image, ['system:time_start'])


def get_modis_lst_summer(year):
    """
    Summer MODIS LST composite:
    - 2000-2001: Terra only; Aqua starts in May 2002.
    - 2002+: combined Terra and Aqua.
    """
    start = ee.Date(f'{year}{SUMMER_START_MMDD}')
    end = ee.Date(f'{year}{SUMMER_END_MMDD}')

    terra = (ee.ImageCollection('MODIS/061/MOD11A2')
             .filterBounds(roi)
             .filterDate(start, end)
             .map(mask_modis_lst))

    if year >= 2002:
        aqua = (ee.ImageCollection('MODIS/061/MYD11A2')
                .filterBounds(roi)
                .filterDate(start, end)
                .map(mask_modis_lst))
        merged = terra.merge(aqua)
        source = 'Terra+Aqua'
    else:
        merged = terra
        source = 'Terra only'

    n_obs = merged.size().getInfo()
    print(f"  MODIS LST {year} summer composite observations: {n_obs} ({source})")

    if n_obs == 0:
        return None

    # Median composite for robustness to residual cloud.
    lst_summer = merged.median().clip(roi)
    return lst_summer


# ============================================================
# 5. Main workflow for five years
# ============================================================
print("\nSubmitting export tasks...\n")


def rename_lst_stats(feature):
    """Single-band reduceRegions returns mean/stdDev/... property names."""
    return feature.set({
        'LST_mean': feature.get('mean'),
        'LST_stdDev': feature.get('stdDev'),
        'LST_min': feature.get('min'),
        'LST_max': feature.get('max'),
        'LST_median': feature.get('median'),
    })


for year in YEARS:
    print(f"--- {year} ---")
    lst_image = get_modis_lst_summer(year)
    if lst_image is None:
        print(f"  ⚠ {year} no valid observations; skipped")
        continue

    # reduceRegions calculates five statistics.
    combined_reducer = (ee.Reducer.mean()
                        .combine(ee.Reducer.stdDev(), sharedInputs=True)
                        .combine(ee.Reducer.min(), sharedInputs=True)
                        .combine(ee.Reducer.max(), sharedInputs=True)
                        .combine(ee.Reducer.median(), sharedInputs=True))

    # MODIS is 1000 m, reduced over the 200 m grid cells.
    # tileScale=2 is sufficient for the small MODIS workload.
    sampled = lst_image.reduceRegions(
        collection=grid,
        reducer=combined_reducer,
        scale=1000,         # Native MODIS resolution
        tileScale=TILE_SCALE,
    ).map(rename_lst_stats)

    sample_props = sampled.first().toDictionary(LST_SELECTORS).getInfo()
    print(f"  Example export fields: {sample_props}")

    file_name = f'shenzhen_LST_MODIS_{year}'
    if args.dry_run:
        print(f"  dry-run: not submitting export task {file_name}")
        continue

    task = ee.batch.Export.table.toDrive(
        collection=sampled,
        description=file_name,
        folder=EXPORT_FOLDER,
        fileNamePrefix=file_name,
        fileFormat='CSV',
        selectors=LST_SELECTORS,
    )
    task.start()
    print(f"  ✓ Task submitted: {file_name}")
    print(f"    Task ID: {task.id}")


print("\n" + "=" * 70)
if args.dry_run:
    print("Dry-run check complete; no GEE export task was submitted.")
else:
    print("All tasks submitted")
print("=" * 70)
if not args.dry_run:
    print("\nCheck progress:")
    print("  Browser: https://code.earthengine.google.com → Tasks tab")
    print("\nEstimated runtime: 2-5 minutes per year; MODIS is faster than Landsat.")
    print("Total: 10-25 minutes")
    print("\nMODIS LST unit note:")
    print("  Exported values are Kelvin (after × 0.02 scaling)")
    print("  For local use, subtract - 273.15 to convert to Celsius")
    print("\nNext: download the five CSV files and run diagnostics/revised thermal analysis.")

"""GEE feature extraction for the 2020 grid.

Exports Landsat, spectral index, LST, nightlight, DEM, and slope statistics for
each Shenzhen grid cell to Google Drive.
"""

import ee
import sys

# ============================================================
# 1. Initialize
# ============================================================
print("=" * 70)
print("Step 3a: GEE feature extraction (2020)")
print("=" * 70)

print("\n[1/6] Initializing GEE...")
try:
    # If ee.Initialize() requires a project, use the project-specific form below.
    # ee.Initialize(project='ee-kzhu0908')
    ee.Initialize(project = 'project-number-123456')  # Replace with your project number
    print("    ✓ Initialization succeeded")
except Exception as e:
    print(f"    ✗ failed: {e}")
    print("\n    If a project is required, update this line: ee.Initialize(project='ee-kzhu0908')")
    sys.exit(1)


# ============================================================
# 2. Configuration
# ============================================================
GRID_ASSET = 'projects/ee-kzhu0908/assets/Shenzhen_Grid_2020'  # GEE asset
YEAR = 2020
EXPORT_DRIVE_FOLDER = 'GEE_Shenzhen_UFS'   # Target Drive folder, created automatically if needed
EXPORT_FILE_NAME = f'shenzhen_features_{YEAR}'

# Cloud-cover threshold
CLOUD_COVER_MAX = 20

print(f"\n[2/6] Configuration:")
print(f"    Grid asset: {GRID_ASSET}")
print(f"    Year:       {YEAR}")
print(f"    Export to:     Drive/{EXPORT_DRIVE_FOLDER}/{EXPORT_FILE_NAME}.csv")


# ============================================================
# 3. Read grid and ROI
# ============================================================
print(f"\n[3/6] Reading grid asset...")
try:
    grid = ee.FeatureCollection(GRID_ASSET)
    n_grid = grid.size().getInfo()
    print(f"    ✓ Grid count: {n_grid}")
    if n_grid != 19727:
        print(f"    Note: expected 19727 grid cells, actual {n_grid}")
except Exception as e:
    print(f"    ✗ read failed: {e}")
    sys.exit(1)

# Use the grid bounds as ROI to avoid insufficient data in individual cells.
roi = grid.geometry().bounds()


# ============================================================
# 4. Build the 2020 feature image
# ============================================================
print(f"\n[4/6] Building feature image...")

start_date = ee.Date(f'{YEAR}-01-01')
end_date = ee.Date(f'{YEAR}-12-31')

# --- Landsat 8 SR ---
def mask_l8sr(image):
    """Apply Landsat 8 Collection 2 Level 2 cloud/shadow mask and scale factors."""
    qa = image.select('QA_PIXEL')
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 5).eq(0))
    optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal = image.select('ST_B10').multiply(0.00341802).add(149.0)
    return image.addBands(optical, overwrite=True) \
                .addBands(thermal, overwrite=True) \
                .updateMask(cloud_mask)

l8_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                 .filterBounds(roi)
                 .filterDate(start_date, end_date)
                 .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_MAX))
                 .map(mask_l8sr))

# Annual median composite to reduce cloud effects.
l8_median = l8_collection.median().clip(roi)

# Six optical bands
l8_bands = l8_median.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])

# Four indices
ndvi = l8_median.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
ndbi = l8_median.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
mndwi = l8_median.normalizedDifference(['SR_B3', 'SR_B6']).rename('MNDWI')
# NDISI: an additional impervious-surface index
# NDISI = (TIR - (MNDWI + NIR + SWIR1) / 3) / (TIR + (MNDWI + NIR + SWIR1) / 3)
# Simplified NBI = (Red * SWIR1) / NIR
nbi_num = l8_median.select('SR_B4').multiply(l8_median.select('SR_B6'))
nbi_den = l8_median.select('SR_B5').add(0.0001)
nbi = nbi_num.divide(nbi_den).rename('NBI')

# LST in Kelvin
lst = l8_median.select('ST_B10').rename('LST')

# --- NTL from monthly VIIRS ---
viirs_collection = (ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG')
                    .filterDate(start_date, end_date)
                    .select('avg_rad'))
ntl = viirs_collection.median().clip(roi).rename('NTL')

# --- DEM and slope ---
dem = ee.Image('USGS/SRTMGL1_003').clip(roi).rename('DEM')
slope = ee.Terrain.slope(dem).rename('slope')

# --- Combine all features ---
features_image = ee.Image.cat([
    l8_bands,    # SR_B2-B7
    ndvi, ndbi, mndwi, nbi,  # Four indices
    lst,         # LST
    ntl,         # Nighttime lights
    dem, slope,  # Terrain
])

# Print band names for inspection
band_names = features_image.bandNames().getInfo()
print(f"    Feature bands ({len(band_names)}): {band_names}")


# ============================================================
# 5. Use reduceRegions to calculate five statistics
# ============================================================
print(f"\n[5/6] Preparing reduceRegions...")

# Combine five reducers
combined_reducer = (ee.Reducer.mean()
                    .combine(ee.Reducer.stdDev(), sharedInputs=True)
                    .combine(ee.Reducer.min(), sharedInputs=True)
                    .combine(ee.Reducer.max(), sharedInputs=True)
                    .combine(ee.Reducer.median(), sharedInputs=True))

# reduceRegions: calculate statistics for each grid cell
sampled = features_image.reduceRegions(
    collection=grid,
    reducer=combined_reducer,
    scale=30,           # 30 m resolution
    tileScale=4,        # Reduce tile workload to avoid memory errors
)

# Fields will be named like SR_B2_mean, SR_B2_stdDev, SR_B2_min, ..., NTL_mean, ..., slope_median.
# Existing grid fields such as grid_id are retained automatically.


# ============================================================
# 6. Submit the export task
# ============================================================
print(f"\n[6/6] Submitting export task to Drive...")

# Select export fields
all_stats = ['mean', 'stdDev', 'min', 'max', 'median']
selectors = ['grid_id']  # Keep grid_id for local joins
for band in band_names:
    for stat in all_stats:
        selectors.append(f'{band}_{stat}')

print(f"    Exporting {len(selectors)} columns (1 grid_id + {len(band_names)} bands × {len(all_stats)} stats)")

task = ee.batch.Export.table.toDrive(
    collection=sampled,
    description=EXPORT_FILE_NAME,
    folder=EXPORT_DRIVE_FOLDER,
    fileNamePrefix=EXPORT_FILE_NAME,
    fileFormat='CSV',
    selectors=selectors,
)

task.start()
print(f"    ✓ Task submitted")
print(f"    Task ID: {task.id}")
print(f"    Task name:  {EXPORT_FILE_NAME}")

print(f"\n{'=' * 70}")
print("The task has been submitted to the GEE server.")
print(f"{'=' * 70}")
print(f"\nCheck progress:")
print(f"  Method 1: open in browser https://code.earthengine.google.com")
print(f"          → right-side Tasks tab")
print(f"  Method 2: run in terminal 'earthengine task list' (list all tasks)")
print("          or: earthengine task info <task_id>")
print("\nEstimated runtime: 20-40 minutes")
print(f"After completion, the CSV will be in Google Drive: {EXPORT_DRIVE_FOLDER}/")
print(f"\nNext: wait for completion and download the CSV for local model training.")
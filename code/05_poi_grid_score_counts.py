"""Grid-level POI scoring script.

Aggregates cleaned POI subclasses to the hexagonal grid and assigns preliminary
functional labels based on class proportions and density thresholds.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PathsConfiguration
# ============================================================
POI_GPKG = Path('data/poi_2020/poi_clean/shenzhen_poi_clean_2020.gpkg')
GRID_SHP = Path('data/boundary/Hexagon_clip.shp')
OUT_DIR = Path('data/poi_2020/poi_clean')
OUT_GPKG = OUT_DIR / 'grid_scored_2020.gpkg'
OUT_CSV = OUT_DIR / 'grid_summary.csv'

# ============================================================
# Parameters
# ============================================================
MAIN_CLASSES = ['industrial', 'residential', 'commercial', 'public_service']
MIN_POI_COUNT = 3        # total POI < 3 is labeled low_density
DOMINANT_THRESHOLD = 0.6 # >=0.6 indicates a single dominant function
MIXED_THRESHOLD = 0.4    # 0.4-0.6 → X-dominated mixed; <0.4 → multi-functional


def shannon_entropy(ratios):
    """Calculate Shannon entropy from four class proportions."""
    r = np.array(ratios, dtype=float)
    r = r[r > 0]
    if len(r) == 0:
        return 0.0
    return float(-(r * np.log(r + 1e-12)).sum())


def classify_grid(row):
    """
    Assign labels to one grid cell.

    Returns four fields:
      ples_class:     production / living / ecological / low_density
      ples_subclass:  industrial / residential / commercial / public_service /
                      <X>-dominated_mixed / multi_functional_mixed / low_density
      dominant: dominant subclass name, or none if low_density
      label_confidence: dominant proportion value from 0 to 1
    """
    total = row['total_main_poi']

    # Low-density grid cells
    if total < MIN_POI_COUNT:
        return pd.Series({
            'ples_class': 'low_density',
            'ples_subclass': 'low_density',
            'dominant': None,
            'label_confidence': 0.0,
        })

    # Get four proportions
    ratios = {c: row[f'r_{c}'] for c in MAIN_CLASSES}
    dominant_cls = max(ratios, key=ratios.get)
    dominant_ratio = ratios[dominant_cls]

    # PLES class mapping
    ples_map = {
        'industrial': 'production',
        'residential': 'living',
        'commercial': 'living',
        'public_service': 'living',
    }

    # Two-threshold rule
    if dominant_ratio >= DOMINANT_THRESHOLD:
        # Single-function label
        return pd.Series({
            'ples_class': ples_map[dominant_cls],
            'ples_subclass': dominant_cls,
            'dominant': dominant_cls,
            'label_confidence': dominant_ratio,
        })
    elif dominant_ratio >= MIXED_THRESHOLD:
        # X-dominated mixed
        return pd.Series({
            'ples_class': ples_map[dominant_cls],
            'ples_subclass': f"{dominant_cls}_dominated_mixed",
            'dominant': dominant_cls,
            'label_confidence': dominant_ratio,
        })
    else:
        # multi-functional mixed
        return pd.Series({
            'ples_class': 'living',  # Multifunctional mixed grids default to living in this workflow.
            'ples_subclass': 'multi_functional_mixed',
            'dominant': dominant_cls,
            'label_confidence': dominant_ratio,
        })


def main():
    # Check inputs
    if not POI_GPKG.exists():
        print(f"POI GeoPackage does not exist: {POI_GPKG}")
        return
    if not GRID_SHP.exists():
        print(f"Grid shapefile does not exist: {GRID_SHP}")
        return

    OUT_DIR.mkdir(exist_ok=True, parents=True)

    # === Read grid ===
    print(f"Reading grid: {GRID_SHP}")
    grid = gpd.read_file(GRID_SHP)
    print(f"   Grid count: {len(grid)}")
    print(f"   CRS:    {grid.crs}")
    print(f"   Geometry type: {grid.geometry.geom_type.iloc[0]}")

    # Detect polygon shape
    sample_geom = grid.geometry.iloc[0]
    n_vertices = len(sample_geom.exterior.coords) - 1
    if n_vertices == 6:
        print("   Shape: hexagonal")
    elif n_vertices == 4:
        print("   Shape: rectangular")
    else:
        print(f"   Shape: polygon with {n_vertices} vertices")

    # Mean area calculated in projected CRS
    grid_proj = grid.to_crs('EPSG:32649')  # UTM Zone 49N for Shenzhen
    avg_area = grid_proj.geometry.area.mean()
    print(f"   Mean area: {avg_area:.0f} m2, about {np.sqrt(avg_area):.0f} m x {np.sqrt(avg_area):.0f} m")

    # Add grid_id
    if 'grid_id' not in grid.columns:
        grid = grid.reset_index(drop=True)
        grid['grid_id'] = grid.index
    grid_for_join = grid[['grid_id', 'geometry']].copy()

    # === Read each POI subclass and spatially join to grid ===
    print("\nReading POI layers and running spatial joins...")

    # Initialize count columns
    df = pd.DataFrame({'grid_id': grid['grid_id']})
    for cls in MAIN_CLASSES + ['ecological']:
        df[f'n_{cls}'] = 0

    for cls in MAIN_CLASSES + ['ecological']:
        print(f"   - {cls} ...", end=' ', flush=True)
        try:
            poi = gpd.read_file(POI_GPKG, layer=cls)
        except Exception as e:
            print(f"read failed: {e}")
            continue

        # Align CRS
        if poi.crs != grid_for_join.crs:
            poi = poi.to_crs(grid_for_join.crs)

        # Spatial join
        joined = gpd.sjoin(poi[['geometry']], grid_for_join,
                          predicate='within', how='inner')
        cnt = joined.groupby('grid_id').size()
        df[f'n_{cls}'] = df['grid_id'].map(cnt).fillna(0).astype(int)
        print(f"{len(poi)} POIs fall in {(df[f'n_{cls}'] > 0).sum()} grid cells")

    # === Calculate proportions ===
    print("\nCalculating four main-class proportions...")
    df['total_main_poi'] = df[[f'n_{c}' for c in MAIN_CLASSES]].sum(axis=1)
    for cls in MAIN_CLASSES:
        df[f'r_{cls}'] = np.where(
            df['total_main_poi'] > 0,
            df[f'n_{cls}'] / df['total_main_poi'],
            0.0,
        )

    # Ecological boolean marker
    df['has_ecological'] = (df['n_ecological'] > 0).astype(int)

    # === Mixed entropy ===
    print("Calculating mixed entropy...")
    ratio_cols = [f'r_{c}' for c in MAIN_CLASSES]
    df['mix_entropy'] = df[ratio_cols].apply(
        lambda row: shannon_entropy(row.values), axis=1
    )
    # Normalize to [0, 1]; log(4) is maximum entropy for four equal classes.
    df['mix_entropy_norm'] = df['mix_entropy'] / np.log(4)

    # === Classification ===
    print("Applying two-threshold classification...")
    cls_result = df.apply(classify_grid, axis=1)
    df = pd.concat([df, cls_result], axis=1)

    # === Statistics ===
    print(f"\n{'=' * 70}")
    print("Classification summary")
    print(f"{'=' * 70}")

    n_total = len(df)
    print("\n  PLES class distribution:")
    for ples in df['ples_class'].value_counts().items():
        cls, cnt = ples
        pct = cnt / n_total * 100
        # Estimate area
        area_km2 = cnt * avg_area / 1e6
        print(f"    {cls:<14} {cnt:>7d}  ({pct:>5.1f}%)  about {area_km2:>6.1f} km2")

    print("\n  Subclass distribution:")
    for sub in df['ples_subclass'].value_counts().items():
        cls, cnt = sub
        pct = cnt / n_total * 100
        print(f"    {cls:<35} {cnt:>7d}  ({pct:>5.1f}%)")

    # Industrial grid ground-truth information
    n_pure_industrial = (df['ples_subclass'] == 'industrial').sum()
    n_industrial_dominated = (df['ples_subclass'] == 'industrial_dominated_mixed').sum()
    print("\n  Industrial grid ground truth for model training:")
    print(f"    Pure industrial (>=0.6): {n_pure_industrial}")
    print(f"    Industrial-dominated (0.4-0.6): {n_industrial_dominated}")
    print(f"    Total usable industrial training samples: {n_pure_industrial + n_industrial_dominated}")

    # Ecological marker
    n_with_eco = df['has_ecological'].sum()
    print(f"\n  Grid cells containing ecological POIs: {n_with_eco}")

    # === Write outputs ===
    print("\nMerging geometry and writing GeoPackage...")
    result = grid.merge(df, on='grid_id', how='left')

    # Handle empty grid cells defensively
    for col in [f'n_{c}' for c in MAIN_CLASSES + ['ecological']] + ['total_main_poi']:
        result[col] = result[col].fillna(0).astype(int)
    for col in [f'r_{c}' for c in MAIN_CLASSES] + ['mix_entropy', 'mix_entropy_norm']:
        result[col] = result[col].fillna(0.0)
    result['ples_class'] = result['ples_class'].fillna('low_density')
    result['ples_subclass'] = result['ples_subclass'].fillna('low_density')

    # Remove old file
    if OUT_GPKG.exists():
        OUT_GPKG.unlink()
    result.to_file(OUT_GPKG, layer='grid_scored', driver='GPKG')
    print(f"   ✓ {OUT_GPKG}")

    # Export industrial ground-truth layer for model training.
    industrial_gt = result[result['ples_subclass'].isin(
        ['industrial', 'industrial_dominated_mixed'])].copy()
    industrial_gt.to_file(OUT_GPKG, layer='industrial_ground_truth', driver='GPKG')
    print(f"   industrial_ground_truth layer: {len(industrial_gt)} grid cells")

    # === Write CSV summary ===
    summary = pd.DataFrame({
        'category': ['total_grids'] +
                   [f"PLES_{c}" for c in df['ples_class'].value_counts().index] +
                   [f"sub_{c}" for c in df['ples_subclass'].value_counts().index],
        'count': [n_total] +
                df['ples_class'].value_counts().tolist() +
                df['ples_subclass'].value_counts().tolist(),
    })
    summary['percent'] = (summary['count'] / n_total * 100).round(2)
    summary.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f"   ✓ {OUT_CSV}")

    print(f"\n{'=' * 70}")
    print("Step 2 complete. Suggested next checks:")
    print(f"  1. Open {OUT_GPKG.name} in QGIS and inspect spatial patterns.")
    print("  2. Check whether industrial grids cluster in Baoan/Longgang.")
    print("  3. Check whether mixed grids appear in urban villages/CBD areas.")
    print("  4. Then continue to model-training sample generation.")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
"""Frequency-density normalization for POI grid scores.

The original raw-count proportions overemphasize abundant commercial POIs. This
script applies frequency-density normalization before assigning functional labels.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Paths
# ============================================================
IN_GPKG = Path('data/poi_2020/poi_clean/grid_scored_2020.gpkg')
OUT_GPKG = Path('data/poi_2020/poi_clean/grid_scored_2020_FD.gpkg')
OUT_CSV = Path('data/poi_2020/poi_clean/grid_summary_FD.csv')

# ============================================================
# Parameters
# ============================================================
MAIN_CLASSES = ['industrial', 'residential', 'commercial', 'public_service']
MIN_POI_COUNT = 3
DOMINANT_THRESHOLD = 0.6
MIXED_THRESHOLD = 0.4


def shannon_entropy(ratios):
    r = np.array(ratios, dtype=float)
    r = r[r > 0]
    if len(r) == 0:
        return 0.0
    return float(-(r * np.log(r + 1e-12)).sum())


def classify_grid(row):
    total = row['total_main_poi']
    if total < MIN_POI_COUNT:
        return pd.Series({
            'ples_class': 'low_density',
            'ples_subclass': 'low_density',
            'dominant': None,
            'label_confidence': 0.0,
        })

    # Use FD proportions rather than raw proportions to determine the dominant class.
    ratios = {c: row[f'rfd_{c}'] for c in MAIN_CLASSES}
    dominant_cls = max(ratios, key=ratios.get)
    dominant_ratio = ratios[dominant_cls]

    ples_map = {
        'industrial': 'production',
        'residential': 'living',
        'commercial': 'living',
        'public_service': 'living',
    }

    if dominant_ratio >= DOMINANT_THRESHOLD:
        return pd.Series({
            'ples_class': ples_map[dominant_cls],
            'ples_subclass': dominant_cls,
            'dominant': dominant_cls,
            'label_confidence': dominant_ratio,
        })
    elif dominant_ratio >= MIXED_THRESHOLD:
        return pd.Series({
            'ples_class': ples_map[dominant_cls],
            'ples_subclass': f"{dominant_cls}_dominated_mixed",
            'dominant': dominant_cls,
            'label_confidence': dominant_ratio,
        })
    else:
        return pd.Series({
            'ples_class': 'living',
            'ples_subclass': 'multi_functional_mixed',
            'dominant': dominant_cls,
            'label_confidence': dominant_ratio,
        })


def main():
    if not IN_GPKG.exists():
        print(f"Input does not exist: {IN_GPKG}")
        return

    print(f"Reading Step 2 result: {IN_GPKG.name}")
    grid = gpd.read_file(IN_GPKG, layer='grid_scored')
    n_total = len(grid)
    print(f"   Grid count: {n_total}")

    # ============ Calculate FD coefficients ============
    print("\nCalculating frequency-density normalization coefficients...")
    print(f"{'Class':<18} {'POI total':<10} {'Grids with POI':<14} {'Mean density':<12}")
    print("-" * 60)

    fd_coef = {}
    for cls in MAIN_CLASSES:
        n_total_class = grid[f'n_{cls}'].sum()
        n_grid_with_class = (grid[f'n_{cls}'] > 0).sum()
        if n_grid_with_class == 0:
            fd_coef[cls] = 1.0
        else:
            fd_coef[cls] = n_total_class / n_grid_with_class
        print(f"  {cls:<16} {n_total_class:<10d} {n_grid_with_class:<12d} {fd_coef[cls]:<10.3f}")

    print(f"\n  Note: industrial mean density = {fd_coef['industrial']:.2f} POIs/grid")
    print(f"        commercial mean density = {fd_coef['commercial']:.2f} POIs/grid")
    print(f"        After FD normalization, one industrial POI represents about {fd_coef['commercial']/fd_coef['industrial']:.1f} commercial POIs")

    # ============ Calculate FD ============
    print("\nCalculating FD values and normalized proportions...")
    for cls in MAIN_CLASSES:
        # FD_i = n_i / mean_density_i
        grid[f'fd_{cls}'] = grid[f'n_{cls}'] / fd_coef[cls]

    # FD total
    fd_cols = [f'fd_{c}' for c in MAIN_CLASSES]
    grid['total_fd'] = grid[fd_cols].sum(axis=1)

    # Normalize FD proportions
    for cls in MAIN_CLASSES:
        grid[f'rfd_{cls}'] = np.where(
            grid['total_fd'] > 0,
            grid[f'fd_{cls}'] / grid['total_fd'],
            0.0,
        )

    # ============ Mixed entropy based on FD proportions ============
    print("Calculating FD-based mixed entropy...")
    rfd_cols = [f'rfd_{c}' for c in MAIN_CLASSES]
    grid['mix_entropy'] = grid[rfd_cols].apply(
        lambda row: shannon_entropy(row.values), axis=1
    )
    grid['mix_entropy_norm'] = grid['mix_entropy'] / np.log(4)

    # ============ Reclassification ============
    print("Applying two-threshold classification based on FD proportions...")
    cls_result = grid.apply(classify_grid, axis=1)
    # Remove old classification columns
    for col in ['ples_class', 'ples_subclass', 'dominant', 'label_confidence']:
        if col in grid.columns:
            grid = grid.drop(columns=[col])
    grid = pd.concat([grid, cls_result], axis=1)

    # ============ Statistics ============
    avg_area_m2 = 99625  # Known from Step 2
    print(f"\n{'=' * 70}")
    print("FD-normalized classification results")
    print(f"{'=' * 70}")

    print("\n  PLES class distribution:")
    for ples in grid['ples_class'].value_counts().items():
        cls, cnt = ples
        pct = cnt / n_total * 100
        area = cnt * avg_area_m2 / 1e6
        print(f"    {cls:<14} {cnt:>7d}  ({pct:>5.1f}%)  about {area:>6.1f} km2")

    print("\n  Subclass distribution:")
    for sub in grid['ples_subclass'].value_counts().items():
        cls, cnt = sub
        pct = cnt / n_total * 100
        print(f"    {cls:<35} {cnt:>7d}  ({pct:>5.1f}%)")

    n_pure_industrial = (grid['ples_subclass'] == 'industrial').sum()
    n_industrial_dominated = (grid['ples_subclass'] == 'industrial_dominated_mixed').sum()
    print("\n  Industrial grid ground truth:")
    print(f"    Pure industrial (>=0.6): {n_pure_industrial}")
    print(f"    Industrial-dominated (0.4-0.6): {n_industrial_dominated}")
    print(f"    Total usable industrial training samples: {n_pure_industrial + n_industrial_dominated}")

    n_pure_residential = (grid['ples_subclass'] == 'residential').sum()
    n_residential_dominated = (grid['ples_subclass'] == 'residential_dominated_mixed').sum()
    print("\n  Residential grid ground truth:")
    print(f"    Pure residential (>=0.6): {n_pure_residential}")
    print(f"    Residential-dominated (0.4-0.6): {n_residential_dominated}")
    print(f"    Total: {n_pure_residential + n_residential_dominated}")

    # ============ Write outputs ============
    print("\nWriting GeoPackage...")
    if OUT_GPKG.exists():
        OUT_GPKG.unlink()
    grid.to_file(OUT_GPKG, layer='grid_scored', driver='GPKG')

    industrial_gt = grid[grid['ples_subclass'].isin(
        ['industrial', 'industrial_dominated_mixed'])].copy()
    industrial_gt.to_file(OUT_GPKG, layer='industrial_ground_truth', driver='GPKG')
    print(f"   ✓ {OUT_GPKG.name}")
    print(f"   industrial_ground_truth: {len(industrial_gt)} grid cells")

    # Write FD coefficients to CSV for methods reporting
    fd_df = pd.DataFrame([
        {'class': c,
         'total_poi': int(grid[f'n_{c}'].sum()),
         'grids_with_poi': int((grid[f'n_{c}'] > 0).sum()),
         'mean_density_per_grid': round(fd_coef[c], 4)}
        for c in MAIN_CLASSES
    ])
    fd_df.to_csv(OUT_CSV.parent / 'fd_coefficients.csv', index=False, encoding='utf-8-sig')
    print("   FD coefficients saved: fd_coefficients.csv")

    # Overall summary
    summary = pd.DataFrame({
        'category': ['total_grids'] +
                   [f"PLES_{c}" for c in grid['ples_class'].value_counts().index] +
                   [f"sub_{c}" for c in grid['ples_subclass'].value_counts().index],
        'count': [n_total] +
                grid['ples_class'].value_counts().tolist() +
                grid['ples_subclass'].value_counts().tolist(),
    })
    summary['percent'] = (summary['count'] / n_total * 100).round(2)
    summary.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    print(f"\n{'=' * 70}")
    print("Done. Suggested QGIS checks:")
    print("  1. Industrial grids should cluster in major industrial zones.")
    print("  2. Residential grids should cover major residential areas.")
    print("  3. Commercial grids should appear in Dongmen/Huaqiangbei/CBD.")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
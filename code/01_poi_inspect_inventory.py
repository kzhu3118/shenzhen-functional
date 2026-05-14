"""POI inventory and quality inspection script (read-only).

Purpose: inspect the raw/sampled POI files before cleaning.
Outputs: console report and CSV reports under data/inspect_reports/.
Usage: set POI_DIR if needed, then run this script from the repository root.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Repository-local paths
POI_DIR = Path('data/poi_2020')
REPORT_DIR = POI_DIR.parent / 'inspect_reports'
REPORT_DIR.mkdir(exist_ok=True)

# ============================================================
# Part 1: file inventory
# ============================================================
def inventory_files(poi_dir):
    """List only top-level .shp files without recursion."""
    print("=" * 70)
    print("[Part 1] File inventory")
    print("=" * 70)

    shp_files = sorted([f for f in poi_dir.iterdir()
                        if f.is_file() and f.suffix.lower() == '.shp'])

    print(f"\nTop-level directory contains {len(shp_files)}  .shp files:\n")

    file_info = []
    for f in shp_files:
        try:
            # Read only minimal shape metadata first
            gdf = gpd.read_file(f, rows=1)
            # Row count is read separately
            full = gpd.read_file(f)
            n_rows = len(full)
            cols = list(full.columns)
            crs = str(full.crs) if full.crs else 'None'
            geom_type = full.geometry.geom_type.iloc[0] if n_rows > 0 else 'Empty'

            print(f"  {f.name}")
            print(f"      Rows: {n_rows:>8d}  Geometry type: {geom_type}  CRS: {crs[:50]}")
            print(f"      Fields: {cols}")
            print()

            file_info.append({
                'filename': f.name,
                'rows': n_rows,
                'geom_type': geom_type,
                'crs': crs,
                'columns': '|'.join(cols),
            })
        except Exception as e:
            print(f"  [!] Could not read {f.name}: {e}")
            file_info.append({
                'filename': f.name,
                'rows': -1,
                'geom_type': 'ERROR',
                'crs': str(e),
                'columns': '',
            })

    pd.DataFrame(file_info).to_csv(REPORT_DIR / '01_file_inventory.csv',
                                    index=False, encoding='utf-8-sig')
    return shp_files


# ============================================================
# Part 2: field diagnostics
# ============================================================
def probe_fields(shp_files):
    """Find likely name/category fields and show samples for each file."""
    print("=" * 70)
    print("[Part 2] Field samples, five rows per file")
    print("=" * 70)

    samples_all = []
    for f in shp_files:
        print(f"\n--- {f.name} ---")
        try:
            gdf = gpd.read_file(f)
            if len(gdf) == 0:
                print("    (empty file)")
                continue

            # Find candidate fields
            cols = list(gdf.columns)
            name_candidates = [c for c in cols if any(
                k in c.lower() for k in ['name', '名称', 'title']
            )]
            type_candidates = [c for c in cols if any(
                k in c.lower() for k in ['type', '类型', '类别', 'class', 'kind', 'cat']
            )]
            addr_candidates = [c for c in cols if any(
                k in c.lower() for k in ['addr', '地址', 'location']
            )]

            print(f"    Candidate name fields: {name_candidates}")
            print(f"    Candidate category fields: {type_candidates}")
            print(f"    Candidate address fields: {addr_candidates}")

            # Sample five rows
            sample = gdf.head(5).drop(columns=['geometry'], errors='ignore')
            print(f"    First five rows:")
            for idx, row in sample.iterrows():
                # Print only the first three non-geometry fields to keep output short
                preview = {c: str(row[c])[:30] for c in sample.columns[:5]}
                print(f"      {preview}")

            # Record samples to a report table
            for _, row in sample.iterrows():
                rec = {'filename': f.name}
                for c in sample.columns[:8]:
                    rec[c] = str(row[c])[:50]
                samples_all.append(rec)

        except Exception as e:
            print(f"    [!] Read error: {e}")

    pd.DataFrame(samples_all).to_csv(REPORT_DIR / '02_field_samples.csv',
                                      index=False, encoding='utf-8-sig')


# ============================================================
# Part 3: category distributions
# ============================================================
def category_distribution(shp_files):
    """Run value_counts for candidate category fields in each file."""
    print("\n" + "=" * 70)
    print("[Part 3] Category field distributions, top 20 values")
    print("=" * 70)

    all_cat = []
    for f in shp_files:
        try:
            gdf = gpd.read_file(f)
            if len(gdf) == 0:
                continue

            # Find category fields
            type_cols = [c for c in gdf.columns if any(
                k in c.lower() for k in ['type', '类型', '类别', 'class', 'kind', 'cat']
            )]
            if not type_cols:
                print(f"\n--- {f.name} --- (No category field found)")
                continue

            print(f"\n--- {f.name} ---")
            for tc in type_cols[:2]:  # Inspect at most two candidate fields
                vc = gdf[tc].astype(str).value_counts().head(20)
                print(f"  Field [{tc}] top values:")
                for val, cnt in vc.items():
                    print(f"      {val[:50]:<55}{cnt:>8d}")

                for val, cnt in vc.items():
                    all_cat.append({
                        'filename': f.name,
                        'field': tc,
                        'value': val,
                        'count': cnt,
                    })
        except Exception as e:
            print(f"    [!] Processing {f.name} failed: {e}")

    pd.DataFrame(all_cat).to_csv(REPORT_DIR / '03_category_distribution.csv',
                                  index=False, encoding='utf-8-sig')


# ============================================================
# Part 4: data quality diagnostics
# ============================================================
def quality_check(shp_files):
    """Check data quality issues in each file."""
    print("\n" + "=" * 70)
    print("[Part 4] Data quality diagnostics")
    print("=" * 70)

    quality_summary = []
    for f in shp_files:
        try:
            gdf = gpd.read_file(f)
            n = len(gdf)
            if n == 0:
                continue

            # 1) Empty geometry
            n_null_geom = gdf.geometry.isna().sum() + (~gdf.geometry.is_valid).sum()

            # 2) Missing names
            name_cols = [c for c in gdf.columns if any(
                k in c.lower() for k in ['name', '名称', 'title']
            )]
            if name_cols:
                nc = name_cols[0]
                n_null_name = gdf[nc].isna().sum() + (gdf[nc].astype(str).str.strip() == '').sum()
                # Duplicate names
                n_dup_name = gdf[nc].duplicated().sum()
            else:
                nc = None
                n_null_name = -1
                n_dup_name = -1

            # 3) Fully duplicated attribute rows, excluding geometry
            non_geom_cols = [c for c in gdf.columns if c != 'geometry']
            if non_geom_cols:
                n_full_dup = gdf[non_geom_cols].duplicated().sum()
            else:
                n_full_dup = 0

            # 4) Duplicate geometries at the same coordinates
            try:
                wkt_series = gdf.geometry.apply(lambda g: g.wkt if g else None)
                n_geom_dup = wkt_series.duplicated().sum()
            except Exception:
                n_geom_dup = -1

            print(f"\n  {f.name}")
            print(f"      Total rows:      {n}")
            print(f"      Empty/invalid geometries:     {n_null_geom}  ({n_null_geom/n*100:.1f}%)")
            print(f"      Name field [{nc}] missing: {n_null_name}  duplicates: {n_dup_name}")
            print(f"      Fully duplicated rows:      {n_full_dup}")
            print(f"      Duplicate coordinate points:      {n_geom_dup}")

            quality_summary.append({
                'filename': f.name,
                'total_rows': n,
                'null_geometry': n_null_geom,
                'null_geometry_pct': round(n_null_geom/n*100, 2),
                'name_field': nc,
                'null_name': n_null_name,
                'duplicate_name': n_dup_name,
                'full_duplicate_rows': n_full_dup,
                'duplicate_geometry': n_geom_dup,
            })

        except Exception as e:
            print(f"    [!] Processing {f.name} failed: {e}")

    pd.DataFrame(quality_summary).to_csv(REPORT_DIR / '04_quality_summary.csv',
                                          index=False, encoding='utf-8-sig')


# ============================================================
# Part 5: cross-file duplicate POIs
# ============================================================
def cross_file_duplicate(shp_files):
    """Check whether the same POI appears in multiple source files."""
    print("\n" + "=" * 70)
    print("[Part 5] Cross-file duplicate POI diagnostics")
    print("=" * 70)

    all_points = []
    for f in shp_files:
        try:
            gdf = gpd.read_file(f)
            if len(gdf) == 0:
                continue
            name_cols = [c for c in gdf.columns if any(
                k in c.lower() for k in ['name', '名称', 'title']
            )]
            if not name_cols:
                continue
            nc = name_cols[0]

            for _, row in gdf.iterrows():
                if row.geometry is None:
                    continue
                # Use name and coordinates rounded to five decimals as a fingerprint
                try:
                    x = round(row.geometry.x, 5)
                    y = round(row.geometry.y, 5)
                    name = str(row[nc])[:40]
                    all_points.append({
                        'filename': f.name,
                        'name': name,
                        'x': x, 'y': y,
                    })
                except Exception:
                    continue
        except Exception as e:
            print(f"    [!] Failed to read {f.name}: {e}")

    if not all_points:
        print("  No usable point data")
        return

    df = pd.DataFrame(all_points)
    df['fingerprint'] = df['name'].astype(str) + '|' + df['x'].astype(str) + '|' + df['y'].astype(str)

    # Cross-file occurrence count
    cross = df.groupby('fingerprint').agg(
        n_files=('filename', 'nunique'),
        files=('filename', lambda s: '|'.join(sorted(set(s)))),
        name=('name', 'first'),
    ).reset_index()

    duplicated_across = cross[cross['n_files'] > 1]

    print(f"\n  Total points:                    {len(df)}")
    print(f"  Unique fingerprints:                {len(cross)}")
    print(f"  Points appearing in multiple files:        {len(duplicated_across)}  ({len(duplicated_across)/len(cross)*100:.1f}%)")

    if len(duplicated_across) > 0:
        print(f"\n  Examples, first 10 cross-file duplicates:")
        for _, row in duplicated_across.head(10).iterrows():
            print(f"      [{row['name']}] appears in {row['n_files']} files: {row['files']}")

    duplicated_across.to_csv(REPORT_DIR / '05_cross_file_duplicates.csv',
                              index=False, encoding='utf-8-sig')


# ============================================================
# Main entry point
# ============================================================
def main():
    if not POI_DIR.exists():
        print(f"❌ Path does not exist: {POI_DIR}")
        print("   Please update POI_DIR near the top of the script")
        return

    print(f"\n📁 POI path: {POI_DIR}")
    print(f"📁 Reports will be written to: {REPORT_DIR}\n")

    shp_files = inventory_files(POI_DIR)
    if not shp_files:
        print("No shapefiles found; exiting.")
        return

    probe_fields(shp_files)
    category_distribution(shp_files)
    quality_check(shp_files)
    cross_file_duplicate(shp_files)

    print("\n" + "=" * 70)
    print("Diagnostics complete. Generated report files:")
    print(f"  1. {REPORT_DIR}/01_file_inventory.csv")
    print(f"  2. {REPORT_DIR}/02_field_samples.csv")
    print(f"  3. {REPORT_DIR}/03_category_distribution.csv")
    print(f"  4. {REPORT_DIR}/04_quality_summary.csv")
    print(f"  5. {REPORT_DIR}/05_cross_file_duplicates.csv")
    print("=" * 70)


if __name__ == '__main__':
    main()
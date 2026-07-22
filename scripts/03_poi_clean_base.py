"""POI cleaning script.

This script converts raw/sampled Gaode POI files into clean PLES subclasses.
Chinese source file names and category values are intentionally preserved because
they are used for data matching.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Path configuration
# ============================================================
POI_DIR = Path('data/poi_2020')
OUT_DIR = POI_DIR /'poi_clean_v0'


OUT_DIR.mkdir(exist_ok=True, parents=True)

OUT_GPKG = OUT_DIR / 'shenzhen_poi_clean_2020.gpkg'
REPORT_CSV = OUT_DIR / 'cleaning_report.csv'


# ============================================================
# Cleaning rule definitions
# ============================================================
# Each source file maps to a processing strategy
# Strategy types:
#   'all_to'          : assign the whole file to one subclass
#   'company_filter'  : filter company records by middle category
#   'residential_filter': remove industrial parks/offices from residential records
#   'discard'         : discard the whole file

FILE_RULES = {
    # Production: industrial
    '深圳市_公司企业.shp': {
        'strategy': 'company_filter',
        'keep_zhonglei': ['工厂', '机械电子', '冶金化工'],
        'subclass': 'industrial',
        'ples': 'production',
    },

    # Living: residential
    '深圳市_商务住宅.shp': {
        'strategy': 'residential_filter',
        'subclass': 'residential',
        'ples': 'living',
    },

    # Living: commercial
    '深圳市_购物服务.shp':   {'strategy': 'all_to', 'subclass': 'commercial', 'ples': 'living'},
    '深圳市_餐饮服务.shp':   {'strategy': 'all_to', 'subclass': 'commercial', 'ples': 'living'},
    '深圳市_住宿服务.shp':   {'strategy': 'all_to', 'subclass': 'commercial', 'ples': 'living'},
    '深圳市_金融保险服务.shp': {'strategy': 'all_to', 'subclass': 'commercial', 'ples': 'living'},
    '深圳市_汽车销售.shp':   {'strategy': 'all_to', 'subclass': 'commercial', 'ples': 'living'},
    '深圳市_生活服务.shp':   {'strategy': 'all_to', 'subclass': 'commercial', 'ples': 'living'},
    '深圳市_体育休闲服务.shp': {'strategy': 'all_to', 'subclass': 'commercial', 'ples': 'living'},

    # Living: public service
    '深圳市_科教文化服务.shp':   {'strategy': 'all_to', 'subclass': 'public_service', 'ples': 'living'},
    '深圳市_医疗保健服务.shp':   {'strategy': 'all_to', 'subclass': 'public_service', 'ples': 'living'},
    '深圳市_政府机构及社会团体.shp': {'strategy': 'all_to', 'subclass': 'public_service', 'ples': 'living'},

    # Ecological auxiliary
    '深圳市_风景名胜.shp': {'strategy': 'all_to', 'subclass': 'ecological', 'ples': 'ecological'},

    # Discarded categories
    '深圳市_公共设施.shp':       {'strategy': 'discard', 'reason': 'streetlights, public toilets, gates; not relevant to PLES'},
    '深圳市_室内设施.shp':       {'strategy': 'discard', 'reason': 'indoor navigation points'},
    '深圳市_通行设施.shp':       {'strategy': 'discard', 'reason': 'metro stations or gates'},
    '深圳市_交通设施服务.shp':   {'strategy': 'discard', 'reason': 'transport infrastructure'},
    '深圳市_道路附属设施.shp':   {'strategy': 'discard', 'reason': 'road facilities'},
    '深圳市_汽车服务.shp':       {'strategy': 'discard', 'reason': 'auto-service POIs with weak functional signal'},
    '深圳市_汽车维修.shp':       {'strategy': 'discard', 'reason': 'auto-repair POIs with weak functional signal'},
    '深圳市_摩托车服务.shp':     {'strategy': 'discard', 'reason': 'too few records in the full source dataset'},
    '深圳市_地名地址信息.shp':   {'strategy': 'discard', 'reason': 'address points rather than functional POIs'},
    '深圳市_事件活动.shp':       {'strategy': 'discard', 'reason': 'very few records in the full source dataset'},
}

# Non-residential subclass keywords to remove from the residential source file
# The Gaode residential source can include industrial parks, offices, or building labels.
RESIDENTIAL_EXCLUDE_KEYWORDS = [
    '产业园区',
    '写字楼',  # Occasionally appears as an office-building subclass in the residential source.
    '建筑物',  # Usually a single-building label rather than a residential function.
    '楼栋',
    '门牌信息',
]


# ============================================================
# Field normalization for consistent output layers
# ============================================================
KEEP_FIELDS = {
    'name': 'name',
    'address': 'address',
    'lon': 'lon',
    'lat': 'lat',
    '大类': 'raw_dalei',
    '中类': 'raw_zhonglei',
    '小类': 'raw_xiaolei',
    'typecode': 'typecode',
}


def standardize_gdf(gdf, file_name):
    """Normalize field names, add poi_id, and keep required columns."""
    # Rename and select columns
    rename_map = {k: v for k, v in KEEP_FIELDS.items() if k in gdf.columns}
    gdf = gdf.rename(columns=rename_map)

    # Ensure required fields exist
    for v in KEEP_FIELDS.values():
        if v not in gdf.columns:
            gdf[v] = None

    # Add source_file for provenance
    gdf['source_file'] = file_name

    keep_cols = list(KEEP_FIELDS.values()) + ['source_file', 'geometry']
    keep_cols = [c for c in keep_cols if c in gdf.columns]
    return gdf[keep_cols].copy()


# ============================================================
# Main cleaning workflow
# ============================================================
def main():
    if not POI_DIR.exists():
        print(f"❌ POI Path does not exist: {POI_DIR}")
        return

    print(f"📁 Input: {POI_DIR}")
    print(f"📁 Output: {OUT_DIR}\n")

    # Remove the existing GPKG to avoid stale layers
    if OUT_GPKG.exists():
        OUT_GPKG.unlink()
        print(f"  (Removing existing file {OUT_GPKG.name})")

    # Accumulators for five subclasses
    pools = {
        'industrial': [],
        'residential': [],
        'commercial': [],
        'public_service': [],
        'ecological': [],
    }

    # Cleaning statistics
    report = []

    shp_files = sorted([f for f in POI_DIR.iterdir()
                        if f.is_file() and f.suffix.lower() == '.shp'])

    print(f"=" * 70)
    print(f"Start cleaning ({len(shp_files)} files)")
    print(f"=" * 70)

    for f in shp_files:
        fname = f.name
        rule = FILE_RULES.get(fname)

        if rule is None:
            print(f"\n⚠ {fname} (no rule, skipped)")
            report.append({
                'filename': fname,
                'strategy': 'no_rule',
                'raw_count': -1,
                'kept_count': 0,
                'subclass': '',
                'note': 'Not listed in the rule table; skipped',
            })
            continue

        try:
            gdf = gpd.read_file(f)
            n_raw = len(gdf)
        except Exception as e:
            print(f"\n❌ {fname} read failed: {e}")
            continue

        if n_raw == 0:
            continue

        strategy = rule['strategy']

        # ============ Discard ============
        if strategy == 'discard':
            print(f"\nDiscarding {fname} ({n_raw} rows): {rule['reason']}")
            report.append({
                'filename': fname,
                'strategy': 'discard',
                'raw_count': n_raw,
                'kept_count': 0,
                'subclass': '',
                'note': rule['reason'],
            })
            continue

        # ============ Whole-file assignment ============
        if strategy == 'all_to':
            sub = gdf
            note = 'whole file assigned'

        # ============ Company filtering ============
        elif strategy == 'company_filter':
            keep_zl = rule['keep_zhonglei']
            if '中类' not in gdf.columns:
                print(f"\nMissing required field '中类' in {fname}")
                continue
            sub = gdf[gdf['中类'].astype(str).isin(keep_zl)].copy()
            note = f"filtered by middle category: {keep_zl}"

        # ============ Residential filtering ============
        elif strategy == 'residential_filter':
            # Check both middle and small categories to remove non-residential records
            zl = gdf['中类'].astype(str) if '中类' in gdf.columns else pd.Series([''] * len(gdf))
            xl = gdf['小类'].astype(str) if '小类' in gdf.columns else pd.Series([''] * len(gdf))
            combined = (zl + '|' + xl).str.lower()

            # Remove records matching any exclusion keyword
            mask_exclude = pd.Series([False] * len(gdf))
            for kw in RESIDENTIAL_EXCLUDE_KEYWORDS:
                mask_exclude = mask_exclude | combined.str.contains(kw.lower(), na=False)
            sub = gdf[~mask_exclude].copy()
            note = f"removed non-residential subclasses: {RESIDENTIAL_EXCLUDE_KEYWORDS}"

        else:
            print(f"\nUnknown strategy {strategy}; skipping {fname}")
            continue

        if len(sub) == 0:
            print(f"\n⚠ {fname} empty after filtering")
            continue

        # Normalize fields
        sub = standardize_gdf(sub, fname)

        # Deduplicate within each subclass using name and rounded coordinates
        n_before_dedup = len(sub)
        sub['_fp'] = (sub['name'].astype(str).str[:50] + '|' +
                      sub.geometry.x.round(5).astype(str) + '|' +
                      sub.geometry.y.round(5).astype(str))
        sub = sub.drop_duplicates(subset=['_fp'])
        sub = sub.drop(columns=['_fp'])
        n_after_dedup = len(sub)
        n_dup_removed = n_before_dedup - n_after_dedup

        # Add class labels
        sub['ples_class'] = rule['ples']
        sub['ples_subclass'] = rule['subclass']

        # Append to the subclass accumulator
        pools[rule['subclass']].append(sub)

        print(f"\nProcessed {fname} ({n_raw} rows)")
        print(f"   → kept after filtering: {n_before_dedup}  duplicates removed: {n_dup_removed}  final: {n_after_dedup}")
        print(f"   → assigned to: {rule['subclass']} ({rule['ples']})  {note}")

        report.append({
            'filename': fname,
            'strategy': strategy,
            'raw_count': n_raw,
            'kept_count': n_after_dedup,
            'dup_removed': n_dup_removed,
            'subclass': rule['subclass'],
            'ples': rule['ples'],
            'note': note,
        })

    # ============ Merge subclasses and write GPKG ============
    print(f"\n{'=' * 70}")
    print("Merging subclasses and writing GeoPackage...")
    print(f"{'=' * 70}\n")

    summary_rows = []
    all_clean = []

    for subcls, dfs in pools.items():
        if not dfs:
            print(f"  [{subcls}] empty, skipped")
            continue
        merged = pd.concat(dfs, ignore_index=True)

        # Deduplicate again across sources within each subclass
        n_before = len(merged)
        merged['_fp'] = (merged['name'].astype(str).str[:50] + '|' +
                         merged.geometry.x.round(5).astype(str) + '|' +
                         merged.geometry.y.round(5).astype(str))
        merged = merged.drop_duplicates(subset=['_fp']).drop(columns=['_fp'])
        n_after = len(merged)

        # Add global poi_id
        merged = merged.reset_index(drop=True)
        merged['poi_id'] = subcls + '_' + merged.index.astype(str).str.zfill(7)

        # Write layer
        merged_gdf = gpd.GeoDataFrame(merged, geometry='geometry', crs='EPSG:4326')
        merged_gdf.to_file(OUT_GPKG, layer=subcls, driver='GPKG')

        print(f"  ✓ Layer [{subcls}]: {n_after} POIs  (cross-source deduplication -{n_before-n_after})")

        summary_rows.append({
            'subclass': subcls,
            'ples': merged['ples_class'].iloc[0],
            'final_count': n_after,
            'cross_source_dup_removed': n_before - n_after,
        })

        all_clean.append(merged_gdf)

    # ============ Write merged layer and reports ============
    if all_clean:
        all_gdf = gpd.GeoDataFrame(pd.concat(all_clean, ignore_index=True),
                                   geometry='geometry', crs='EPSG:4326')
        all_gdf.to_file(OUT_GPKG, layer='all_clean', driver='GPKG')
        print(f"\n  Layer [all_clean]: {len(all_gdf)} POIs (merged)")

    # Save cleaning report
    pd.DataFrame(report).to_csv(REPORT_CSV, index=False, encoding='utf-8-sig')

    # Save subclass summary
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / 'subclass_summary.csv',
                                       index=False, encoding='utf-8-sig')

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"Cleaning complete")
    print(f"{'=' * 70}")
    print(f"  GeoPackage: {OUT_GPKG}")
    print(f"  Cleaning report:   {REPORT_CSV}")
    print(f"  Subclass summary:   {OUT_DIR / 'subclass_summary.csv'}\n")

    print(f"  POI counts by subclass:")
    for row in summary_rows:
        print(f"    {row['subclass']:<18} ({row['ples']:<11})  {row['final_count']:>8d}")

    total_raw = sum(r.get('raw_count', 0) for r in report if r.get('raw_count', 0) > 0)
    total_kept = sum(r.get('kept_count', 0) for r in report)
    total_discard = sum(r.get('raw_count', 0) for r in report
                        if r.get('strategy') == 'discard')
    print(f"\n  Statistics:")
    print(f"    Raw POI total: {total_raw}")
    print(f"    Kept POIs:      {total_kept}  ({total_kept/total_raw*100:.1f}%)")
    print(f"    Discarded POIs:      {total_discard}  ({total_discard/total_raw*100:.1f}%)")
    print(f"    duplicates removed:      {total_raw - total_kept - total_discard}")

    print(f"\nNext: inspect the cleaned results, then run the grid-scoring script.")


if __name__ == '__main__':
    main()
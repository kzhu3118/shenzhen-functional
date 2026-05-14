"""Industrial POI patch script.

Adds company records whose detailed subclasses indicate machinery/electronics or
metallurgy/chemical industry to the industrial POI layer.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Repository-local paths
POI_FILE = Path('data/poi_2020/深圳市_公司企业.shp')
OUT_GPKG = Path('data/poi_2020/poi_clean/shenzhen_poi_clean_2020.gpkg')

# Industrial subclasses to add
INDUSTRIAL_XIAOLEI_TO_ADD = ['机械电子', '冶金化工']


def main():
    if not POI_FILE.exists():
        print(f"Company source file does not exist: {POI_FILE}")
        return
    if not OUT_GPKG.exists():
        print(f"GeoPackage does not exist: {OUT_GPKG}")
        return

    print("Reading company source file...")
    gdf = gpd.read_file(POI_FILE)
    print(f"   Total rows: {len(gdf)}")

    # Filter records where 中类 == 公司 and 小类 is in the industrial subclass list.
    if '中类' not in gdf.columns or '小类' not in gdf.columns:
        print("Missing required fields: 中类 or 小类")
        return

    mask = (gdf['中类'].astype(str) == '公司') & \
           (gdf['小类'].astype(str).isin(INDUSTRIAL_XIAOLEI_TO_ADD))
    sub = gdf[mask].copy()

    print(f"\n   Filter result (中类=公司, 小类 in {INDUSTRIAL_XIAOLEI_TO_ADD}):")
    for xl in INDUSTRIAL_XIAOLEI_TO_ADD:
        cnt = (sub['小类'].astype(str) == xl).sum()
        print(f"      {xl}: {cnt}")
    print(f"   Subtotal: {len(sub)}")

    if len(sub) == 0:
        print("No supplemental records found")
        return

    # Normalize fields to match Step 1.
    KEEP_FIELDS = {
        'name': 'name', 'address': 'address',
        'lon': 'lon', 'lat': 'lat',
        '大类': 'raw_dalei', '中类': 'raw_zhonglei', '小类': 'raw_xiaolei',
        'typecode': 'typecode',
    }
    rename_map = {k: v for k, v in KEEP_FIELDS.items() if k in sub.columns}
    sub = sub.rename(columns=rename_map)
    for v in KEEP_FIELDS.values():
        if v not in sub.columns:
            sub[v] = None
    sub['source_file'] = '深圳市_公司企业.shp'
    sub['ples_class'] = 'production'
    sub['ples_subclass'] = 'industrial'

    keep_cols = list(KEEP_FIELDS.values()) + ['source_file', 'ples_class', 'ples_subclass', 'geometry']
    sub = sub[[c for c in keep_cols if c in sub.columns]].copy()

    # Internal deduplication
    sub['_fp'] = (sub['name'].astype(str).str[:50] + '|' +
                  sub.geometry.x.round(5).astype(str) + '|' +
                  sub.geometry.y.round(5).astype(str))
    n_before = len(sub)
    sub = sub.drop_duplicates(subset=['_fp']).drop(columns=['_fp'])
    print(f"   duplicates removed: {n_before - len(sub)}, supplemental final count: {len(sub)}")

    # Read existing industrial layer
    print("\nReading existing industrial layer...")
    existing = gpd.read_file(OUT_GPKG, layer='industrial')
    print(f"   Existing records: {len(existing)}")

    # Merge
    combined = pd.concat([existing.drop(columns=['poi_id']) if 'poi_id' in existing.columns else existing,
                          sub], ignore_index=True)

    # cross-source deduplication
    combined['_fp'] = (combined['name'].astype(str).str[:50] + '|' +
                       combined.geometry.x.round(5).astype(str) + '|' +
                       combined.geometry.y.round(5).astype(str))
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=['_fp']).drop(columns=['_fp'])
    n_after = len(combined)
    print(f"   Cross-source deduplication after merge: -{n_before - n_after}")

    # Regenerate poi_id
    combined = combined.reset_index(drop=True)
    combined['poi_id'] = 'industrial_' + combined.index.astype(str).str.zfill(7)

    combined_gdf = gpd.GeoDataFrame(combined, geometry='geometry', crs='EPSG:4326')
    print(f"\nRewriting industrial layer, new count: {len(combined_gdf)}...")
    combined_gdf.to_file(OUT_GPKG, layer='industrial', driver='GPKG')

    # Regenerate all_clean layer
    print("Regenerating all_clean layer...")
    all_layers = []
    for layer in ['industrial', 'residential', 'commercial', 'public_service', 'ecological']:
        try:
            ly = gpd.read_file(OUT_GPKG, layer=layer)
            all_layers.append(ly)
        except Exception as e:
            print(f"   Failed to read {layer}: {e}")

    if all_layers:
        all_clean = gpd.GeoDataFrame(pd.concat(all_layers, ignore_index=True),
                                     geometry='geometry', crs='EPSG:4326')
        all_clean.to_file(OUT_GPKG, layer='all_clean', driver='GPKG')
        print(f"   all_clean total: {len(all_clean)}")

    print("\nPatch complete")
    print(f"  industrial: 3549 → {len(combined_gdf)} (+{len(combined_gdf)-3549})")


if __name__ == '__main__':
    main()
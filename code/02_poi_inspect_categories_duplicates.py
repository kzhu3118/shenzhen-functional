"""POI category and duplicate inspection script.

This second inspection focuses on Gaode category distributions, company/industrial
subclasses, and duplicate POI records across source files.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Repository-local paths
POI_DIR = Path('data/poi_2020')
REPORT_DIR = POI_DIR.parent / 'inspect_reports'
REPORT_DIR.mkdir(exist_ok=True, parents=True)


# ============================================================
# Part 1: Gaode major-category distribution
# ============================================================
def gaode_dalei(shp_files):
    """Inspect the actual Gaode major-category distribution in each file."""
    print("=" * 70)
    print("[Part 1] Gaode major-category distribution")
    print("=" * 70)

    all_records = []
    for f in shp_files:
        try:
            gdf = gpd.read_file(f)
            if len(gdf) == 0:
                continue

            # Prefer the 大类 column; fall back to typename when needed.
            if '大类' in gdf.columns:
                col = '大类'
            elif 'typename' in gdf.columns:
                col = 'typename'
            else:
                print(f"  [{f.name}] has no major-category field; skipped")
                continue

            vc = gdf[col].astype(str).value_counts().head(10)
            print(f"\n--- {f.name}  (Fields: {col}) ---")
            for val, cnt in vc.items():
                print(f"      {val[:40]:<42}{cnt:>8d}")

            for val, cnt in vc.items():
                all_records.append({
                    'filename': f.name,
                    'field': col,
                    'value': str(val)[:60],
                    'count': cnt,
                })
        except Exception as e:
            print(f"  [!] {f.name}: {e}")

    pd.DataFrame(all_records).to_csv(REPORT_DIR / 'A1_gaode_dalei.csv',
                                      index=False, encoding='utf-8-sig')


# ============================================================
# Part 2: detailed inspection of the company source file
# ============================================================
def company_anatomy(poi_dir):
    """
    Inspect the company source file in detail.
    This is the key source for identifying industrial POIs.
    """
    print("\n" + "=" * 70)
    print("[Part 2] Detailed inspection of company records")
    print("=" * 70)

    f = poi_dir / '深圳市_公司企业.shp'
    if not f.exists():
        print(f"  Could not find {f.name}")
        return

    gdf = gpd.read_file(f)
    n = len(gdf)
    print(f"\n  Total company records: {n}")
    print(f"  Fields: {list(gdf.columns)[:20]}\n")

    # === 2A: middle-category distribution, top 30 ===
    print("  --- Middle-category distribution, top 30 ---")
    if '中类' in gdf.columns:
        vc = gdf['中类'].astype(str).value_counts().head(30)
        for val, cnt in vc.items():
            pct = cnt / n * 100
            print(f"      {val[:35]:<37}{cnt:>7d}  ({pct:>5.1f}%)")
        vc.to_csv(REPORT_DIR / 'A2a_company_zhonglei.csv', encoding='utf-8-sig')

    # === 2B: small-category distribution, top 50 ===
    print("\n  --- Small-category distribution, top 50 ---")
    if '小类' in gdf.columns:
        vc = gdf['小类'].astype(str).value_counts().head(50)
        for val, cnt in vc.items():
            pct = cnt / n * 100
            print(f"      {val[:40]:<42}{cnt:>7d}  ({pct:>5.1f}%)")
        gdf['小类'].astype(str).value_counts().to_csv(
            REPORT_DIR / 'A2b_company_xiaolei.csv', encoding='utf-8-sig')

    # === 2C: typecode-prefix analysis ===
    print("\n  --- Typecode-prefix analysis, top 20 ---")
    if 'typecode' in gdf.columns:
        # typecode has six digits; the first two encode major category and first four encode middle category.
        gdf['_tc4'] = gdf['typecode'].astype(str).str[:4]
        vc = gdf['_tc4'].value_counts().head(20)
        for val, cnt in vc.items():
            pct = cnt / n * 100
            print(f"      typecode first four digits [{val}]: {cnt:>7d} ({pct:>5.1f}%)")

    # === 2D: name-keyword scan for likely factories ===
    print("\n  --- Name-keyword scan ---")
    name_col = 'name' if 'name' in gdf.columns else None
    if name_col:
        names = gdf[name_col].astype(str)

        # Keyword groups
        kw_groups = {
            '工厂_后缀': ['工厂$', '制造厂$', '加工厂$', '化工厂$', '机械厂$',
                       '电子厂$', '塑胶厂$', '五金厂$', '印刷厂$'],
            '工厂_含': ['工厂', '制造厂', '加工厂'],
            '工业园区': ['工业园', '产业园', '工业区'],
            '科技园区': ['科技园', '创新园', '研发中心', '总部'],
            '咨询服务': ['咨询', '顾问', '事务所'],
            '贸易': ['贸易', '商贸', '商务'],
            '投资': ['投资', '资本', '基金', '控股'],
            '文化传媒': ['文化', '传媒', '广告', '影视', '设计'],
            '科技公司': ['科技有限公司', '信息技术'],
            '教育培训': ['教育', '培训', '学校'],
            '律师所': ['律师', '法律服务'],
        }

        print(f"      Keyword matches among {n} company records:")
        for label, kws in kw_groups.items():
            cnt = 0
            for kw in kws:
                cnt += names.str.contains(kw, na=False, regex=True).sum()
            pct = cnt / n * 100
            print(f"        {label:<15}{cnt:>7d}  ({pct:>5.1f}%)")

    # === 2E: company middle-category by small-category table ===
    if '中类' in gdf.columns and '小类' in gdf.columns:
        joint = gdf.groupby(['中类', '小类']).size().reset_index(name='count')
        joint = joint.sort_values('count', ascending=False)
        joint.to_csv(REPORT_DIR / 'A2e_company_zhong_xiao_joint.csv',
                     index=False, encoding='utf-8-sig')
        print(f"\n  Number of company middle x small category combinations: {len(joint)}")


# ============================================================
# Part 3: detailed cross-file duplicate diagnosis
# ============================================================
def cross_dup_v2(shp_files):
    """Use the real name field and coordinates for cross-file duplicate diagnosis."""
    print("\n" + "=" * 70)
    print("[Part 3] Detailed cross-file duplicate diagnosis")
    print("=" * 70)

    all_pts = []
    for f in shp_files:
        try:
            gdf = gpd.read_file(f)
            if len(gdf) == 0 or 'name' not in gdf.columns:
                continue
            for _, row in gdf.iterrows():
                if row.geometry is None:
                    continue
                try:
                    x = round(row.geometry.x, 5)
                    y = round(row.geometry.y, 5)
                    nm = str(row['name'])[:40] if pd.notna(row['name']) else ''
                    all_pts.append({
                        'file': f.name.replace('深圳市_', '').replace('.shp', ''),
                        'name': nm,
                        'x': x, 'y': y,
                    })
                except Exception:
                    continue
        except Exception as e:
            print(f"  [!] {f.name}: {e}")

    if not all_pts:
        print("  No data")
        return

    df = pd.DataFrame(all_pts)
    df['fp'] = df['name'] + '|' + df['x'].astype(str) + '|' + df['y'].astype(str)

    cross = df.groupby('fp').agg(
        nf=('file', 'nunique'),
        files=('file', lambda s: '+'.join(sorted(set(s)))),
        nm=('name', 'first'),
    ).reset_index()
    dup = cross[cross['nf'] > 1]

    print(f"\n  Total points:                {len(df)}")
    print(f"  Unique fingerprints:            {len(cross)}")
    print(f"  Cross-file duplicate fingerprints: {len(dup)} ({len(dup)/len(cross)*100:.1f}%)")
    print(f"  Duplicate point records involved: {len(df) - len(cross)}\n")

    # Top 20 cross-file duplicate combinations
    print("  Top 20 cross-file duplicate combinations:")
    combo_vc = dup['files'].value_counts().head(20)
    for combo, cnt in combo_vc.items():
        print(f"      {combo[:60]:<62}{cnt:>6d}")

    combo_vc.to_csv(REPORT_DIR / 'A3_cross_file_combos.csv', encoding='utf-8-sig')

    # Random duplicate samples for inspection
    sample = dup.sample(min(50, len(dup)), random_state=42)
    sample.to_csv(REPORT_DIR / 'A3_dup_samples.csv', index=False, encoding='utf-8-sig')


# ============================================================
# Main entry point
# ============================================================
def main():
    if not POI_DIR.exists():
        print(f"❌ Path does not exist: {POI_DIR}")
        return

    print(f"\n📁 POI path: {POI_DIR}")
    print(f"📁 Reports will be written to: {REPORT_DIR}\n")

    shp_files = sorted([f for f in POI_DIR.iterdir()
                        if f.is_file() and f.suffix.lower() == '.shp'])
    print(f"  Found {len(shp_files)} shapefiles\n")

    gaode_dalei(shp_files)
    company_anatomy(POI_DIR)
    cross_dup_v2(shp_files)

    print("\n" + "=" * 70)
    print("Second-round diagnostics complete. Generated report files:")
    print(f"  1. {REPORT_DIR}/A1_gaode_dalei.csv")
    print(f"  2. {REPORT_DIR}/A2a_company_zhonglei.csv")
    print(f"  3. {REPORT_DIR}/A2b_company_xiaolei.csv")
    print(f"  4. {REPORT_DIR}/A2e_company_zhong_xiao_joint.csv")
    print(f"  5. {REPORT_DIR}/A3_cross_file_combos.csv")
    print(f"  6. {REPORT_DIR}/A3_dup_samples.csv")
    print("  + Console output, optionally saved with tee")
    print("=" * 70)


if __name__ == '__main__':
    main()
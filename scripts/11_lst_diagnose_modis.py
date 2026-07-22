"""MODIS LST quality diagnosis.

Compares MODIS summer LST completeness, stable ecological-grid temperature
changes, citywide distributions, interannual variability, and Landsat annual LST.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LST_DIR = PROJECT_ROOT / "data/GEE/shenzhen_features_2000_2020"
PLES_GPKG = PROJECT_ROOT / "data/rf_multiyear/noNTL_manual2005_gaia_ratio_thr10/shenzhen_ples_multiyear_noNTL_manual2005_gaia_ratio.gpkg"
PLES_LAYER = "ples_multiyear_noNTL_manual2005_gaia_ratio"

YEARS = [2000, 2005, 2010, 2015, 2020]


def load_modis_lst(year: int) -> pd.DataFrame:
    path = LST_DIR / f"shenzhen_LST_MODIS_{year}.csv"
    df = pd.read_csv(path)
    # MODIS data are already in K after the 0.02 scale factor; convert to Celsius.
    for col in ['LST_mean', 'LST_min', 'LST_max', 'LST_median']:
        if col in df.columns:
            df[col] = df[col] - 273.15
    return df


def load_annual_lst(year: int) -> pd.DataFrame:
    """Load the original annual Landsat LST for comparison."""
    if year == 2020:
        path = LST_DIR / "shenzhen_features_2020.csv"
    else:
        path = LST_DIR / f"shenzhen_features_{year}_mixedNTL.csv"
    df = pd.read_csv(path, usecols=['grid_id', 'LST_mean'])
    df['LST_mean'] = df['LST_mean'] - 273.15
    df = df.rename(columns={'LST_mean': 'LST_annual'})
    return df


def main():
    print("=" * 70)
    print("MODIS LST quality diagnosis")
    print("=" * 70)

    # === [1] Data completeness ===
    print("\n[1] Data completeness; MODIS should be close to 100%")
    print("-" * 70)
    print(f"{'Year':<8}{'Total':<10}{'Valid':<10}{'Missing':<10}{'Pct':<10}")
    print("-" * 70)
    summer_data = {}
    for year in YEARS:
        df = load_modis_lst(year)
        df_valid = df.dropna(subset=['LST_mean'])
        summer_data[year] = df_valid
        n_total = len(df)
        n_valid = len(df_valid)
        n_missing = n_total - n_valid
        pct = n_missing / n_total * 100
        flag = "✓" if pct < 5 else "⚠"
        print(f"{year:<8}{n_total:<10}{n_valid:<10}{n_missing:<10}{pct:<5.1f}%   {flag}")

    # === [2] Stable ecological-grid delta LST ===
    print("\n[2] Stable ecological-grid delta LST, key seasonal-bias indicator")
    print("-" * 70)
    ples = gpd.read_file(PLES_GPKG, layer=PLES_LAYER)
    ples_df = pd.DataFrame(ples[['grid_id'] + [f'ples_{y}' for y in YEARS]])

    stages = [(2010, 2015), (2015, 2020), (2005, 2010), (2000, 2005)]
    print(f"{'Stage':<15}{'N_eco_stable':<18}{'MODIS delta LST':<18}{'(vs annual Landsat)':<20}")
    print("-" * 70)
    for t0, t1 in stages:
        eco_mask = (ples_df[f'ples_{t0}'] == 'ecological') & (ples_df[f'ples_{t1}'] == 'ecological')
        eco_grids = ples_df.loc[eco_mask, 'grid_id'].values

        s0 = load_modis_lst(t0).set_index('grid_id').reindex(eco_grids)['LST_mean']
        s1 = load_modis_lst(t1).set_index('grid_id').reindex(eco_grids)['LST_mean']
        d_modis = (s1 - s0).dropna()

        try:
            a0 = load_annual_lst(t0).set_index('grid_id').reindex(eco_grids)['LST_annual']
            a1 = load_annual_lst(t1).set_index('grid_id').reindex(eco_grids)['LST_annual']
            d_annual = (a1 - a0).dropna()
            ann_str = f"{d_annual.mean():+.3f}"
        except Exception:
            ann_str = "N/A"

        flag = "✓" if abs(d_modis.mean()) < 1.0 else ("◐" if abs(d_modis.mean()) < 2.0 else "⚠")
        print(f"{t0}→{t1:<10}{len(d_modis):<18}{d_modis.mean():+.3f}     {flag}    {ann_str}")

    print("\n   Guideline: |delta LST| < 1.0 C good; 1.0-2.0 C moderate; >2.0 C concerning")

    # === [3] Overall distribution ===
    print("\n[3] Five-year summer MODIS LST distribution (C)")
    print("-" * 70)
    print(f"{'Year':<8}{'N':<10}{'Mean':<10}{'Median':<10}{'StdDev':<10}{'Min':<10}{'Max':<10}")
    print("-" * 70)
    for year in YEARS:
        v = summer_data[year]['LST_mean']
        flag = "✓" if 28 < v.mean() < 42 else "⚠"
        print(f"{year:<8}{len(v):<10}{v.mean():<10.2f}{v.median():<10.2f}"
              f"{v.std():<10.2f}{v.min():<10.2f}{v.max():<10.2f}  {flag}")

    print("\n   Plausible range: Shenzhen summer mean LST about 28-42 C")

    # === [4] Interannual variability ===
    print("\n[4] Interannual mean differences")
    print("-" * 70)
    means = {y: summer_data[y]['LST_mean'].mean() for y in YEARS}
    for y0, y1 in zip(YEARS[:-1], YEARS[1:]):
        diff = means[y1] - means[y0]
        flag = "✓" if abs(diff) < 1.5 else "⚠"
        print(f"   {y0} → {y1}: {diff:+.2f} °C  {flag}")

    print("\n   Guideline: citywide interannual mean variation < 1.5 C")

    # === [5] Comparison with annual Landsat LST ===
    print("\n[5] MODIS summer LST vs annual Landsat LST")
    print("-" * 70)
    print(f"{'Year':<8}{'Landsat annual':<18}{'MODIS summer':<18}{'Diff (M-L)':<15}")
    print("-" * 70)
    for year in YEARS:
        try:
            ann = load_annual_lst(year)
            ann_mean = ann['LST_annual'].mean()
            mod_mean = means[year]
            diff = mod_mean - ann_mean
            print(f"{year:<8}{ann_mean:<18.2f}{mod_mean:<18.2f}{diff:+.2f}")
        except Exception as e:
            print(f"{year:<8}(annual load failed: {e})")

    print("\n   Interpretation: MODIS summer LST being 5-12 C higher than annual Landsat median is expected.")

    # === Summary ===
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    eco_diffs = []
    for t0, t1 in [(2010, 2015), (2015, 2020)]:
        eco_mask = (ples_df[f'ples_{t0}'] == 'ecological') & (ples_df[f'ples_{t1}'] == 'ecological')
        eco_grids = ples_df.loc[eco_mask, 'grid_id'].values
        s0 = load_modis_lst(t0).set_index('grid_id').reindex(eco_grids)['LST_mean']
        s1 = load_modis_lst(t1).set_index('grid_id').reindex(eco_grids)['LST_mean']
        d = (s1 - s0).dropna()
        eco_diffs.append((t0, t1, d.mean()))

    print("\nOverall stable ecological-grid delta LST assessment:")
    for t0, t1, d in eco_diffs:
        print(f"   {t0}→{t1}: {d:+.3f}°C")

    abs_diffs = [abs(d) for _, _, d in eco_diffs]
    if all(d < 1.0 for d in abs_diffs):
        print("\nMODIS data quality is acceptable; step 5 can be rerun.")
    elif all(d < 2.0 for d in abs_diffs):
        print("\nMODIS data are usable but show slight seasonal bias; consider background adjustment.")
    else:
        print("\nMODIS data still show large bias; further diagnosis is needed.")


if __name__ == '__main__':
    main()
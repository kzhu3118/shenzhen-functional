"""Health check for the exported 2020 GEE feature CSV.

Checks file readability, expected row/column counts, grid_id integrity, missing
values, and plausible value ranges before model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Repository-local GEE feature table.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / 'data/GEE/shenzhen_features_2000_2020/shenzhen_features_2020.csv'

# Expected values
EXPECTED_ROWS = 19727
EXPECTED_BANDS = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                  'NDVI', 'NDBI', 'MNDWI', 'NBI',
                  'LST', 'NTL', 'DEM', 'slope']
EXPECTED_STATS = ['mean', 'stdDev', 'min', 'max', 'median']

# Plausible value ranges for anomaly checks
VALUE_RANGES = {
    'SR_B2': (0, 0.5),    # Landsat 8 surface reflectance
    'SR_B3': (0, 0.5),
    'SR_B4': (0, 0.5),
    'SR_B5': (0, 0.7),
    'SR_B6': (0, 0.6),
    'SR_B7': (0, 0.5),
    'NDVI': (-1, 1),
    'NDBI': (-1, 1),
    'MNDWI': (-1, 1),
    'NBI': (0, 5),
    'LST': (280, 340),    # K, plausible Shenzhen seasonal range
    'NTL': (0, 100),      # VIIRS avg_rad
    'DEM': (-10, 1000),   # Wutong Mountain is about 944 m
    'slope': (0, 90),     # degrees
}


def main():
    if not CSV_PATH.exists():
        print(f"File does not exist: {CSV_PATH}")
        print("\n   Possible reasons:")
        print("   1. The Google Drive client has not synced the file yet.")
        print("   2. The GEE task is still running; check Tasks in Earth Engine.")
        print("   3. The configured path is incorrect.")
        return

    print(f"📁 Reading CSV: {CSV_PATH.name}")
    print(f"   File size: {CSV_PATH.stat().st_size / 1024**2:.1f} MB")

    # ============ Step 1: basic read ============
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"❌ read failed: {e}")
        return

    n_rows = len(df)
    n_cols = len(df.columns)
    print(f"\n[1/6] Basic information")
    print(f"   Rows: {n_rows}  (expected {EXPECTED_ROWS})")
    print(f"   Columns: {n_cols}  (expected 1 + {len(EXPECTED_BANDS)} × {len(EXPECTED_STATS)} = {1 + len(EXPECTED_BANDS) * len(EXPECTED_STATS)})")

    if n_rows != EXPECTED_ROWS:
        print(f"   Row count mismatch; difference: {EXPECTED_ROWS - n_rows}")
    else:
        print("   Row count is correct")

    # ============ Step 2: grid_id check ============
    print(f"\n[2/6] grid_id check")
    if 'grid_id' not in df.columns:
        print("   Missing grid_id column")
        print(f"   Existing columns: {list(df.columns)[:5]}...")
        return

    grid_ids = df['grid_id']
    print(f"   Range: {grid_ids.min()} ~ {grid_ids.max()}")
    print(f"   Unique values: {grid_ids.nunique()}")
    print(f"   Duplicates:   {grid_ids.duplicated().sum()}")

    expected_ids = set(range(EXPECTED_ROWS))
    actual_ids = set(grid_ids.tolist())
    missing = expected_ids - actual_ids
    if missing:
        print(f"   Missing {len(missing)} grid_id values, for example: {sorted(missing)[:5]}")
    else:
        print(f"   ✓ grid_id complete (0 ~ {EXPECTED_ROWS - 1})")

    # ============ Step 3: Column-name check ============
    print(f"\n[3/6] Column-name check")
    expected_cols = ['grid_id']
    for band in EXPECTED_BANDS:
        for stat in EXPECTED_STATS:
            expected_cols.append(f'{band}_{stat}')

    missing_cols = [c for c in expected_cols if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected_cols]

    if missing_cols:
        print(f"   Missing {len(missing_cols)} columns, for example: {missing_cols[:5]}")
    else:
        print("   All expected columns are present")

    if extra_cols:
        print(f"   Extra columns ({len(extra_cols)}): {extra_cols[:5]}{'...' if len(extra_cols)>5 else ''}")

    # ============ Step 4: Missing-value check ============
    print(f"\n[4/6] Missing-value check (missing rate for each feature mean)")
    null_summary = []
    for band in EXPECTED_BANDS:
        col = f'{band}_mean'
        if col in df.columns:
            n_null = df[col].isna().sum()
            pct = n_null / n_rows * 100
            null_summary.append((band, n_null, pct))
            flag = "⚠" if pct > 5 else ("ℹ" if pct > 0 else "✓")
            print(f"   {flag} {band:<10} missing: {n_null:>6d}  ({pct:>5.1f}%)")

    high_null = [b for b, _, p in null_summary if p > 20]
    if high_null:
        print(f"\n   High missing rate (>20%) features: {high_null}")
        print("     Possible cause: heavy cloud cover or invalid LST over water.")

    # ============ Step 5: plausible value ranges ============
    print("\n[5/6] Plausible value ranges (mean columns only)")
    print(f"   {'Feature':<10} {'Observed range':<25} {'Expected range':<20} {'Status'}")
    print(f"   " + "-" * 65)

    for band in EXPECTED_BANDS:
        col = f'{band}_mean'
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            print(f"   {band:<10} (all missing)")
            continue

        actual_min, actual_max = vals.min(), vals.max()
        expected_min, expected_max = VALUE_RANGES.get(band, (None, None))

        # Check whether values are within the expected range
        if expected_min is None:
            status = "—"
        elif actual_min >= expected_min and actual_max <= expected_max:
            status = "✓"
        elif actual_min < expected_min - abs(expected_min) * 0.5 or actual_max > expected_max * 1.5:
            status = "⚠ severe anomaly"
        else:
            status = "slightly outside"

        print(f"   {band:<10} {actual_min:>10.3f} ~ {actual_max:>10.3f}   "
              f"{expected_min:>6} ~ {expected_max:<6}   {status}")

    # ============ Step 6: sample inspection ============
    print(f"\n[6/6] Five-row sample (inspect values)")
    sample_cols = ['grid_id', 'SR_B4_mean', 'SR_B5_mean', 'NDVI_mean',
                   'LST_mean', 'NTL_mean', 'DEM_mean']
    sample_cols = [c for c in sample_cols if c in df.columns]
    print(df[sample_cols].sample(5, random_state=42).to_string(index=False))

    # ============ Summary ============
    print(f"\n{'=' * 70}")
    print(f"Summary judgment")
    print(f"{'=' * 70}")

    issues = []
    if n_rows != EXPECTED_ROWS:
        issues.append(f"Row count mismatch ({n_rows} vs {EXPECTED_ROWS})")
    if missing_cols:
        issues.append(f"Missing {len(missing_cols)} columns")
    if grid_ids.duplicated().sum() > 0:
        issues.append("grid_id has duplicates")
    high_null_critical = [b for b, _, p in null_summary if p > 30]
    if high_null_critical:
        issues.append(f"Severe missingness (>30%): {high_null_critical}")

    if not issues:
        print("\n   ✓ All checks passed; the CSV can be used for model training.")
        print("   Next: use this CSV path for the model-training step.")
    else:
        print("\n   Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"     {i}. {issue}")
        print("\n   Recommendation: review this output before model training.")


if __name__ == '__main__':
    main()
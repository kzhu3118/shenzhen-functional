#!/usr/bin/env python3
"""MODIS LST analysis for joint MLP five-class Shenzhen land-use results.

Inputs:
  - joint MLP five-class GPKG from step4e
  - MODIS summer LST CSVs
  - yearly feature CSVs for PSM covariates

Outputs:
  - Table1 class LST by year
  - Table2 pairwise LST contrasts
  - Table3 conversion dLST
  - Table4 background-normalized dLST
  - Table5 PSM for key transitions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLES_GPKG = (
    PROJECT_ROOT
    / "data/mlp_multiyear_5class_joint_tculuWeak_manual2005_gaia_thr10/shenzhen_5class_joint_mlp_multiyear.gpkg"
)
PLES_LAYER = "pred5_joint_mlp"
LST_DIR = PROJECT_ROOT / "data/GEE/shenzhen_features_2000_2020"
FEATURE_DIR = LST_DIR
OUT_DIR = PROJECT_ROOT / "data/thermal_joint_mlp_5class_modis"

YEARS = [2000, 2005, 2010, 2015, 2020]
STAGES = [(2000, 2005), (2005, 2010), (2010, 2015), (2015, 2020)]
CLASSES = ["production", "living", "green_space", "water", "farmland"]

PAIRWISE_CONTRASTS = [
    ("production", "green_space"),
    ("living", "green_space"),
    ("production", "living"),
    ("production", "water"),
    ("living", "water"),
    ("farmland", "green_space"),
    ("farmland", "living"),
    ("production", "farmland"),
]

KEY_PSM_COMPARISONS = [
    ("green_space_to_living", "green_space_stable"),
    ("green_space_to_production", "green_space_stable"),
    ("farmland_to_living", "farmland_stable"),
    ("farmland_to_production", "farmland_stable"),
    ("water_to_living", "water_stable"),
    ("water_to_production", "water_stable"),
    ("living_to_production", "living_stable"),
    ("production_to_living", "production_stable"),
]

PSM_COVARIATES = ["DEM_mean", "slope_mean", "NDVI_mean", "NDBI_mean", "MNDWI_mean"]
N_BOOTSTRAP = 1000
RANDOM_STATE = 42
MIN_N_FOR_TABLE = 30
MIN_N_FOR_PSM = 80
CALIPER_FACTOR = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--min-n-table", type=int, default=MIN_N_FOR_TABLE)
    parser.add_argument("--min-n-psm", type=int, default=MIN_N_FOR_PSM)
    return parser.parse_args()


def load_modis_lst(year: int) -> pd.DataFrame:
    path = LST_DIR / f"shenzhen_LST_MODIS_{year}.csv"
    df = pd.read_csv(path, usecols=["grid_id", "LST_mean"])
    df["LST_mean"] = df["LST_mean"] - 273.15
    return df.rename(columns={"LST_mean": f"LST_{year}"})


def load_covariates(year: int, master: pd.DataFrame) -> pd.DataFrame:
    path = FEATURE_DIR / ("shenzhen_features_2020.csv" if year == 2020 else f"shenzhen_features_{year}_noNTL.csv")
    use = ["grid_id"] + PSM_COVARIATES
    cov = pd.read_csv(path, usecols=lambda c: c in use)
    cov = cov.rename(columns={c: f"{c}_t0" for c in PSM_COVARIATES if c in cov.columns})
    cov = cov.merge(master[["grid_id", f"LST_{year}"]], on="grid_id", how="left")
    return cov.rename(columns={f"LST_{year}": "MODIS_LST_t0"})


def build_master_table() -> pd.DataFrame:
    print("[1] Load joint MLP five-class labels...")
    gdf = gpd.read_file(PLES_GPKG, layer=PLES_LAYER)
    cols = ["grid_id"] + [f"mlp_pred5_{y}" for y in YEARS]
    df = pd.DataFrame(gdf[cols]).copy()
    for year in YEARS:
        src = f"mlp_pred5_{year}"
        dst = f"class_{year}"
        df[dst] = df[src].where(df[src].isin(CLASSES))
        print(f"    {year}: valid labelled grids {df[dst].notna().sum()}/{len(df)}")

    print("[2] Load MODIS summer LST...")
    for year in YEARS:
        lst = load_modis_lst(year)
        df = df.merge(lst, on="grid_id", how="left")
        print(f"    {year}: valid LST {df[f'LST_{year}'].notna().sum()}/{len(df)}")
    return df


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int = RANDOM_STATE) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int = RANDOM_STATE) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    diff = a.mean() - b.mean()
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = rng.choice(a, size=len(a), replace=True).mean() - rng.choice(b, size=len(b), replace=True).mean()
    return float(diff), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def define_conversion(row: pd.Series, t0: int, t1: int) -> str | None:
    c0 = row.get(f"class_{t0}")
    c1 = row.get(f"class_{t1}")
    if c0 not in CLASSES or c1 not in CLASSES:
        return None
    if c0 == c1:
        return f"{c0}_stable"
    return f"{c0}_to_{c1}"


def class_lst_table(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        for cls, grp in df.dropna(subset=[f"class_{year}", f"LST_{year}"]).groupby(f"class_{year}"):
            vals = grp[f"LST_{year}"].to_numpy()
            mean, lo, hi = bootstrap_mean_ci(vals, n_boot)
            rows.append(
                {
                    "year": year,
                    "class": cls,
                    "n_grids": len(grp),
                    "LST_mean": round(mean, 3),
                    "LST_ci_lo": round(lo, 3),
                    "LST_ci_hi": round(hi, 3),
                    "LST_median": round(float(np.nanmedian(vals)), 3),
                    "LST_std": round(float(np.nanstd(vals, ddof=1)), 3) if len(vals) > 1 else np.nan,
                    "LST_q25": round(float(np.nanpercentile(vals, 25)), 3),
                    "LST_q75": round(float(np.nanpercentile(vals, 75)), 3),
                }
            )
    return pd.DataFrame(rows).sort_values(["year", "class"])


def pairwise_contrast_table(df: pd.DataFrame, n_boot: int, min_n: int) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        sub = df.dropna(subset=[f"class_{year}", f"LST_{year}"])
        for a, b in PAIRWISE_CONTRASTS:
            av = sub.loc[sub[f"class_{year}"] == a, f"LST_{year}"].to_numpy()
            bv = sub.loc[sub[f"class_{year}"] == b, f"LST_{year}"].to_numpy()
            if len(av) < min_n or len(bv) < min_n:
                continue
            diff, lo, hi = bootstrap_diff_ci(av, bv, n_boot)
            rows.append(
                {
                    "year": year,
                    "contrast": f"{a}_minus_{b}",
                    "class_a": a,
                    "class_b": b,
                    "n_a": len(av),
                    "n_b": len(bv),
                    "dLST_mean": round(diff, 3),
                    "dLST_ci_lo": round(lo, 3),
                    "dLST_ci_hi": round(hi, 3),
                }
            )
    return pd.DataFrame(rows).sort_values(["year", "contrast"])


def conversion_table(df: pd.DataFrame, t0: int, t1: int, n_boot: int, min_n: int) -> pd.DataFrame:
    sub = df.copy()
    sub["conversion"] = sub.apply(lambda r: define_conversion(r, t0, t1), axis=1)
    sub = sub.dropna(subset=["conversion", f"LST_{t0}", f"LST_{t1}"])
    sub[f"dLST_{t0}_{t1}"] = sub[f"LST_{t1}"] - sub[f"LST_{t0}"]

    rows = []
    for conv, grp in sub.groupby("conversion"):
        if len(grp) < min_n:
            continue
        vals = grp[f"dLST_{t0}_{t1}"].to_numpy()
        mean, lo, hi = bootstrap_mean_ci(vals, n_boot)
        source, target = conv.split("_to_") if "_to_" in conv else (conv.replace("_stable", ""), conv.replace("_stable", ""))
        rows.append(
            {
                "stage": f"{t0}_{t1}",
                "source_class": source,
                "target_class": target,
                "conversion": conv,
                "n_grids": len(grp),
                f"LST_{t0}_mean": round(float(grp[f"LST_{t0}"].mean()), 3),
                f"LST_{t1}_mean": round(float(grp[f"LST_{t1}"].mean()), 3),
                "dLST_mean": round(mean, 3),
                "dLST_ci_lo": round(lo, 3),
                "dLST_ci_hi": round(hi, 3),
            }
        )
    return pd.DataFrame(rows).sort_values(["stage", "source_class", "target_class"])


def background_normalized_table(df: pd.DataFrame, t0: int, t1: int, n_boot: int, min_n: int) -> pd.DataFrame:
    sub = df.copy()
    sub["conversion"] = sub.apply(lambda r: define_conversion(r, t0, t1), axis=1)
    sub = sub.dropna(subset=["conversion", f"LST_{t0}", f"LST_{t1}"])
    sub[f"dLST_{t0}_{t1}"] = sub[f"LST_{t1}"] - sub[f"LST_{t0}"]

    baseline = sub[sub["conversion"].isin(["green_space_stable", "water_stable"])][f"dLST_{t0}_{t1}"].to_numpy()
    if len(baseline) < min_n:
        baseline = sub[sub["conversion"] == "green_space_stable"][f"dLST_{t0}_{t1}"].to_numpy()
    if len(baseline) < min_n:
        return pd.DataFrame()

    baseline_mean = float(np.nanmean(baseline))
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for conv, grp in sub.groupby("conversion"):
        if len(grp) < min_n:
            continue
        vals = grp[f"dLST_{t0}_{t1}"].to_numpy()
        boot = np.empty(n_boot)
        for i in range(n_boot):
            boot[i] = rng.choice(vals, size=len(vals), replace=True).mean() - rng.choice(baseline, size=len(baseline), replace=True).mean()
        rows.append(
            {
                "stage": f"{t0}_{t1}",
                "conversion": conv,
                "n_grids": len(grp),
                "raw_dLST": round(float(np.nanmean(vals)), 3),
                "background_dLST": round(baseline_mean, 3),
                "attributed_dLST": round(float(np.nanmean(vals) - baseline_mean), 3),
                "attributed_ci_lo": round(float(np.percentile(boot, 2.5)), 3),
                "attributed_ci_hi": round(float(np.percentile(boot, 97.5)), 3),
                "background_reference": "green_space_stable_or_water_stable",
            }
        )
    return pd.DataFrame(rows).sort_values(["stage", "conversion"])


def psm_compare(
    master: pd.DataFrame,
    covariates: pd.DataFrame,
    t0: int,
    t1: int,
    treated_conv: str,
    control_conv: str,
    n_boot: int,
    min_n: int,
) -> dict[str, Any]:
    df = master.copy()
    df["conversion"] = df.apply(lambda r: define_conversion(r, t0, t1), axis=1)
    df = df[df["conversion"].isin([treated_conv, control_conv])].copy()
    df = df.dropna(subset=[f"LST_{t0}", f"LST_{t1}"])
    df = df.merge(covariates, on="grid_id", how="inner")
    df["treated"] = (df["conversion"] == treated_conv).astype(int)
    df[f"dLST_{t0}_{t1}"] = df[f"LST_{t1}"] - df[f"LST_{t0}"]

    cov_cols = [f"{c}_t0" for c in PSM_COVARIATES if f"{c}_t0" in df.columns] + ["MODIS_LST_t0"]
    df = df.dropna(subset=cov_cols + [f"dLST_{t0}_{t1}"])
    n_treated = int(df["treated"].sum())
    n_control = int((1 - df["treated"]).sum())
    out = {
        "stage": f"{t0}_{t1}",
        "treated": treated_conv,
        "control": control_conv,
        "n_treated_pre": n_treated,
        "n_control_pre": n_control,
    }
    if n_treated < min_n or n_control < min_n:
        out["status"] = "skipped_insufficient_samples"
        return out

    X = df[cov_cols].to_numpy()
    y = df["treated"].to_numpy()
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    lr.fit(Xs, y)
    ps = np.clip(lr.predict_proba(Xs)[:, 1], 1e-6, 1 - 1e-6)
    df["propensity"] = ps

    treated = df[df["treated"] == 1].reset_index(drop=True)
    control = df[df["treated"] == 0].reset_index(drop=True)
    treated_logit = np.log(treated["propensity"] / (1 - treated["propensity"])).to_numpy().reshape(-1, 1)
    control_logit = np.log(control["propensity"] / (1 - control["propensity"])).to_numpy().reshape(-1, 1)
    caliper = CALIPER_FACTOR * float(np.std(np.log(ps / (1 - ps))))

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control_logit)
    dist, idx = nn.kneighbors(treated_logit)
    ok = dist.flatten() < caliper
    mt = treated.loc[ok].reset_index(drop=True)
    mc = control.iloc[idx.flatten()[ok]].reset_index(drop=True)
    if len(mt) < min_n:
        out.update({"status": "skipped_few_matched", "n_matched": int(len(mt))})
        return out

    att_values = mt[f"dLST_{t0}_{t1}"].to_numpy() - mc[f"dLST_{t0}_{t1}"].to_numpy()
    att, lo, hi = bootstrap_mean_ci(att_values, n_boot)
    out.update(
        {
            "status": "ok",
            "n_matched": int(len(mt)),
            "att_dLST": round(att, 3),
            "att_ci_lo": round(lo, 3),
            "att_ci_hi": round(hi, 3),
            "treated_raw_dLST": round(float(mt[f"dLST_{t0}_{t1}"].mean()), 3),
            "control_raw_dLST": round(float(mc[f"dLST_{t0}_{t1}"].mean()), 3),
            "caliper": round(float(caliper), 4),
            "covariates": ",".join(cov_cols),
        }
    )
    return out


def psm_table(master: pd.DataFrame, n_boot: int, min_n: int) -> pd.DataFrame:
    rows = []
    for t0, t1 in STAGES:
        cov = load_covariates(t0, master)
        for treated, control in KEY_PSM_COMPARISONS:
            rows.append(psm_compare(master, cov, t0, t1, treated, control, n_boot, min_n))
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Step 5: Joint MLP five-class MODIS thermal analysis")
    print("=" * 70)
    master = build_master_table()
    master.to_csv(args.out_dir / "master_joint_mlp_5class_lst_table.csv", index=False, encoding="utf-8-sig")

    print("\n[Table1] class LST by year")
    t1 = class_lst_table(master, args.n_bootstrap)
    print(t1.to_string(index=False))
    t1.to_csv(args.out_dir / "Table1_joint_mlp_5class_lst_by_year.csv", index=False, encoding="utf-8-sig")

    print("\n[Table2] pairwise contrasts")
    t2 = pairwise_contrast_table(master, args.n_bootstrap, args.min_n_table)
    print(t2.to_string(index=False))
    t2.to_csv(args.out_dir / "Table2_joint_mlp_5class_pairwise_contrasts.csv", index=False, encoding="utf-8-sig")

    print("\n[Table3] conversion dLST")
    conv_tables = [conversion_table(master, t0, t1, args.n_bootstrap, args.min_n_table) for t0, t1 in STAGES]
    t3 = pd.concat([x for x in conv_tables if not x.empty], ignore_index=True)
    print(t3.to_string(index=False))
    t3.to_csv(args.out_dir / "Table3_joint_mlp_5class_conversion_dLST.csv", index=False, encoding="utf-8-sig")

    print("\n[Table4] background-normalized dLST")
    bg_tables = [background_normalized_table(master, t0, t1, args.n_bootstrap, args.min_n_table) for t0, t1 in STAGES]
    t4 = pd.concat([x for x in bg_tables if not x.empty], ignore_index=True)
    print(t4.to_string(index=False))
    t4.to_csv(args.out_dir / "Table4_joint_mlp_5class_background_normalized_dLST.csv", index=False, encoding="utf-8-sig")

    print("\n[Table5] PSM for key transitions")
    t5 = psm_table(master, args.n_bootstrap, args.min_n_psm)
    print(t5.to_string(index=False))
    t5.to_csv(args.out_dir / "Table5_joint_mlp_5class_key_transition_psm.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "ples_gpkg": str(PLES_GPKG),
        "ples_layer": PLES_LAYER,
        "lst_dir": str(LST_DIR),
        "feature_dir": str(FEATURE_DIR),
        "years": YEARS,
        "stages": STAGES,
        "classes": CLASSES,
        "n_bootstrap": args.n_bootstrap,
        "min_n_table": args.min_n_table,
        "min_n_psm": args.min_n_psm,
        "note": "Uses GAIA-adjusted mlp_pred5_YEAR labels from joint MLP five-class multiyear prediction.",
    }
    with open(args.out_dir / "metadata_joint_mlp_5class_thermal_modis.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nSaved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()

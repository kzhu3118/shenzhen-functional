#!/usr/bin/env python3
"""Batch compare TCULU production weights for 5-class RF/MLP experiments.

Runs, for each requested TCULU production weight:
  1. 16_fuse_multisource_labels_2020.py
  2. 18_train_mlp_multisource_2020.py

Then collects model summaries and class reports into one comparison CSV.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUSE_SCRIPT = PROJECT_ROOT / "code/16_fuse_multisource_labels_2020.py"
TRAIN_SCRIPT = PROJECT_ROOT / "code/18_train_mlp_multisource_2020.py"
BASE_OUT = PROJECT_ROOT / "data/multisource_labels_2020_5class_weight_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="1.0,1.2,1.5")
    parser.add_argument("--out-root", type=Path, default=BASE_OUT)
    parser.add_argument("--drop-ntl", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def weight_tag(w: float) -> str:
    return str(w).replace(".", "p")


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def collect_one(run_dir: Path, weight: float) -> list[dict]:
    model_dir = run_dir / "model_test_mlp"
    summary = pd.read_csv(model_dir / "model_summary.csv")
    rows = []
    for _, s in summary.iterrows():
        model = s["model"]
        report_path = model_dir / f"classification_report_{model}.csv"
        report = pd.read_csv(report_path, index_col=0)
        row = {
            "tculu_production_weight": weight,
            "model": model,
            "accuracy": float(s["accuracy"]),
            "f1_macro": float(s["f1_macro"]),
            "f1_weighted": float(s["f1_weighted"]),
        }
        for cls in ["production", "living", "green_space", "water", "farmland"]:
            if cls in report.index:
                row[f"{cls}_precision"] = float(report.loc[cls, "precision"])
                row[f"{cls}_recall"] = float(report.loc[cls, "recall"])
                row[f"{cls}_f1"] = float(report.loc[cls, "f1-score"])
                row[f"{cls}_support"] = float(report.loc[cls, "support"])
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    weights = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
    all_rows = []

    for w in weights:
        tag = weight_tag(w)
        run_dir = args.out_root / f"tculuProd_{tag}"
        label_gpkg = run_dir / "multisource_labels_2020_5class.gpkg"
        model_dir = run_dir / "model_test_mlp"

        fuse_cmd = [
            args.python,
            str(FUSE_SCRIPT),
            "--tculu-production-weight",
            str(w),
            "--out-dir",
            str(run_dir),
        ]
        run(fuse_cmd)

        train_cmd = [
            args.python,
            str(TRAIN_SCRIPT),
            "--label-gpkg",
            str(label_gpkg),
            "--out-dir",
            str(model_dir),
        ]
        if args.drop_ntl:
            train_cmd.append("--drop-ntl")
        run(train_cmd)

        all_rows.extend(collect_one(run_dir, w))

    comparison = pd.DataFrame(all_rows)
    comparison_path = args.out_root / "rf_mlp_weight_sweep_summary.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    print("\nComparison:")
    cols = [
        "tculu_production_weight",
        "model",
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "production_precision",
        "production_recall",
        "production_f1",
        "living_f1",
        "green_space_f1",
    ]
    print(comparison[cols].round(4).to_string(index=False))
    print(f"\nSaved: {comparison_path}")


if __name__ == "__main__":
    main()

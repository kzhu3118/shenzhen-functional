#!/usr/bin/env python3
"""Train RF and MLP baselines with fused multi-source 2020 labels.

This is a low-cost experiment to test whether a tabular neural network can
improve over RF for fine functional land-use classification. It uses CSV
features rather than image patches, so the MLP is a tabular baseline rather
than a semantic-segmentation model.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = PROJECT_ROOT / "data/GEE/shenzhen_features_2000_2020/shenzhen_features_2020.csv"
LABEL_GPKG = PROJECT_ROOT / "data/multisource_labels_2020_5class/multisource_labels_2020_5class.gpkg"
OUT_DIR = PROJECT_ROOT / "data/multisource_labels_2020_5class/model_test_mlp"

RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", type=Path, default=FEATURES_CSV)
    parser.add_argument("--label-gpkg", type=Path, default=LABEL_GPKG)
    parser.add_argument("--label-layer", default="labels")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--min-confidence", type=float, default=0.40)
    parser.add_argument("--min-weight", type=float, default=0.20)
    parser.add_argument("--drop-ntl", action="store_true")
    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument("--mlp-hidden", default="128,64")
    parser.add_argument("--mlp-max-iter", type=int, default=500)
    return parser.parse_args()


def parse_hidden(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def load_training_data(args: argparse.Namespace):
    features = pd.read_csv(args.features_csv)
    labels_gdf = gpd.read_file(args.label_gpkg, layer=args.label_layer)
    labels = pd.DataFrame(labels_gdf.drop(columns="geometry"))
    df = features.merge(labels, on="grid_id", how="inner")

    feature_cols = [c for c in features.columns if c != "grid_id"]
    if args.drop_ntl:
        feature_cols = [c for c in feature_cols if not c.startswith("NTL_")]
    df = df.dropna(subset=feature_cols).copy()
    df = df[
        df["fused_label"].notna()
        & (df["fused_confidence"] >= args.min_confidence)
        & (df["sample_weight"] >= args.min_weight)
    ].copy()
    df["fused_label"] = df["fused_label"].astype(str)
    return df, feature_cols


def fit_with_optional_weights(model, X, y, sample_weight):
    """Fit estimator/pipeline with sample weights when supported."""
    try:
        if isinstance(model, Pipeline):
            final_name = model.steps[-1][0]
            final_est = model.steps[-1][1]
            if "sample_weight" in inspect.signature(final_est.fit).parameters:
                model.fit(X, y, **{f"{final_name}__sample_weight": sample_weight})
            else:
                model.fit(X, y)
        elif "sample_weight" in inspect.signature(model.fit).parameters:
            model.fit(X, y, sample_weight=sample_weight)
        else:
            model.fit(X, y)
    except TypeError:
        model.fit(X, y)
    return model


def evaluate_model(name: str, model, X_test, y_test, label_encoder: LabelEncoder) -> dict:
    pred_encoded = model.predict(X_test)
    y_true = label_encoder.inverse_transform(y_test)
    pred = label_encoder.inverse_transform(pred_encoded)
    labels = label_encoder.classes_.tolist()
    report = classification_report(y_true, pred, labels=labels, output_dict=True, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(y_true, pred, labels=labels), index=labels, columns=labels)
    return {
        "name": name,
        "pred": pred,
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted")),
        "report": report,
        "confusion_matrix": cm,
    }


def source_evaluation(pred_df: pd.DataFrame, pred_col: str, label_col: str, classes: list[str]) -> dict:
    sub = pred_df[pred_df[label_col].isin(classes) & pred_df[pred_col].isin(classes)].copy()
    if sub.empty:
        return {"n": 0}
    return {
        "n": int(len(sub)),
        "accuracy": float(accuracy_score(sub[label_col], sub[pred_col])),
        "f1_macro": float(f1_score(sub[label_col], sub[pred_col], labels=classes, average="macro", zero_division=0)),
        "report": classification_report(sub[label_col], sub[pred_col], labels=classes, output_dict=True, zero_division=0),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Load fused labels and features")
    df, feature_cols = load_training_data(args)
    classes = sorted(df["fused_label"].unique().tolist())
    print(f"    samples: {len(df)}")
    print(f"    feature count: {len(feature_cols)}")
    print("    class distribution:")
    print(df["fused_label"].value_counts().to_string())

    X = df[feature_cols].to_numpy()
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    y = label_encoder.transform(df["fused_label"].to_numpy())
    sw = df["sample_weight"].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test, sw_train, sw_test, idx_train, idx_test = train_test_split(
        X,
        y,
        sw,
        df.index.to_numpy(),
        test_size=args.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = {
        "rf": RandomForestClassifier(
            n_estimators=args.rf_trees,
            class_weight="balanced",
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "mlp": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=parse_hidden(args.mlp_hidden),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        batch_size=256,
                        learning_rate_init=1e-3,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=25,
                        max_iter=args.mlp_max_iter,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    results = {}
    test_rows = df.loc[idx_test, ["grid_id", "fused_label", "poi_label", "euluc_label", "tculu_label"]].copy()
    for name, model in models.items():
        print(f"\n[2] Training {name}")
        fit_with_optional_weights(model, X_train, y_train, sw_train)
        joblib.dump(model, args.out_dir / f"{name}_multisource_2020.joblib")
        res = evaluate_model(name, model, X_test, y_test, label_encoder)
        results[name] = res
        test_rows[f"pred_{name}"] = res["pred"]
        print(f"    accuracy={res['accuracy']:.4f} f1_macro={res['f1_macro']:.4f} f1_weighted={res['f1_weighted']:.4f}")
        print(classification_report(label_encoder.inverse_transform(y_test), res["pred"], labels=classes, zero_division=0))
        res["confusion_matrix"].to_csv(args.out_dir / f"confusion_matrix_{name}.csv", encoding="utf-8-sig")
        pd.DataFrame(res["report"]).T.to_csv(args.out_dir / f"classification_report_{name}.csv", encoding="utf-8-sig")

    test_rows.to_csv(args.out_dir / "heldout_predictions.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    source_reports = {}
    for name, res in results.items():
        row = {
            "model": name,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": res["accuracy"],
            "f1_macro": res["f1_macro"],
            "f1_weighted": res["f1_weighted"],
        }
        summary_rows.append(row)
        source_reports[name] = {
            "vs_poi": source_evaluation(test_rows, f"pred_{name}", "poi_label", classes),
            "vs_euluc": source_evaluation(test_rows, f"pred_{name}", "euluc_label", classes),
            "vs_tculu": source_evaluation(test_rows, f"pred_{name}", "tculu_label", classes),
        }

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "model_summary.csv", index=False, encoding="utf-8-sig")
    with open(args.out_dir / "source_evaluation_reports.json", "w", encoding="utf-8") as f:
        json.dump(source_reports, f, ensure_ascii=False, indent=2)

    metadata = {
        "features_csv": str(args.features_csv),
        "label_gpkg": str(args.label_gpkg),
        "feature_count": len(feature_cols),
        "classes": classes,
        "args": vars(args),
        "note": "MLP is sklearn tabular MLP. geoenv currently has no torch/tensorflow installed.",
    }
    with open(args.out_dir / "metadata_train_mlp_multisource_2020.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

    print("\nSummary:")
    print(summary.round(4).to_string(index=False))
    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()

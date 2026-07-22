#!/usr/bin/env python3
"""Train a PyTorch FT-Transformer-style tabular model for 2020 five-class labels.

This script reads CSV files only, so it can run in a lightweight PyTorch
environment such as `dlgeo` without geopandas/rasterio.

The implementation treats each numeric feature as a token:
  token_i = feature_value_i * feature_embedding_i + feature_bias_i
Then a Transformer encoder pools the tokens through a CLS token.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = PROJECT_ROOT / "data/GEE/shenzhen_features_2000_2020/shenzhen_features_2020.csv"
LABEL_CSV = PROJECT_ROOT / "data/multisource_labels_2020_5class/multisource_labels_2020_5class.csv"
OUT_DIR = PROJECT_ROOT / "data/multisource_labels_2020_5class/model_test_fttransformer"

RANDOM_STATE = 42


class NumericFeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_token: int = 64,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.15,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.tokenizer = NumericFeatureTokenizer(n_features, d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, n_classes),
        )
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        cls_out = self.norm(encoded[:, 0])
        return self.head(cls_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", type=Path, default=FEATURES_CSV)
    parser.add_argument("--label-csv", type=Path, default=LABEL_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--min-confidence", type=float, default=0.40)
    parser.add_argument("--min-weight", type=float, default=0.20)
    parser.add_argument("--drop-ntl", action="store_true")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-token", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--monitor", choices=["val_f1_macro", "val_loss"], default="val_f1_macro")
    parser.add_argument("--num-threads", type=int, default=0, help="PyTorch intra-op CPU threads. 0 uses os.cpu_count().")
    parser.add_argument("--num-inter-op-threads", type=int, default=0, help="PyTorch inter-op CPU threads. 0 keeps PyTorch default.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. For in-memory tensors, 0 is often fastest.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data(args: argparse.Namespace):
    features = pd.read_csv(args.features_csv)
    labels = pd.read_csv(args.label_csv)
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


def configure_torch_threads(num_threads: int, num_inter_op_threads: int) -> None:
    if num_threads == 0:
        num_threads = os.cpu_count() or torch.get_num_threads()
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    if num_inter_op_threads > 0:
        # Must be set before parallel work starts.
        torch.set_num_interop_threads(num_inter_op_threads)


def make_loader(X, y, w, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(w, dtype=torch.float32),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for xb, _, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> float:
    model.eval()
    losses = []
    weights = []
    for xb, yb, wb in loader:
        xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append((loss * wb).sum().item())
        weights.append(wb.sum().item())
    return float(sum(losses) / max(sum(weights), 1e-8))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(RANDOM_STATE)
    configure_torch_threads(args.num_threads, args.num_inter_op_threads)
    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(
        "PyTorch CPU threads: "
        f"intra_op={torch.get_num_threads()} inter_op={torch.get_num_interop_threads()} "
        f"data_workers={args.num_workers}"
    )

    df, feature_cols = load_data(args)
    le = LabelEncoder()
    y_all = le.fit_transform(df["fused_label"].to_numpy())
    classes = le.classes_.tolist()
    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    w_all = df["sample_weight"].to_numpy(dtype=np.float32)

    idx_trainval, idx_test = train_test_split(
        np.arange(len(df)),
        test_size=args.test_size,
        stratify=y_all,
        random_state=RANDOM_STATE,
    )
    y_trainval = y_all[idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=0.15,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_all[idx_train]).astype(np.float32)
    X_val = scaler.transform(X_all[idx_val]).astype(np.float32)
    X_test = scaler.transform(X_all[idx_test]).astype(np.float32)
    y_train, y_val, y_test = y_all[idx_train], y_all[idx_val], y_all[idx_test]
    w_train, w_val, w_test = w_all[idx_train], w_all[idx_val], w_all[idx_test]

    class_counts = np.bincount(y_train, minlength=len(classes)).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    train_loader = make_loader(X_train, y_train, w_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(X_val, y_val, w_val, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(X_test, y_test, w_test, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = FTTransformer(
        n_features=len(feature_cols),
        n_classes=len(classes),
        d_token=args.d_token,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion_none = nn.CrossEntropyLoss(weight=class_weights_t, reduction="none")

    best_metric = math.inf if args.monitor == "val_loss" else -math.inf
    best_epoch = 0
    best_state = None
    wait = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_num = 0.0
        train_weight_den = 0.0
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss_vec = criterion_none(logits, yb)
            loss = (loss_vec * wb).sum() / wb.sum().clamp_min(1e-8)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_num += (loss_vec.detach() * wb).sum().item()
            train_weight_den += wb.sum().item()

        train_loss = train_loss_num / max(train_weight_den, 1e-8)
        val_loss = evaluate_loss(model, val_loader, criterion_none, device)
        val_pred = predict(model, val_loader, device)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_f1_macro": val_f1})

        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

        monitor_value = val_loss if args.monitor == "val_loss" else val_f1
        improved = (
            monitor_value < best_metric - args.min_delta
            if args.monitor == "val_loss"
            else monitor_value > best_metric + args.min_delta
        )
        if improved:
            best_metric = monitor_value
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best {args.monitor}={best_metric:.4f} at epoch {best_epoch}"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred = predict(model, test_loader, device)
    acc = accuracy_score(y_test, test_pred)
    f1_macro = f1_score(y_test, test_pred, average="macro")
    f1_weighted = f1_score(y_test, test_pred, average="weighted")
    report = classification_report(y_test, test_pred, target_names=classes, output_dict=True, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(y_test, test_pred), index=classes, columns=classes)

    print("\nTest:")
    print(f"accuracy={acc:.4f} f1_macro={f1_macro:.4f} f1_weighted={f1_weighted:.4f}")
    print(classification_report(y_test, test_pred, target_names=classes, zero_division=0))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "feature_cols": feature_cols,
            "args": vars(args),
        },
        args.out_dir / "fttransformer_2020_5class.pt",
    )
    joblib.dump(scaler, args.out_dir / "scaler.joblib")
    joblib.dump(le, args.out_dir / "label_encoder.joblib")

    pd.DataFrame(history).to_csv(args.out_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(report).T.to_csv(args.out_dir / "classification_report_fttransformer.csv", encoding="utf-8-sig")
    cm.to_csv(args.out_dir / "confusion_matrix_fttransformer.csv", encoding="utf-8-sig")
    pd.DataFrame(
        {
            "grid_id": df.iloc[idx_test]["grid_id"].to_numpy(),
            "y_true": le.inverse_transform(y_test),
            "y_pred": le.inverse_transform(test_pred),
            "sample_weight": w_test,
        }
    ).to_csv(args.out_dir / "heldout_predictions_fttransformer.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model": "fttransformer",
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "classes": classes,
        "feature_count": len(feature_cols),
        "device": str(device),
        "monitor": args.monitor,
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "num_workers": int(args.num_workers),
    }
    pd.DataFrame([summary]).to_csv(args.out_dir / "model_summary_fttransformer.csv", index=False, encoding="utf-8-sig")
    with open(args.out_dir / "metadata_fttransformer.json", "w", encoding="utf-8") as f:
        json.dump(summary | {"args": vars(args)}, f, ensure_ascii=False, indent=2, default=str)

    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()

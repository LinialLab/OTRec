"""Temporal prospective evaluation for the frozen BioClinical-ModernBERT + MLP baseline.

Protocol matches run_temporal_repeated.py:
- Train on OTP 2022.02  (history_df.parquet)
- Test on temporal anti-join of OTP 2025.06 vs 2022.02 (final_df.parquet)
- Repeat MLP training with 5 random seeds (42-46)
- Early stopping on a validation set carved from *training* targets only
- The test set is never seen during training or model selection
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from run_frozen_encoder_mlp import (
    PairIndexDataset,
    PairMLP,
    build_sentence_transformer,
    encode_unique,
    evaluate_model,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = ROOT.parent / "code" / "history_df.parquet"
DEFAULT_FUTURE = ROOT.parent / "code" / "final_df.parquet"
DEFAULT_DISEASE = ROOT.parent / "code" / "copy_proc" / "disease_df.parquet"
DEFAULT_TARGET = ROOT.parent / "code" / "copy_proc" / "target_df.parquet"
DEFAULT_OUT = ROOT / "Outputs" / "temporal_frozen_encoder_mlp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--future", type=Path, default=DEFAULT_FUTURE)
    parser.add_argument("--disease", type=Path, default=DEFAULT_DISEASE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-name", type=str, default="thomas-sounack/BioClinical-ModernBERT-base")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--val-size", type=float, default=0.1)
    return parser.parse_args()


def merge_text(df, disease_df, target_df):
    out = df.merge(disease_df[["diseaseId", "disease_text_embed"]], on="diseaseId", how="left")
    out = out.merge(target_df[["targetId", "target_text_embed"]], on="targetId", how="left")
    out = out.rename(columns={"disease_text_embed": "disease_text", "target_text_embed": "target_text"})
    out["disease_text"] = out["disease_text"].fillna("").astype(str)
    out["target_text"] = out["target_text"].fillna("").astype(str)
    return out


def build_temporal_test_set(history_df, future_df):
    join_keys = ["diseaseId", "targetId", "label"]
    return (
        future_df.merge(history_df[join_keys], on=join_keys, how="left", indicator=True)
        .query('_merge == "left_only"')
        .drop(columns="_merge")
        .reset_index(drop=True)
    )


def train_mlp_once(train_df, val_df, test_df, disease_embeddings, target_embeddings,
                   disease_to_idx, target_to_idx, args, device, seed):
    set_seed(seed)

    def make_loader(df, shuffle):
        return DataLoader(
            PairIndexDataset(df, disease_to_idx, target_to_idx),
            batch_size=args.train_batch_size, shuffle=shuffle, num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = make_loader(train_df, True)
    val_loader = make_loader(val_df, False)
    test_loader = make_loader(test_df, False)

    classifier = PairMLP(emb_dim=disease_embeddings.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    pos = float(train_df["label"].sum())
    neg = float(len(train_df) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_state = None
    best_val_pr_auc = -float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.max_epochs + 1):
        classifier.train()
        for disease_idx, target_idx, labels in train_loader:
            disease_idx = disease_idx.to(device, non_blocking=True)
            target_idx = target_idx.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(disease_embeddings[disease_idx], target_embeddings[target_idx])
            loss_fn(logits, labels).backward()
            optimizer.step()

        val_metrics = evaluate_model(classifier, val_loader, disease_embeddings, target_embeddings, device, loss_fn)
        scheduler.step(val_metrics.pr_auc)
        print(f"  [seed {seed} epoch {epoch}] val_pr_auc={val_metrics.pr_auc:.4f}", flush=True)

        if val_metrics.pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_metrics.pr_auc
            best_state = {k: v.detach().cpu() for k, v in classifier.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  Early stopping at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        classifier.load_state_dict(best_state)
    test_metrics = evaluate_model(classifier, test_loader, disease_embeddings, target_embeddings, device)
    return test_metrics.roc_auc, test_metrics.pr_auc


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    overall_start = time.perf_counter()

    history_raw = pd.read_parquet(args.history)
    future_raw = pd.read_parquet(args.future)
    disease_df = pd.read_parquet(args.disease)
    target_df = pd.read_parquet(args.target)

    test_raw = build_temporal_test_set(history_raw, future_raw)
    train_df_full = merge_text(history_raw, disease_df, target_df)
    test_df = merge_text(test_raw, disease_df, target_df)

    print(f"Train rows: {len(train_df_full)}, Test rows: {len(test_df)}")
    print(f"Train +rate: {train_df_full['label'].mean():.4f}, Test +rate: {test_df['label'].mean():.4f}")

    all_df = pd.concat([train_df_full, test_df], ignore_index=True)
    encoder = build_sentence_transformer(args.model_name, args.max_seq_length)
    encode_start = time.perf_counter()
    disease_to_idx, disease_embeddings_np, _ = encode_unique(encoder, all_df["diseaseId"], all_df["disease_text"], args.encode_batch_size)
    target_to_idx, target_embeddings_np, _ = encode_unique(encoder, all_df["targetId"], all_df["target_text"], args.encode_batch_size)
    encode_runtime = time.perf_counter() - encode_start
    print(f"Encoding done in {encode_runtime:.1f}s", flush=True)

    disease_embeddings = torch.tensor(disease_embeddings_np, dtype=torch.float32, device=device)
    target_embeddings = torch.tensor(target_embeddings_np, dtype=torch.float32, device=device)

    train_targets = train_df_full["targetId"].unique()
    target_level = (
        train_df_full[["targetId", "label"]].groupby("targetId")["label"].max().reset_index()
        .set_index("targetId").loc[train_targets].reset_index()
    )
    y_target = target_level["label"].to_numpy()
    stratify_arg = y_target if np.unique(y_target).size > 1 else None

    run_rows = []
    for seed in args.seeds:
        print(f"\n[seed {seed}] training MLP", flush=True)
        inner_train_targets, val_targets = train_test_split(
            train_targets, test_size=args.val_size, random_state=seed,
            shuffle=True, stratify=stratify_arg,
        )
        inner_train_df = train_df_full[train_df_full["targetId"].isin(inner_train_targets)].copy()
        val_df = train_df_full[train_df_full["targetId"].isin(val_targets)].copy()

        roc_auc, pr_auc = train_mlp_once(
            inner_train_df, val_df, test_df,
            disease_embeddings, target_embeddings, disease_to_idx, target_to_idx,
            args, device, seed,
        )
        print(f"[seed {seed}] ROC-AUC={roc_auc:.5f}  PR-AUC={pr_auc:.5f}", flush=True)
        run_rows.append({"Model": "Frozen BioClinical-ModernBERT + MLP", "seed": seed, "ROC-AUC": roc_auc, "PR-AUC": pr_auc})
        pd.DataFrame(run_rows).to_csv(args.output_dir / "temporal_run_metrics.csv", index=False)

    total_runtime = time.perf_counter() - overall_start
    run_df = pd.DataFrame(run_rows)
    summary = pd.DataFrame([{
        "Model": "Frozen BioClinical-ModernBERT + MLP",
        "ROC-AUC": run_df["ROC-AUC"].mean(),
        "ROC-AUC SD": run_df["ROC-AUC"].std(ddof=0),
        "PR-AUC": run_df["PR-AUC"].mean(),
        "PR-AUC SD": run_df["PR-AUC"].std(ddof=0),
        "n": len(run_df),
    }])
    run_df.to_csv(args.output_dir / "temporal_run_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "temporal_summary.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "model_name": args.model_name, "seeds": args.seeds,
        "roc_auc_mean": float(run_df["ROC-AUC"].mean()), "roc_auc_std": float(run_df["ROC-AUC"].std(ddof=0)),
        "pr_auc_mean": float(run_df["PR-AUC"].mean()), "pr_auc_std": float(run_df["PR-AUC"].std(ddof=0)),
        "encode_runtime_seconds": encode_runtime, "total_runtime_seconds": total_runtime,
    }, indent=2))
    print({
        "roc_auc_mean": round(float(run_df["ROC-AUC"].mean()), 5),
        "roc_auc_std": round(float(run_df["ROC-AUC"].std(ddof=0)), 5),
        "pr_auc_mean": round(float(run_df["PR-AUC"].mean()), 5),
        "pr_auc_std": round(float(run_df["PR-AUC"].std(ddof=0)), 5),
        "total_runtime_minutes": round(total_runtime / 60.0, 2),
    })


if __name__ == "__main__":
    main()

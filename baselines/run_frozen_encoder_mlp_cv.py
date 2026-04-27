import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
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
DEFAULT_DATA = Path("/mnt/d/Research/OpenTargetsTransfer/data/proc/df_learn.parquet")
DEFAULT_OUT = ROOT / "Outputs" / "CV_frozen_encoder_mlp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated target-disjoint CV for a frozen pretrained encoder + MLP."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--model-name",
        type=str,
        default="thomas-sounack/BioClinical-ModernBERT-base",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=4)
    parser.add_argument("--val-size-within-train-targets", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


def build_repeated_splits(
    df: pd.DataFrame, n_splits: int, n_repeats: int, random_state: int
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    target_level = (
        df[["targetId", "label"]]
        .groupby("targetId", as_index=False)["label"]
        .max()
        .sort_values("targetId")
        .reset_index(drop=True)
    )
    tids = target_level["targetId"].to_numpy()
    y_target = target_level["label"].astype(int).to_numpy()

    splits: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state + repeat,
        )
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(X=np.zeros_like(y_target), y=y_target, groups=tids)
        ):
            splits.append((repeat, fold, tids[train_idx], tids[test_idx]))
    return splits


def split_train_val_targets(
    train_targets: np.ndarray, df: pd.DataFrame, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    target_level = (
        df[df["targetId"].isin(train_targets)][["targetId", "label"]]
        .groupby("targetId", as_index=False)["label"]
        .max()
        .sort_values("targetId")
        .reset_index(drop=True)
    )
    y_target = target_level["label"].astype(int).to_numpy()
    stratify_arg = y_target if np.unique(y_target).size > 1 else None
    inner_train, val = train_test_split(
        target_level["targetId"].to_numpy(),
        test_size=val_frac,
        random_state=seed,
        shuffle=True,
        stratify=stratify_arg,
    )
    return inner_train, val


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    overall_start = time.perf_counter()
    df = pd.read_parquet(
        args.data_path,
        columns=["diseaseId", "targetId", "label", "disease_text", "target_text"],
    )

    model = build_sentence_transformer(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
    )
    encode_start = time.perf_counter()
    disease_to_idx, disease_embeddings_np, disease_meta = encode_unique(
        model=model,
        ids=df["diseaseId"],
        texts=df["disease_text"],
        batch_size=args.encode_batch_size,
    )
    target_to_idx, target_embeddings_np, target_meta = encode_unique(
        model=model,
        ids=df["targetId"],
        texts=df["target_text"],
        batch_size=args.encode_batch_size,
    )
    encode_runtime_seconds = time.perf_counter() - encode_start

    disease_embeddings = torch.tensor(
        disease_embeddings_np, dtype=torch.float32, device=device
    )
    target_embeddings = torch.tensor(
        target_embeddings_np, dtype=torch.float32, device=device
    )

    fold_rows: list[dict[str, float | int]] = []
    oof_parts: list[pd.DataFrame] = []

    for repeat, fold, train_targets, test_targets in build_repeated_splits(
        df=df,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.seed,
    ):
        fold_seed = args.seed + 1000 * repeat + fold
        set_seed(fold_seed)
        inner_train_targets, val_targets = split_train_val_targets(
            train_targets=train_targets,
            df=df,
            val_frac=args.val_size_within_train_targets,
            seed=fold_seed,
        )

        train_df = df[df["targetId"].isin(inner_train_targets)].copy()
        val_df = df[df["targetId"].isin(val_targets)].copy()
        test_df = df[df["targetId"].isin(test_targets)].copy()

        train_dataset = PairIndexDataset(train_df, disease_to_idx, target_to_idx)
        val_dataset = PairIndexDataset(val_df, disease_to_idx, target_to_idx)
        test_dataset = PairIndexDataset(test_df, disease_to_idx, target_to_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        classifier = PairMLP(
            emb_dim=disease_embeddings.shape[1],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        pos = float(train_df["label"].sum())
        neg = float(len(train_df) - pos)
        pos_weight = torch.tensor(
            [neg / max(pos, 1.0)], dtype=torch.float32, device=device
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=1,
        )

        best_state = None
        best_val_pr_auc = -float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        fold_start = time.perf_counter()

        for epoch in range(1, args.max_epochs + 1):
            classifier.train()
            for disease_idx, target_idx, labels in train_loader:
                disease_idx = disease_idx.to(device, non_blocking=True)
                target_idx = target_idx.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = classifier(
                    disease_embeddings[disease_idx],
                    target_embeddings[target_idx],
                )
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

            val_metrics = evaluate_model(
                model=classifier,
                loader=val_loader,
                disease_embeddings=disease_embeddings,
                target_embeddings=target_embeddings,
                device=device,
                loss_fn=loss_fn,
            )
            scheduler.step(val_metrics.pr_auc)

            if val_metrics.pr_auc > best_val_pr_auc:
                best_val_pr_auc = val_metrics.pr_auc
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu()
                    for key, value in classifier.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    break

        fold_runtime_seconds = time.perf_counter() - fold_start
        if best_state is not None:
            classifier.load_state_dict(best_state)

        test_metrics = evaluate_model(
            model=classifier,
            loader=test_loader,
            disease_embeddings=disease_embeddings,
            target_embeddings=target_embeddings,
            device=device,
            loss_fn=loss_fn,
        )

        classifier.eval()
        logits_parts: list[np.ndarray] = []
        with torch.no_grad():
            for disease_idx, target_idx, _ in test_loader:
                disease_idx = disease_idx.to(device, non_blocking=True)
                target_idx = target_idx.to(device, non_blocking=True)
                logits = classifier(
                    disease_embeddings[disease_idx],
                    target_embeddings[target_idx],
                )
                logits_parts.append(logits.cpu().numpy())
        test_scores = np.concatenate(logits_parts)
        fold_oof = test_df[["diseaseId", "targetId", "label"]].copy()
        fold_oof["pred"] = test_scores
        fold_oof["repeat"] = repeat
        fold_oof["fold"] = fold
        oof_parts.append(fold_oof)

        fold_row = {
            "repeat": repeat,
            "fold": fold,
            "roc_auc": test_metrics.roc_auc,
            "pr_auc": test_metrics.pr_auc,
            "loss": test_metrics.loss,
            "best_epoch": best_epoch,
            "best_val_pr_auc": best_val_pr_auc,
            "runtime_seconds": fold_runtime_seconds,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        }
        fold_rows.append(fold_row)
        print(fold_row)

    fold_metrics = pd.DataFrame(fold_rows)
    oof_df = pd.concat(oof_parts, ignore_index=True)
    summary = pd.DataFrame(
        [
            {
                "metric": "auc",
                "mean": fold_metrics["roc_auc"].mean(),
                "std": fold_metrics["roc_auc"].std(ddof=0),
            },
            {
                "metric": "pr_auc",
                "mean": fold_metrics["pr_auc"].mean(),
                "std": fold_metrics["pr_auc"].std(ddof=0),
            },
            {
                "metric": "loss",
                "mean": fold_metrics["loss"].mean(),
                "std": fold_metrics["loss"].std(ddof=0),
            },
            {
                "metric": "best_epoch",
                "mean": fold_metrics["best_epoch"].mean(),
                "std": fold_metrics["best_epoch"].std(ddof=0),
            },
            {
                "metric": "runtime_seconds",
                "mean": fold_metrics["runtime_seconds"].mean(),
                "std": fold_metrics["runtime_seconds"].std(ddof=0),
            },
        ]
    )

    total_runtime_seconds = time.perf_counter() - overall_start
    metrics_json = {
        "model_name": args.model_name,
        "data_path": str(args.data_path),
        "device": str(device),
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "max_seq_length": args.max_seq_length,
        "encode_batch_size": args.encode_batch_size,
        "train_batch_size": args.train_batch_size,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "encode_runtime_seconds": encode_runtime_seconds,
        "total_runtime_seconds": total_runtime_seconds,
        "auc_mean": float(fold_metrics["roc_auc"].mean()),
        "auc_std": float(fold_metrics["roc_auc"].std(ddof=0)),
        "pr_auc_mean": float(fold_metrics["pr_auc"].mean()),
        "pr_auc_std": float(fold_metrics["pr_auc"].std(ddof=0)),
        "loss_mean": float(fold_metrics["loss"].mean()),
        "loss_std": float(fold_metrics["loss"].std(ddof=0)),
    }

    fold_metrics.to_csv(args.output_dir / "frozen_encoder_fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "frozen_encoder_summary.csv", index=False)
    oof_df.to_parquet(args.output_dir / "frozen_encoder_oof.parquet", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2))
    np.savez_compressed(
        args.output_dir / "disease_embeddings.npz",
        ids=disease_meta["id"].to_numpy(),
        embeddings=disease_embeddings_np,
    )
    np.savez_compressed(
        args.output_dir / "target_embeddings.npz",
        ids=target_meta["id"].to_numpy(),
        embeddings=target_embeddings_np,
    )

    print(
        {
            "auc_mean": round(metrics_json["auc_mean"], 5),
            "auc_std": round(metrics_json["auc_std"], 5),
            "pr_auc_mean": round(metrics_json["pr_auc_mean"], 5),
            "pr_auc_std": round(metrics_json["pr_auc_std"], 5),
            "encode_runtime_minutes": round(encode_runtime_seconds / 60.0, 2),
            "total_runtime_minutes": round(total_runtime_seconds / 60.0, 2),
            "output_dir": str(args.output_dir),
        }
    )


if __name__ == "__main__":
    main()

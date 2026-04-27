import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, models
from torch import nn
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a frozen pretrained encoder + MLP target-disjoint experiment."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/mnt/d/Research/OpenTargetsTransfer/data/proc/df_learn.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Outputs/frozen_encoder_mlp_bioclinical_modernbert_base"),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="thomas-sounack/BioClinical-ModernBERT-base",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_disjoint_split(
    df: pd.DataFrame,
    test_size: float,
    val_size_within_train_targets: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_level = (
        df[["targetId", "label"]].groupby("targetId")["label"].max().reset_index()
    )
    target_ids = target_level["targetId"].to_numpy()
    strat = target_level["label"].to_numpy()
    stratify_arg = strat if np.unique(strat).size > 1 else None

    train_targets, test_targets = train_test_split(
        target_ids,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify_arg,
    )

    train_target_level = target_level[target_level["targetId"].isin(train_targets)]
    inner_strat = train_target_level["label"].to_numpy()
    inner_stratify_arg = inner_strat if np.unique(inner_strat).size > 1 else None
    train_targets, val_targets = train_test_split(
        train_targets,
        test_size=val_size_within_train_targets,
        random_state=seed,
        shuffle=True,
        stratify=inner_stratify_arg,
    )

    train_df = df[df["targetId"].isin(train_targets)].copy()
    val_df = df[df["targetId"].isin(val_targets)].copy()
    test_df = df[df["targetId"].isin(test_targets)].copy()
    return train_df, val_df, test_df


def build_sentence_transformer(model_name: str, max_seq_length: int) -> SentenceTransformer:
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": "sdpa",
    }
    transformer = models.Transformer(
        model_name,
        max_seq_length=max_seq_length,
        model_kwargs=model_kwargs,
    )
    pooling = models.Pooling(
        transformer.get_embedding_dimension(),
        pooling_mode="mean",
    )
    normalize = models.Normalize()
    return SentenceTransformer(modules=[transformer, pooling, normalize])


def encode_unique(
    model: SentenceTransformer,
    ids: pd.Series,
    texts: pd.Series,
    batch_size: int,
) -> tuple[dict[str, int], np.ndarray, pd.DataFrame]:
    unique_df = (
        pd.DataFrame({"id": ids, "text": texts.astype(str)})
        .drop_duplicates(subset=["id"])
        .reset_index(drop=True)
    )
    embeddings = model.encode(
        unique_df["text"].tolist(),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    id_to_idx = {value: idx for idx, value in enumerate(unique_df["id"].tolist())}
    return id_to_idx, embeddings, unique_df


class PairIndexDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        disease_to_idx: dict[str, int],
        target_to_idx: dict[str, int],
    ) -> None:
        self.disease_idx = torch.tensor(
            [disease_to_idx[x] for x in df["diseaseId"].tolist()], dtype=torch.long
        )
        self.target_idx = torch.tensor(
            [target_to_idx[x] for x in df["targetId"].tolist()], dtype=torch.long
        )
        self.labels = torch.tensor(df["label"].to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.disease_idx[idx], self.target_idx[idx], self.labels[idx]


class PairMLP(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        input_dim = emb_dim * 4 + 1
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, disease_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        prod = disease_emb * target_emb
        cosine = torch.sum(prod, dim=1, keepdim=True)
        features = torch.cat(
            [
                disease_emb,
                target_emb,
                torch.abs(disease_emb - target_emb),
                prod,
                cosine,
            ],
            dim=1,
        )
        return self.net(features).squeeze(1)


@dataclass
class EpochMetrics:
    loss: float
    roc_auc: float
    pr_auc: float


def evaluate_model(
    model: PairMLP,
    loader: DataLoader,
    disease_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    device: torch.device,
    loss_fn: nn.Module | None = None,
) -> EpochMetrics:
    model.eval()
    labels_list: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for disease_idx, target_idx, labels in loader:
            disease_idx = disease_idx.to(device, non_blocking=True)
            target_idx = target_idx.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(disease_embeddings[disease_idx], target_embeddings[target_idx])
            if loss_fn is not None:
                running_loss += float(loss_fn(logits, labels).item()) * labels.shape[0]
                count += labels.shape[0]

            labels_list.append(labels.cpu().numpy())
            logits_list.append(logits.cpu().numpy())

    y_true = np.concatenate(labels_list)
    y_score = np.concatenate(logits_list)
    avg_loss = running_loss / max(1, count) if loss_fn is not None else float("nan")
    return EpochMetrics(
        loss=avg_loss,
        roc_auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=float(average_precision_score(y_true, y_score)),
    )


def evaluate_raw_cosine(
    df: pd.DataFrame,
    disease_embeddings_np: np.ndarray,
    target_embeddings_np: np.ndarray,
    disease_to_idx: dict[str, int],
    target_to_idx: dict[str, int],
) -> tuple[dict[str, float], np.ndarray]:
    disease_idx = np.array([disease_to_idx[x] for x in df["diseaseId"].tolist()])
    target_idx = np.array([target_to_idx[x] for x in df["targetId"].tolist()])
    scores = np.sum(
        disease_embeddings_np[disease_idx] * target_embeddings_np[target_idx], axis=1
    )
    labels = df["label"].to_numpy()
    metrics = {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }
    return metrics, scores


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
    train_df, val_df, test_df = target_disjoint_split(
        df=df,
        test_size=args.test_size,
        val_size_within_train_targets=args.val_size_within_train_targets,
        seed=args.seed,
    )
    split_summary = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_targets": int(train_df["targetId"].nunique()),
        "val_targets": int(val_df["targetId"].nunique()),
        "test_targets": int(test_df["targetId"].nunique()),
        "train_positive_rate": float(train_df["label"].mean()),
        "val_positive_rate": float(val_df["label"].mean()),
        "test_positive_rate": float(test_df["label"].mean()),
    }
    print(split_summary)

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

    raw_cosine_metrics, raw_cosine_scores = evaluate_raw_cosine(
        df=test_df,
        disease_embeddings_np=disease_embeddings_np,
        target_embeddings_np=target_embeddings_np,
        disease_to_idx=disease_to_idx,
        target_to_idx=target_to_idx,
    )
    print({"raw_cosine_test": raw_cosine_metrics})

    disease_embeddings = torch.tensor(disease_embeddings_np, dtype=torch.float32, device=device)
    target_embeddings = torch.tensor(target_embeddings_np, dtype=torch.float32, device=device)

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
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
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
    train_start = time.perf_counter()

    history: list[dict[str, float]] = []
    for epoch in range(1, args.max_epochs + 1):
        classifier.train()
        running_loss = 0.0
        seen = 0
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

            running_loss += float(loss.item()) * labels.shape[0]
            seen += labels.shape[0]

        train_loss = running_loss / max(1, seen)
        val_metrics = evaluate_model(
            model=classifier,
            loader=val_loader,
            disease_embeddings=disease_embeddings,
            target_embeddings=target_embeddings,
            device=device,
            loss_fn=loss_fn,
        )
        scheduler.step(val_metrics.pr_auc)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics.loss,
                "val_roc_auc": val_metrics.roc_auc,
                "val_pr_auc": val_metrics.pr_auc,
            }
        )
        print(history[-1])

        if val_metrics.pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_metrics.pr_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in classifier.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    train_runtime_seconds = time.perf_counter() - train_start
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
    total_runtime_seconds = time.perf_counter() - overall_start

    summary = {
        "model_name": args.model_name,
        "data_path": str(args.data_path),
        "device": str(device),
        "max_seq_length": args.max_seq_length,
        "encode_batch_size": args.encode_batch_size,
        "train_batch_size": args.train_batch_size,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "seed": args.seed,
        "split_summary": split_summary,
        "raw_cosine_test_metrics": raw_cosine_metrics,
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_val_pr_auc,
        "test_metrics": {
            "loss": test_metrics.loss,
            "roc_auc": test_metrics.roc_auc,
            "pr_auc": test_metrics.pr_auc,
        },
        "encode_runtime_seconds": encode_runtime_seconds,
        "train_runtime_seconds": train_runtime_seconds,
        "total_runtime_seconds": total_runtime_seconds,
        "history": history,
    }

    metrics_path = args.output_dir / "metrics.json"
    summary_csv_path = args.output_dir / "summary.csv"
    predictions_path = args.output_dir / "test_predictions.parquet"
    disease_emb_path = args.output_dir / "disease_embeddings.npz"
    target_emb_path = args.output_dir / "target_embeddings.npz"

    metrics_path.write_text(json.dumps(summary, indent=2))
    pd.DataFrame(
        [
            {
                "model_name": args.model_name,
                "raw_cosine_roc_auc": raw_cosine_metrics["roc_auc"],
                "raw_cosine_pr_auc": raw_cosine_metrics["pr_auc"],
                "mlp_roc_auc": test_metrics.roc_auc,
                "mlp_pr_auc": test_metrics.pr_auc,
                "best_epoch": best_epoch,
                "best_val_pr_auc": best_val_pr_auc,
                "encode_runtime_seconds": encode_runtime_seconds,
                "train_runtime_seconds": train_runtime_seconds,
                "total_runtime_seconds": total_runtime_seconds,
                **split_summary,
            }
        ]
    ).to_csv(summary_csv_path, index=False)
    pred_df = test_df[["diseaseId", "targetId", "label"]].copy()
    pred_df["raw_cosine_score"] = raw_cosine_scores
    pred_df.to_parquet(predictions_path, index=False)
    np.savez_compressed(
        disease_emb_path,
        ids=disease_meta["id"].to_numpy(),
        embeddings=disease_embeddings_np,
    )
    np.savez_compressed(
        target_emb_path,
        ids=target_meta["id"].to_numpy(),
        embeddings=target_embeddings_np,
    )

    print(
        {
            "raw_cosine_roc_auc": round(raw_cosine_metrics["roc_auc"], 5),
            "raw_cosine_pr_auc": round(raw_cosine_metrics["pr_auc"], 5),
            "mlp_roc_auc": round(test_metrics.roc_auc, 5),
            "mlp_pr_auc": round(test_metrics.pr_auc, 5),
            "encode_runtime_minutes": round(encode_runtime_seconds / 60.0, 2),
            "train_runtime_minutes": round(train_runtime_seconds / 60.0, 2),
            "total_runtime_minutes": round(total_runtime_seconds / 60.0, 2),
            "output_dir": str(args.output_dir),
        }
    )


if __name__ == "__main__":
    main()

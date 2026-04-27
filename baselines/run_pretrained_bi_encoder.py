import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
    models,
)
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import EarlyStoppingCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a target-disjoint pretrained bi-encoder experiment."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/mnt/d/Research/OpenTargetsTransfer/data/proc/df_learn.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Outputs/pretrained_bi_encoder_bioclinical_modernbert_base"),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="thomas-sounack/BioClinical-ModernBERT-base",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--val-size-within-train-targets",
        type=float,
        default=0.1,
        help="Fraction of train targets used for validation.",
    )
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--train-batch-size", type=int, default=96)
    parser.add_argument("--eval-batch-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--evals-per-epoch", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument(
        "--val-eval-max-pairs",
        type=int,
        default=20000,
        help="Cap the validation evaluator set for speed.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
    )
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize_lengths(df: pd.DataFrame) -> dict[str, float]:
    return {
        "disease_text_tokens_p50": float(
            df["disease_text"].astype(str).str.split().str.len().median()
        ),
        "target_text_tokens_p50": float(
            df["target_text"].astype(str).str.split().str.len().median()
        ),
    }


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
    strat_arg = strat if np.unique(strat).size > 1 else None

    train_targets, test_targets = train_test_split(
        target_ids,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=strat_arg,
    )

    train_target_level = target_level[target_level["targetId"].isin(train_targets)]
    inner_strat = train_target_level["label"].to_numpy()
    inner_strat_arg = inner_strat if np.unique(inner_strat).size > 1 else None
    train_targets, val_targets = train_test_split(
        train_targets,
        test_size=val_size_within_train_targets,
        random_state=seed,
        shuffle=True,
        stratify=inner_strat_arg,
    )

    train_df = df[df["targetId"].isin(train_targets)].copy()
    val_df = df[df["targetId"].isin(val_targets)].copy()
    test_df = df[df["targetId"].isin(test_targets)].copy()
    return train_df, val_df, test_df


def make_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_dict(
        {
            "sentence1": df["disease_text"].astype(str).tolist(),
            "sentence2": df["target_text"].astype(str).tolist(),
            "label": df["label"].astype(float).tolist(),
        }
    )


def build_model(model_name: str, max_seq_length: int) -> SentenceTransformer:
    model_args = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": "sdpa",
    }
    transformer = models.Transformer(
        model_name,
        max_seq_length=max_seq_length,
        model_args=model_args,
    )
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
    )
    normalize = models.Normalize()
    return SentenceTransformer(modules=[transformer, pooling, normalize])


def encode_unique_texts(
    model: SentenceTransformer,
    ids: pd.Series,
    texts: pd.Series,
    batch_size: int,
) -> dict[str, np.ndarray]:
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
    )
    return {
        row_id: embeddings[idx]
        for idx, row_id in enumerate(unique_df["id"].tolist())
    }


def evaluate_test_pairs(
    model: SentenceTransformer,
    test_df: pd.DataFrame,
    eval_batch_size: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    disease_embeddings = encode_unique_texts(
        model,
        ids=test_df["diseaseId"],
        texts=test_df["disease_text"],
        batch_size=eval_batch_size,
    )
    target_embeddings = encode_unique_texts(
        model,
        ids=test_df["targetId"],
        texts=test_df["target_text"],
        batch_size=eval_batch_size,
    )

    disease_matrix = np.vstack(
        [disease_embeddings[d_id] for d_id in test_df["diseaseId"].tolist()]
    )
    target_matrix = np.vstack(
        [target_embeddings[t_id] for t_id in test_df["targetId"].tolist()]
    )
    scores = np.sum(disease_matrix * target_matrix, axis=1)
    labels = test_df["label"].to_numpy()
    metrics = {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }
    pred_df = test_df[["diseaseId", "targetId", "label"]].copy()
    pred_df["score"] = scores
    return metrics, pred_df


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    args = parse_args()
    set_reproducibility(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hf_output_dir = args.output_dir / "hf_trainer"
    hf_output_dir.mkdir(parents=True, exist_ok=True)

    overall_start = time.perf_counter()
    print(f"Loading data from {args.data_path}")
    df = pd.read_parquet(
        args.data_path,
        columns=["diseaseId", "targetId", "label", "disease_text", "target_text"],
    )
    length_summary = summarize_lengths(df)
    print(
        {
            "rows": len(df),
            "targets": int(df["targetId"].nunique()),
            "diseases": int(df["diseaseId"].nunique()),
            "positive_rate": float(df["label"].mean()),
            **length_summary,
        }
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

    train_dataset = make_dataset(train_df)
    val_dataset = make_dataset(val_df)
    val_eval_n = min(args.val_eval_max_pairs, len(val_dataset))
    val_eval_dataset = val_dataset.shuffle(seed=args.seed).select(range(val_eval_n))

    model = build_model(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
    )

    train_loss = losses.OnlineContrastiveLoss(model=model, margin=args.margin)
    evaluator = BinaryClassificationEvaluator(
        sentences1=val_eval_dataset["sentence1"],
        sentences2=val_eval_dataset["sentence2"],
        labels=[int(x) for x in val_eval_dataset["label"]],
        name="val",
        show_progress_bar=False,
        batch_size=max(32, args.eval_batch_size // 2),
        write_csv=False,
    )

    steps_per_epoch = math.ceil(len(train_df) / args.train_batch_size)
    eval_steps = max(100, steps_per_epoch // max(1, args.evals_per_epoch))

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(hf_output_dir),
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        logging_steps=max(20, eval_steps // 5),
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_val_cosine_ap",
        greater_is_better=True,
        report_to="none",
        bf16=torch.cuda.is_available(),
        fp16=False,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_eval_dataset,
        loss=train_loss,
        evaluator=evaluator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    train_start = time.perf_counter()
    train_output = trainer.train()
    train_wall_seconds = time.perf_counter() - train_start
    print(train_output)

    eval_start = time.perf_counter()
    test_metrics, pred_df = evaluate_test_pairs(
        model=trainer.model,
        test_df=test_df,
        eval_batch_size=args.eval_batch_size,
    )
    eval_wall_seconds = time.perf_counter() - eval_start

    total_wall_seconds = time.perf_counter() - overall_start
    summary = {
        "model_name": args.model_name,
        "data_path": str(args.data_path),
        "max_seq_length": args.max_seq_length,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_epochs": args.max_epochs,
        "eval_steps": eval_steps,
        "patience": args.patience,
        "margin": args.margin,
        "seed": args.seed,
        "split_summary": split_summary,
        "length_summary": length_summary,
        "train_runtime_seconds": train_wall_seconds,
        "test_eval_runtime_seconds": eval_wall_seconds,
        "total_runtime_seconds": total_wall_seconds,
        "test_metrics": test_metrics,
        "trainer_metrics": getattr(train_output, "metrics", {}),
    }

    pred_path = args.output_dir / "test_predictions.parquet"
    metrics_path = args.output_dir / "metrics.json"
    summary_path = args.output_dir / "summary.csv"
    pred_df.to_parquet(pred_path, index=False)
    metrics_path.write_text(json.dumps(summary, indent=2))
    pd.DataFrame(
        [
            {
                "model_name": args.model_name,
                "roc_auc": test_metrics["roc_auc"],
                "pr_auc": test_metrics["pr_auc"],
                "train_runtime_seconds": train_wall_seconds,
                "test_eval_runtime_seconds": eval_wall_seconds,
                "total_runtime_seconds": total_wall_seconds,
                **split_summary,
            }
        ]
    ).to_csv(summary_path, index=False)

    print("\nFinal test metrics")
    print(
        {
            "roc_auc": round(test_metrics["roc_auc"], 5),
            "pr_auc": round(test_metrics["pr_auc"], 5),
            "train_runtime_minutes": round(train_wall_seconds / 60.0, 2),
            "test_eval_runtime_minutes": round(eval_wall_seconds / 60.0, 2),
            "total_runtime_minutes": round(total_wall_seconds / 60.0, 2),
            "output_dir": str(args.output_dir),
        }
    )


if __name__ == "__main__":
    main()

"""Run simple target-disjoint CV baselines for OTRec.

This script reconstructs the repeated target-disjoint cross-validation
used by the neural retriever:

- groups are targetId
- stratification is by per-target positivity (max label)
- splits use StratifiedGroupKFold(n_splits=5, shuffle=True,
  random_state=42 + repeat)

It evaluates deliberately simple cold-start baselines:

- Disease mean positive-rate prior
- Target mean positive-rate prior
- Raw Open Targets score
- Matrix factorization over positive interaction matrix
- Node2Vec over positive disease-target graph (GPU via PyG)
- TF-IDF cosine between disease and target text descriptions
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

try:
    import torch
    from torch_geometric.nn import Node2Vec as PyGNode2Vec

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "Outputs" / "CV_DL" / "oof_dl_preds.parquet"
DEFAULT_OUT_DIR = ROOT / "Outputs" / "CV_baselines"


@dataclass
class BaselineConfig:
    models: tuple[str, ...]
    n_splits: int = 5
    n_repeats: int = 5
    random_state: int = 42
    als_factors: int = 64
    node2vec_dimensions: int = 64
    node2vec_walk_length: int = 20
    node2vec_num_walks: int = 10
    node2vec_window: int = 10
    node2vec_epochs: int = 3
    node2vec_workers: int = 0
    node2vec_device: str = "cuda"
    node2vec_batch_size: int = 1024
    node2vec_lr: float = 0.01
    tfidf_max_features: int = 50000
    tfidf_min_df: int = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mf", "node2vec"],
        choices=[
            "disease_mean",
            "target_mean",
            "ot_score",
            "mf",
            "node2vec",
            "tfidf",
        ],
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--node2vec-dimensions", type=int, default=64)
    parser.add_argument("--node2vec-walk-length", type=int, default=20)
    parser.add_argument("--node2vec-num-walks", type=int, default=10)
    parser.add_argument("--node2vec-window", type=int, default=10)
    parser.add_argument("--node2vec-epochs", type=int, default=3)
    parser.add_argument("--node2vec-workers", type=int, default=0)
    parser.add_argument("--node2vec-device", type=str, default="cuda")
    parser.add_argument("--node2vec-batch-size", type=int, default=1024)
    parser.add_argument("--node2vec-lr", type=float, default=0.01)

    return parser.parse_args()


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def load_learning_df(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {
        "diseaseId",
        "targetId",
        "score",
        "label",
        "disease_text",
        "target_text",
        "pred",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df.copy()


def build_repeated_splits(
    df_learn: pd.DataFrame, n_splits: int, n_repeats: int, random_state: int
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    target_level = (
        df_learn[["targetId", "label"]]
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


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def fit_group_mean_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str,
) -> np.ndarray:
    group_means = train_df.groupby(group_col)["label"].mean()
    prior = float(train_df["label"].mean())
    return test_df[group_col].map(group_means).fillna(prior).to_numpy(dtype=np.float32)


def fit_ot_score_predict(test_df: pd.DataFrame) -> np.ndarray:
    return test_df["score"].to_numpy(dtype=np.float32)


def fit_mf_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: BaselineConfig,
) -> np.ndarray:
    positive_train = train_df.loc[
        train_df["label"] == 1, ["diseaseId", "targetId"]
    ].drop_duplicates()
    if positive_train.empty:
        prior = float(train_df["label"].mean())
        return np.full(len(test_df), prior, dtype=np.float32)

    disease_ids = np.sort(positive_train["diseaseId"].unique())
    train_target_ids = np.sort(positive_train["targetId"].unique())
    disease_index = {disease_id: idx for idx, disease_id in enumerate(disease_ids)}
    target_index = {target_id: idx for idx, target_id in enumerate(train_target_ids)}

    row_idx = positive_train["targetId"].map(target_index).to_numpy()
    col_idx = positive_train["diseaseId"].map(disease_index).to_numpy()
    values = np.ones(len(positive_train), dtype=np.float32)
    item_user = sparse.csr_matrix(
        (values, (row_idx, col_idx)),
        shape=(len(train_target_ids), len(disease_ids)),
        dtype=np.float32,
    )

    max_components = min(item_user.shape) - 1
    n_components = max(2, min(config.als_factors, max_components))
    svd = TruncatedSVD(n_components=n_components, random_state=config.random_state)
    target_factors = svd.fit_transform(item_user)
    disease_factors = svd.components_.T
    mean_target_factor = target_factors.mean(axis=0)
    global_prior = float(train_df["label"].mean())

    scores = np.empty(len(test_df), dtype=np.float32)
    disease_values = test_df["diseaseId"].to_numpy()
    for idx, disease_id in enumerate(disease_values):
        user_idx = disease_index.get(disease_id)
        if user_idx is None:
            scores[idx] = global_prior
            continue
        raw_score = float(np.dot(disease_factors[user_idx], mean_target_factor))
        scores[idx] = float(sigmoid(np.array([raw_score], dtype=np.float32))[0])
    return scores


def fit_node2vec_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: BaselineConfig,
) -> np.ndarray:
    try:
        import networkx as nx
        from node2vec import Node2Vec as GensimNode2Vec
    except ImportError as exc:
        raise ImportError(
            "node2vec and networkx are required: pip install node2vec networkx"
        ) from exc

    positive_train = train_df.loc[
        train_df["label"] == 1, ["diseaseId", "targetId"]
    ].drop_duplicates()
    if positive_train.empty:
        prior = float(train_df["label"].mean())
        return np.full(len(test_df), prior, dtype=np.float32)

    np.random.seed(config.random_state)

    G = nx.Graph()
    disease_vals = positive_train["diseaseId"].to_numpy()
    target_vals = positive_train["targetId"].to_numpy()
    for d, t in zip(disease_vals, target_vals):
        G.add_edge(f"d::{d}", f"t::{t}")

    n2v = GensimNode2Vec(
        G,
        dimensions=config.node2vec_dimensions,
        walk_length=config.node2vec_walk_length,
        num_walks=config.node2vec_num_walks,
        p=1.0,
        q=1.0,
        workers=max(1, config.node2vec_workers) if config.node2vec_workers > 0 else 4,
        seed=config.random_state,
        quiet=True,
    )
    w2v = n2v.fit(
        window=config.node2vec_window,
        min_count=1,
        sg=1,
        epochs=config.node2vec_epochs,
    )

    all_target_nodes = [f"t::{t}" for t in sorted(positive_train["targetId"].unique())]
    target_vectors = np.array(
        [w2v.wv[node] for node in all_target_nodes if node in w2v.wv]
    )
    if len(target_vectors) == 0:
        prior = float(train_df["label"].mean())
        return np.full(len(test_df), prior, dtype=np.float32)

    mean_target_embedding = target_vectors.mean(axis=0)
    mean_target_norm = float(np.linalg.norm(mean_target_embedding)) or 1.0
    global_prior = float(train_df["label"].mean())

    scores = np.empty(len(test_df), dtype=np.float32)
    for idx, disease_id in enumerate(test_df["diseaseId"].to_numpy()):
        node_key = f"d::{disease_id}"
        if node_key not in w2v.wv:
            scores[idx] = global_prior
            continue
        dvec = w2v.wv[node_key]
        denom = float(np.linalg.norm(dvec)) * mean_target_norm
        cosine = (
            float(np.dot(dvec, mean_target_embedding) / denom) if denom > 0 else 0.0
        )
        scores[idx] = 0.5 * (cosine + 1.0)
    return scores


def fit_tfidf_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: BaselineConfig,
) -> np.ndarray:
    train_disease_text = (
        train_df[["diseaseId", "disease_text"]]
        .drop_duplicates(subset=["diseaseId"])
        .fillna("")
    )
    train_target_text = (
        train_df[["targetId", "target_text"]]
        .drop_duplicates(subset=["targetId"])
        .fillna("")
    )
    corpus = pd.concat(
        [train_disease_text["disease_text"], train_target_text["target_text"]],
        ignore_index=True,
    )
    corpus = corpus.astype(str).str.strip()
    if corpus.empty or corpus.eq("").all():
        prior = float(train_df["label"].mean())
        return np.full(len(test_df), prior, dtype=np.float32)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=config.tfidf_min_df,
        max_features=config.tfidf_max_features,
    )
    vectorizer.fit(corpus.tolist())
    test_disease_text = (
        test_df[["diseaseId", "disease_text"]]
        .drop_duplicates(subset=["diseaseId"])
        .fillna("")
        .reset_index(drop=True)
    )
    test_target_text = (
        test_df[["targetId", "target_text"]]
        .drop_duplicates(subset=["targetId"])
        .fillna("")
        .reset_index(drop=True)
    )
    disease_lookup = {
        disease_id: idx
        for idx, disease_id in enumerate(test_disease_text["diseaseId"].tolist())
    }
    target_lookup = {
        target_id: idx
        for idx, target_id in enumerate(test_target_text["targetId"].tolist())
    }
    disease_matrix = vectorizer.transform(
        test_disease_text["disease_text"].astype(str).tolist()
    )
    target_matrix = vectorizer.transform(
        test_target_text["target_text"].astype(str).tolist()
    )
    disease_rows = test_df["diseaseId"].map(disease_lookup).to_numpy()
    target_rows = test_df["targetId"].map(target_lookup).to_numpy()
    scores = np.asarray(
        disease_matrix[disease_rows].multiply(target_matrix[target_rows]).sum(axis=1)
    ).ravel()
    return scores.astype(np.float32)


def run_model(
    model_name: str,
    df_learn: pd.DataFrame,
    splits: list[tuple[int, int, np.ndarray, np.ndarray]],
    config: BaselineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, float | int | str]] = []
    oof_parts: list[pd.DataFrame] = []

    for repeat, fold, train_tids, test_tids in splits:
        train_mask = df_learn["targetId"].isin(train_tids)
        test_mask = df_learn["targetId"].isin(test_tids)
        train_df = df_learn.loc[
            train_mask,
            ["diseaseId", "targetId", "label", "disease_text", "target_text"],
        ].copy()
        test_df = df_learn.loc[test_mask].copy()

        if model_name == "disease_mean":
            preds = fit_group_mean_predict(train_df, test_df, group_col="diseaseId")
        elif model_name == "target_mean":
            preds = fit_group_mean_predict(train_df, test_df, group_col="targetId")
        elif model_name == "ot_score":
            preds = fit_ot_score_predict(test_df)
        elif model_name == "mf":
            preds = fit_mf_predict(train_df, test_df, config)
        elif model_name == "node2vec":
            preds = fit_node2vec_predict(train_df, test_df, config)
        elif model_name == "tfidf":
            preds = fit_tfidf_predict(train_df, test_df, config)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        metrics = evaluate_predictions(test_df["label"].to_numpy(), preds)
        metric_rows.append(
            {
                "model": model_name,
                "repeat": repeat,
                "fold": fold,
                **metrics,
            }
        )

        fold_oof = test_df[
            ["diseaseId", "targetId", "score", "label", "disease_text", "target_text"]
        ].copy()
        fold_oof["pred"] = preds
        fold_oof["repeat"] = repeat
        fold_oof["fold"] = fold
        oof_parts.append(fold_oof)

    metrics_df = pd.DataFrame(metric_rows)
    detailed_oof = pd.concat(oof_parts, ignore_index=True)
    aggregated_oof = (
        detailed_oof.groupby(
            ["diseaseId", "targetId", "score", "label", "disease_text", "target_text"],
            as_index=False,
        )
        .agg(pred=("pred", "mean"), oof_count=("pred", "size"))
        .sort_values(["diseaseId", "targetId"])
        .reset_index(drop=True)
    )
    return metrics_df, aggregated_oof


def summarise_metrics(all_metrics: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, float | str]] = []
    for model_name, group in all_metrics.groupby("model"):
        summary_rows.append(
            {
                "model": model_name,
                "stat": "mean",
                "roc_auc": float(group["roc_auc"].mean()),
                "pr_auc": float(group["pr_auc"].mean()),
            }
        )
        summary_rows.append(
            {
                "model": model_name,
                "stat": "std",
                "roc_auc": float(group["roc_auc"].std(ddof=1)),
                "pr_auc": float(group["pr_auc"].std(ddof=1)),
            }
        )
    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = BaselineConfig(
        models=tuple(args.models),
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        node2vec_dimensions=args.node2vec_dimensions,
        node2vec_walk_length=args.node2vec_walk_length,
        node2vec_num_walks=args.node2vec_num_walks,
        node2vec_window=args.node2vec_window,
        node2vec_epochs=args.node2vec_epochs,
        node2vec_workers=args.node2vec_workers,
        node2vec_device=args.node2vec_device,
        node2vec_batch_size=args.node2vec_batch_size,
        node2vec_lr=args.node2vec_lr,
    )
    df_learn = load_learning_df(args.input)
    splits = build_repeated_splits(
        df_learn, config.n_splits, config.n_repeats, config.random_state
    )

    all_metric_frames: list[pd.DataFrame] = []
    for model_name in config.models:
        metrics_df, aggregated_oof = run_model(model_name, df_learn, splits, config)
        metrics_df.to_csv(out_dir / f"{model_name}_fold_metrics.csv", index=False)
        aggregated_oof.to_parquet(out_dir / f"{model_name}_cv.parquet", index=False)
        all_metric_frames.append(metrics_df)

    all_metrics = pd.concat(all_metric_frames, ignore_index=True)
    all_metrics.to_csv(out_dir / "all_baseline_fold_metrics.csv", index=False)
    summary = summarise_metrics(all_metrics)
    summary.to_csv(out_dir / "baselines_summary.csv", index=False)

    for row in summary.itertuples(index=False):
        print(
            f"{row.model:10s} {row.stat:4s} "
            f"ROC-AUC={row.roc_auc:.4f} PR-AUC={row.pr_auc:.4f}"
        )


if __name__ == "__main__":
    main()

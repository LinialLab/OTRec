"""Run repeated temporal validation for OTRec and temporal baselines.

This script reconstructs the temporal setup from `2-Temporal-Eval.ipynb` using
the processed OTP 22.02 history snapshot as training data and the OTP 25.06
anti-join cohort as test data.

Stochastic models are rerun over multiple seeds:
- OTRec (TensorFlow/Keras two-tower model)
- OTTree (CatBoost)
- Matrix Factorization (randomized TruncatedSVD)

Deterministic baselines are evaluated once and reported with zero run-to-run
variance:
- Historical OTP score
- Historical target mean
- Historical disease mean
- TF-IDF cosine

Per-seed metrics and aggregate summary statistics are written to a dedicated
output directory so existing temporal results remain untouched until validated.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

from dl_model_def import build_two_tower_model
from run_baselines import BaselineConfig, fit_mf_predict, fit_tfidf_predict

DEFAULT_HISTORY = ROOT.parent / "code" / "history_df.parquet"
DEFAULT_FUTURE = ROOT.parent / "code" / "final_df.parquet"
DEFAULT_DISEASE = ROOT.parent / "code" / "copy_proc" / "disease_df.parquet"
DEFAULT_TARGET = ROOT.parent / "code" / "copy_proc" / "target_df.parquet"
DEFAULT_OUT_DIR = ROOT / "Outputs" / "temporal_repeats_5seed"
DEFAULT_TABLE2 = ROOT / "Outputs" / "Table 2 - Temporal Prospective validation.csv"

STOCHASTIC_MODELS = ("OTRec", "OTTree (CatBoost)", "Matrix Factorization")
DETERMINISTIC_MODELS = (
    "Target Mean Baseline",
    "TF-IDF cosine",
    "Disease Mean Baseline",
    "Open Targets Score",
)
MODEL_ORDER = [
    "OTRec",
    "OTTree (CatBoost)",
    "Target Mean Baseline",
    "Matrix Factorization",
    "TF-IDF cosine",
    "Disease Mean Baseline",
    "Open Targets Score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--future", type=Path, default=DEFAULT_FUTURE)
    parser.add_argument("--disease", type=Path, default=DEFAULT_DISEASE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--table2", type=Path, default=DEFAULT_TABLE2)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Random seeds for stochastic temporal reruns.",
    )
    parser.add_argument(
        "--otrec-epochs",
        type=int,
        default=6,
        help="Maximum OTRec epochs per seed.",
    )
    parser.add_argument(
        "--update-table2",
        action="store_true",
        help="Write the aggregate mean/SD results back to the manuscript Table 2 CSV.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def make_ds(df: pd.DataFrame) -> tf.data.Dataset:
    feats = {
        "query": {
            "disease_text": df["disease_text"],
            "diseaseId": df["diseaseId"],
        },
        "candidate": {
            "target_text": df["target_text"],
            "targetId": df["targetId"],
        },
    }
    y = {
        "cls": df["label"].astype("float32"),
        "score": df["score"].astype("float32"),
    }
    return tf.data.Dataset.from_tensor_slices((feats, y))


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(y_true, y_score)), float(
        average_precision_score(y_true, y_score)
    )


def merge_df_dis_target(
    df: pd.DataFrame, disease_df: pd.DataFrame, target_df: pd.DataFrame
) -> pd.DataFrame:
    out = df.merge(
        disease_df[["diseaseId", "disease_text_embed"]], on="diseaseId", how="left"
    )
    out = out.merge(
        target_df[["targetId", "target_text_embed"]], on="targetId", how="left"
    )
    out = out.rename(
        columns={
            "disease_text_embed": "disease_text",
            "target_text_embed": "target_text",
        }
    )
    out["disease_text"] = out["disease_text"].fillna("").astype(str)
    out["target_text"] = out["target_text"].fillna("").astype(str)
    return out


def build_temporal_test_set(
    history_df: pd.DataFrame, future_df: pd.DataFrame
) -> pd.DataFrame:
    join_keys = ["diseaseId", "targetId", "label"]
    return (
        future_df.merge(history_df[join_keys], on=join_keys, how="left", indicator=True)
        .query('_merge == "left_only"')
        .drop(columns="_merge")
        .reset_index(drop=True)
    )


def add_historical_score(
    history_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    return test_df.merge(
        history_df[["diseaseId", "targetId", "score"]].rename(
            columns={"score": "score_past"}
        ),
        on=["diseaseId", "targetId"],
        how="left",
    )


def target_mean_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    target_means = train_df.groupby("targetId")["label"].mean()
    prior = float(train_df["label"].mean())
    return (
        test_df["targetId"].map(target_means).fillna(prior).to_numpy(dtype=np.float32)
    )


def disease_mean_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    disease_means = train_df.groupby("diseaseId")["label"].mean()
    prior = float(train_df["label"].mean())
    return (
        test_df["diseaseId"].map(disease_means).fillna(prior).to_numpy(dtype=np.float32)
    )


def train_otrec_once(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    epochs: int,
) -> tuple[float, float]:
    set_global_seed(seed)
    keras.backend.clear_session()

    train_tids, val_tids = train_test_split(
        history_df["targetId"].unique(),
        test_size=0.01,
        random_state=seed,
        shuffle=True,
    )
    train_df = history_df[history_df["targetId"].isin(train_tids)].copy()
    val_df = history_df[history_df["targetId"].isin(val_tids)].copy()

    model = build_two_tower_model(history_df)
    losses = {
        "cls": keras.losses.BinaryCrossentropy(from_logits=False),
        "score": keras.losses.MeanSquaredError(),
    }
    model.compile(
        optimizer=keras.optimizers.Adam(7e-3),
        loss=losses,
        loss_weights={"cls": 1.0, "score": 0.1},
        metrics={
            "cls": [
                keras.metrics.AUC(name="auc"),
                keras.metrics.AUC(curve="PR", name="pr_auc"),
            ],
            "score": [keras.metrics.RootMeanSquaredError(name="rmse")],
        },
    )

    train_ds = (
        make_ds(train_df)
        .shuffle(200_000, seed=seed, reshuffle_each_iteration=True)
        .batch(512)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = make_ds(val_df).batch(2048).prefetch(tf.data.AUTOTUNE)
    test_ds = (
        make_ds(test_df[train_df.columns.to_list()])
        .batch(1024)
        .prefetch(tf.data.AUTOTUNE)
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            "val_cls_loss", mode="min", factor=0.2, patience=1
        ),
        keras.callbacks.EarlyStopping(
            "val_cls_loss", mode="min", patience=2, restore_best_weights=False
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0,
    )
    y_pred = model.predict(test_ds, verbose=0)["cls"].ravel()
    return compute_metrics(test_df["label"].to_numpy(), y_pred)


def train_ottree_once(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> tuple[float, float]:
    from catboost import CatBoostClassifier, Pool

    features = ["disease_text", "target_text", "diseaseId"]
    text_features = ["disease_text", "target_text"]
    cat_features = ["diseaseId"]

    train_pool = Pool(
        data=history_df[features],
        label=history_df["label"],
        text_features=text_features,
        cat_features=cat_features,
    )
    test_pool = Pool(
        data=test_df[features],
        label=test_df["label"],
        text_features=text_features,
        cat_features=cat_features,
    )

    params: dict[str, object] = {
        "depth": 8,
        "eval_metric": "AUC",
        "random_seed": seed,
        "verbose": False,
    }
    if tf.config.list_physical_devices("GPU"):
        params["task_type"] = "GPU"

    model = CatBoostClassifier(**params)
    model.fit(train_pool)
    y_pred = model.predict_proba(test_pool)[:, 1]
    return compute_metrics(test_df["label"].to_numpy(), y_pred)


def run_matrix_factorization_once(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> tuple[float, float]:
    config = BaselineConfig(models=("mf",), random_state=seed)
    y_pred = fit_mf_predict(
        train_df=history_df[
            ["diseaseId", "targetId", "label", "disease_text", "target_text"]
        ],
        test_df=test_df[
            ["diseaseId", "targetId", "label", "disease_text", "target_text"]
        ],
        config=config,
    )
    return compute_metrics(test_df["label"].to_numpy(), y_pred)


def deterministic_baselines(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    tfidf_scores = fit_tfidf_predict(
        train_df=history_df[
            ["diseaseId", "targetId", "label", "disease_text", "target_text"]
        ],
        test_df=test_df[
            ["diseaseId", "targetId", "label", "disease_text", "target_text"]
        ],
        config=BaselineConfig(models=("tfidf",)),
    )

    outputs = {
        "Target Mean Baseline": compute_metrics(
            test_df["label"].to_numpy(), target_mean_baseline(history_df, test_df)
        ),
        "TF-IDF cosine": compute_metrics(test_df["label"].to_numpy(), tfidf_scores),
        "Disease Mean Baseline": compute_metrics(
            test_df["label"].to_numpy(), disease_mean_baseline(history_df, test_df)
        ),
        "Open Targets Score": compute_metrics(
            test_df["label"].to_numpy(),
            test_df["score_past"].to_numpy(dtype=np.float32),
        ),
    }
    return outputs


def summarise_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model_name in MODEL_ORDER:
        group = run_df[run_df["Model"] == model_name].copy()
        if group.empty:
            continue
        n = int(len(group))
        roc_mean = float(group["ROC-AUC"].mean())
        pr_mean = float(group["PR-AUC"].mean())
        if n > 1:
            roc_sd = float(group["ROC-AUC"].std(ddof=1))
            pr_sd = float(group["PR-AUC"].std(ddof=1))
            crit = float(student_t.ppf(0.975, df=n - 1))
            roc_half = crit * roc_sd / np.sqrt(n)
            pr_half = crit * pr_sd / np.sqrt(n)
        else:
            roc_sd = 0.0
            pr_sd = 0.0
            roc_half = 0.0
            pr_half = 0.0
        rows.append(
            {
                "Model": model_name,
                "n_runs": n,
                "ROC-AUC": roc_mean,
                "ROC-AUC SD": roc_sd,
                "ROC-AUC CI Low": roc_mean - roc_half,
                "ROC-AUC CI High": roc_mean + roc_half,
                "PR-AUC": pr_mean,
                "PR-AUC SD": pr_sd,
                "PR-AUC CI Low": pr_mean - pr_half,
                "PR-AUC CI High": pr_mean + pr_half,
            }
        )
    return pd.DataFrame(rows)


def update_table2_csv(table2_path: Path, summary_df: pd.DataFrame) -> None:
    export = summary_df[
        ["Model", "ROC-AUC", "ROC-AUC SD", "PR-AUC", "PR-AUC SD"]
    ].copy()
    export.to_csv(table2_path, index=False)


def write_checkpoint(
    out_dir: Path, run_rows: list[dict[str, float | int | str]]
) -> None:
    """Persist partial and final outputs so long runs can be monitored safely."""
    if not run_rows:
        return
    run_df = pd.DataFrame(run_rows)
    summary_df = summarise_runs(run_df)
    run_df.to_csv(out_dir / "temporal_run_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "temporal_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    history_raw = pd.read_parquet(args.history)
    future_raw = pd.read_parquet(args.future)
    disease_df = pd.read_parquet(args.disease)
    target_df = pd.read_parquet(args.target)

    test_raw = build_temporal_test_set(history_raw, future_raw)
    test_raw = add_historical_score(history_raw, test_raw)
    test_raw["score_past"] = test_raw["score_past"].fillna(0.0)

    history_df = merge_df_dis_target(history_raw, disease_df, target_df)
    test_df = merge_df_dis_target(test_raw, disease_df, target_df)

    run_rows: list[dict[str, float | int | str]] = []
    deterministic = deterministic_baselines(history_df, test_df)
    for model_name, (roc_auc, pr_auc) in deterministic.items():
        run_rows.append(
            {
                "Model": model_name,
                "seed": "deterministic",
                "ROC-AUC": roc_auc,
                "PR-AUC": pr_auc,
            }
        )
    write_checkpoint(args.out_dir, run_rows)

    for seed in args.seeds:
        print(f"[seed {seed}] training OTRec", flush=True)
        otrec_metrics = train_otrec_once(
            history_df, test_df, seed=seed, epochs=args.otrec_epochs
        )
        run_rows.append(
            {
                "Model": "OTRec",
                "seed": seed,
                "ROC-AUC": otrec_metrics[0],
                "PR-AUC": otrec_metrics[1],
            }
        )
        write_checkpoint(args.out_dir, run_rows)
        print(
            f"[seed {seed}] OTRec ROC-AUC={otrec_metrics[0]:.6f} PR-AUC={otrec_metrics[1]:.6f}",
            flush=True,
        )

        print(f"[seed {seed}] training OTTree (CatBoost)", flush=True)
        ottree_metrics = train_ottree_once(history_df, test_df, seed=seed)
        run_rows.append(
            {
                "Model": "OTTree (CatBoost)",
                "seed": seed,
                "ROC-AUC": ottree_metrics[0],
                "PR-AUC": ottree_metrics[1],
            }
        )
        write_checkpoint(args.out_dir, run_rows)
        print(
            f"[seed {seed}] OTTree ROC-AUC={ottree_metrics[0]:.6f} PR-AUC={ottree_metrics[1]:.6f}",
            flush=True,
        )

        print(f"[seed {seed}] training Matrix Factorization", flush=True)
        mf_metrics = run_matrix_factorization_once(history_df, test_df, seed=seed)
        run_rows.append(
            {
                "Model": "Matrix Factorization",
                "seed": seed,
                "ROC-AUC": mf_metrics[0],
                "PR-AUC": mf_metrics[1],
            }
        )
        write_checkpoint(args.out_dir, run_rows)
        print(
            f"[seed {seed}] MF ROC-AUC={mf_metrics[0]:.6f} PR-AUC={mf_metrics[1]:.6f}",
            flush=True,
        )

    run_df = pd.DataFrame(run_rows)
    summary_df = summarise_runs(run_df)

    run_df.to_csv(args.out_dir / "temporal_run_metrics.csv", index=False)
    summary_df.to_csv(args.out_dir / "temporal_summary.csv", index=False)

    print("Temporal repeated-run summary")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if args.update_table2:
        update_table2_csv(args.table2, summary_df)
        print(f"\nUpdated temporal table CSV: {args.table2}")


if __name__ == "__main__":
    main()

"""Compute fast temporal baselines (MF + TF-IDF cosine) for OTRec.

This script mirrors the temporal split logic used in `2-Temporal-Eval.ipynb`:
- train/history: OTP 22.02 snapshot (`history_df.parquet`)
- future test: OTP 25.06 snapshot (`final_df.parquet`)
- test set: anti-join on (diseaseId, targetId, label)

It then computes two fast baselines using the same implementation used in
`baselines/run_baselines.py`:
- Matrix Factorization (`fit_mf_predict`)
- TF-IDF cosine (`fit_tfidf_predict`)

Optionally, it updates `Outputs/Table 2 - Temporal Prospective validation.csv`
by adding/replacing rows for these two baselines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from run_baselines import BaselineConfig, fit_mf_predict, fit_tfidf_predict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = ROOT.parent / "code" / "history_df.parquet"
DEFAULT_FUTURE = ROOT.parent / "code" / "final_df.parquet"
DEFAULT_DISEASE = ROOT.parent / "code" / "copy_proc" / "disease_df.parquet"
DEFAULT_TARGET = ROOT.parent / "code" / "copy_proc" / "target_df.parquet"
DEFAULT_TABLE2 = ROOT / "Outputs" / "Table 2 - Temporal Prospective validation.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--future", type=Path, default=DEFAULT_FUTURE)
    parser.add_argument("--disease", type=Path, default=DEFAULT_DISEASE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--table2", type=Path, default=DEFAULT_TABLE2)
    parser.add_argument(
        "--update-table2",
        action="store_true",
        help="Update Table 2 CSV with Matrix Factorization and TF-IDF rows.",
    )
    return parser.parse_args()


def merge_df_dis_target(
    df: pd.DataFrame, disease_df: pd.DataFrame, target_df: pd.DataFrame
) -> pd.DataFrame:
    """Attach disease/target text fields required by text baselines."""
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
    """Apply anti-join on (diseaseId, targetId, label), matching notebook logic."""
    join_keys = ["diseaseId", "targetId", "label"]
    test_df = (
        future_df.merge(history_df[join_keys], on=join_keys, how="left", indicator=True)
        .query('_merge == "left_only"')
        .drop(columns="_merge")
    )
    return test_df


def compute_metrics(y_true, y_score) -> tuple[float, float]:
    return float(roc_auc_score(y_true, y_score)), float(
        average_precision_score(y_true, y_score)
    )


def update_table2(
    table2_path: Path,
    mf_metrics: tuple[float, float],
    tfidf_metrics: tuple[float, float],
) -> pd.DataFrame:
    """Add or replace Matrix Factorization / TF-IDF rows in the temporal table CSV."""
    df = pd.read_csv(table2_path)

    replacement_rows = pd.DataFrame(
        [
            {
                "Model": "Matrix Factorization",
                "ROC-AUC": round(mf_metrics[0], 3),
                "PR-AUC": round(mf_metrics[1], 3),
            },
            {
                "Model": "TF-IDF cosine",
                "ROC-AUC": round(tfidf_metrics[0], 3),
                "PR-AUC": round(tfidf_metrics[1], 3),
            },
        ]
    )

    df = df[~df["Model"].isin(["Matrix Factorization", "TF-IDF cosine"])].copy()

    model_order = [
        "OTRec",
        "OTTree (CatBoost)",
        "Target Mean Baseline",
        "Matrix Factorization",
        "TF-IDF cosine",
        "Disease Mean Baseline",
        "Open Targets Score",
    ]

    df = pd.concat([df, replacement_rows], ignore_index=True)
    df["_order"] = (
        df["Model"].map({name: i for i, name in enumerate(model_order)}).fillna(10_000)
    )
    df = (
        df.sort_values(["_order", "Model"])
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )
    df.to_csv(table2_path, index=False)
    return df


def main() -> None:
    args = parse_args()

    history_df = pd.read_parquet(args.history)
    future_df = pd.read_parquet(args.future)
    disease_df = pd.read_parquet(args.disease)
    target_df = pd.read_parquet(args.target)

    test_df = build_temporal_test_set(history_df, future_df)
    history_df = merge_df_dis_target(
        history_df, disease_df=disease_df, target_df=target_df
    )
    test_df = merge_df_dis_target(test_df, disease_df=disease_df, target_df=target_df)

    config = BaselineConfig(models=("mf", "tfidf"))

    mf_scores = fit_mf_predict(
        train_df=history_df[
            ["diseaseId", "targetId", "label", "disease_text", "target_text"]
        ],
        test_df=test_df[
            ["diseaseId", "targetId", "label", "disease_text", "target_text"]
        ],
        config=config,
    )
    tfidf_scores = fit_tfidf_predict(
        train_df=history_df[
            ["diseaseId", "targetId", "label", "disease_text", "target_text"]
        ],
        test_df=test_df[
            ["diseaseId", "targetId", "label", "disease_text", "target_text"]
        ],
        config=config,
    )

    y_true = test_df["label"].to_numpy()
    mf_metrics = compute_metrics(y_true=y_true, y_score=mf_scores)
    tfidf_metrics = compute_metrics(y_true=y_true, y_score=tfidf_scores)

    print(f"Temporal test size: {len(test_df)}")
    print(f"Temporal prevalence: {test_df['label'].mean():.6f}")
    print(
        f"Matrix Factorization  ROC-AUC={mf_metrics[0]:.6f} PR-AUC={mf_metrics[1]:.6f}"
    )
    print(
        f"TF-IDF cosine         ROC-AUC={tfidf_metrics[0]:.6f} PR-AUC={tfidf_metrics[1]:.6f}"
    )

    if args.update_table2:
        updated = update_table2(
            args.table2, mf_metrics=mf_metrics, tfidf_metrics=tfidf_metrics
        )
        print("\nUpdated temporal table CSV:")
        print(updated.to_string(index=False))


if __name__ == "__main__":
    main()

"""Baselines + OTTree on the fully release-native 23.06 -> 25.12 split.

Reuses the paper's own baseline implementations verbatim from
OTRec/baselines/run_temporal_repeated.py and run_baselines.py -- no
reimplementation, so the numbers are directly comparable to Table 2.

Usage:
  python3 run_baselines_native.py cpu     # Target/Disease Mean, OT Score, TF-IDF, MF, Node2Vec
  python3 run_baselines_native.py ottree  # CatBoost, 5 seeds (GPU)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/d/Research/OpenTargetsTransfer/OTRec")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines"))

from run_baselines import BaselineConfig, fit_node2vec_predict
from run_temporal_repeated import (
    add_historical_score, build_temporal_test_set, compute_metrics, deterministic_baselines,
    merge_df_dis_target, run_matrix_factorization_once, train_ottree_once,
)

OUT = Path("/mnt/d/Research/OpenTargetsTransfer/rebuttal_scratch/native_2306")
MODE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
SEEDS = [42, 43, 44, 45, 46]


def load_split():
    history_raw = pd.read_parquet(OUT / "native2306_train_frame.parquet")
    future_raw = pd.read_parquet(OUT / "native2306_eval_frame.parquet")
    disease_df = pd.read_parquet(OUT / "native2306_disease_df.parquet")
    target_df = pd.read_parquet(OUT / "native2306_target_df.parquet")
    test_raw = add_historical_score(history_raw, build_temporal_test_set(history_raw, future_raw))
    test_raw["score_past"] = test_raw["score_past"].fillna(0.0)
    history_df = merge_df_dis_target(history_raw, disease_df, target_df)
    test_df = merge_df_dis_target(test_raw, disease_df, target_df)
    print(f"train {history_df.shape} | test {test_df.shape} | pos {int(test_df.label.sum()):,} "
          f"({test_df.label.mean():.2%})", flush=True)
    return history_df, test_df


def save(results, name):
    path = OUT / f"native2306_baselines_{name}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {path}")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"  {k:<38} ROC {v['roc']:.4f} PR {v['pr']:.4f}"
                  + (f"  (SD {v['roc_sd']:.4f} / {v['pr_sd']:.4f})" if "roc_sd" in v else ""))


if __name__ == "__main__":
    history_df, test_df = load_split()
    results = {}

    if MODE == "cpu":
        print("\n--- deterministic baselines (Target Mean, Disease Mean, OT Score, TF-IDF) ---", flush=True)
        for name, (roc, pr) in deterministic_baselines(history_df, test_df).items():
            results[name] = {"roc": roc, "pr": pr}
            print(f"  {name:<38} ROC {roc:.4f} PR {pr:.4f}", flush=True)

        print("\n--- Matrix Factorization (seed 42) ---", flush=True)
        roc, pr = run_matrix_factorization_once(history_df, test_df, 42)
        results["Matrix Factorization"] = {"roc": roc, "pr": pr}
        print(f"  ROC {roc:.4f} PR {pr:.4f}", flush=True)

        save(results, "cpu_partial")

    if MODE in ("cpu", "node2vec"):
        print("\n--- Node2Vec (seed 42) ---", flush=True)
        y_pred = fit_node2vec_predict(
            train_df=history_df[["diseaseId", "targetId", "label", "disease_text", "target_text"]],
            test_df=test_df[["diseaseId", "targetId", "label", "disease_text", "target_text"]],
            config=BaselineConfig(models=("node2vec",), random_state=42),
        )
        roc, pr = compute_metrics(test_df["label"].to_numpy(), y_pred)
        results["Node2Vec"] = {"roc": roc, "pr": pr}
        print(f"  ROC {roc:.4f} PR {pr:.4f}", flush=True)
        save(results, "cpu" if MODE == "cpu" else "node2vec")

    if MODE == "ottree":
        print("\n--- OTTree (CatBoost), 5 seeds ---", flush=True)
        rocs, prs = [], []
        for s in SEEDS:
            if s == SEEDS[0]:
                # seed 42: keep per-pair predictions for the shortlist analysis
                from catboost import CatBoostClassifier, Pool
                import tensorflow as tf
                feats = ["disease_text", "target_text", "diseaseId"]
                tr = Pool(history_df[feats], history_df["label"],
                          text_features=["disease_text", "target_text"], cat_features=["diseaseId"])
                te = Pool(test_df[feats], test_df["label"],
                          text_features=["disease_text", "target_text"], cat_features=["diseaseId"])
                params = {"depth": 8, "eval_metric": "AUC", "random_seed": s, "verbose": False}
                if tf.config.list_physical_devices("GPU"):
                    params["task_type"] = "GPU"
                m = CatBoostClassifier(**params)
                m.fit(tr)
                yp = m.predict_proba(te)[:, 1]
                test_df[["diseaseId", "targetId", "label"]].assign(ottree=yp).to_parquet(
                    OUT / "native2306_ottree_preds.parquet", index=False)
                roc, pr = compute_metrics(test_df["label"].to_numpy(), yp)
            else:
                roc, pr = train_ottree_once(history_df, test_df, s)
            rocs.append(roc); prs.append(pr)
            print(f"  seed {s}: ROC {roc:.4f} PR {pr:.4f}", flush=True)
        results["OTTree (CatBoost)"] = {
            "roc": float(np.mean(rocs)), "pr": float(np.mean(prs)),
            "roc_sd": float(np.std(rocs, ddof=1)), "pr_sd": float(np.std(prs, ddof=1)),
            "seeds": SEEDS, "roc_all": rocs, "pr_all": prs,
        }
        save(results, "ottree")

    if MODE not in ("cpu", "node2vec", "ottree"):
        raise SystemExit(f"unknown mode {MODE!r}")

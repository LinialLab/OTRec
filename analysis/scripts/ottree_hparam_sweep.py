"""Hyperparameter sweep for the matched OTTree (CatBoost) baseline, temporal setting.

Addresses the reviewer objection that baselines were compared at a single fixed
configuration and were therefore "not given their best chance". Mirrors
train_ottree_once() from baselines/run_temporal_repeated.py (same features, same
train/test construction, same seed); only the CatBoost params vary.

Writes analysis/results/ottree_hparam_sweep.csv.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # .../OTRec
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines"))

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score

from run_temporal_repeated import (
    add_historical_score,
    build_temporal_test_set,
    merge_df_dis_target,
)

OUT = ROOT / "analysis" / "results"
SEED = 42

FEATURES = ["disease_text", "target_text", "diseaseId"]
TEXT_FEATURES = ["disease_text", "target_text"]
CAT_FEATURES = ["diseaseId"]

# Published configuration first, then a spread over the parameters that matter
# most for this model class. Kept small deliberately: the claim under test is
# "does tuning the matched baseline overturn the comparison", not "find the optimum".
GRID = [
    {"depth": 8},                                                        # published
    {"depth": 6},
    {"depth": 10},
    {"depth": 8, "learning_rate": 0.03},
    {"depth": 8, "learning_rate": 0.15},
    {"depth": 8, "iterations": 2000},
    {"depth": 8, "l2_leaf_reg": 10.0},
    {"depth": 10, "learning_rate": 0.03, "iterations": 2000},
    {"depth": 8, "auto_class_weights": "Balanced"},
]


def main() -> None:
    history_raw = pd.read_parquet(ROOT.parent / "code" / "history_df.parquet")
    future_raw = pd.read_parquet(ROOT.parent / "code" / "final_df.parquet")
    disease_df = pd.read_parquet(ROOT.parent / "code" / "copy_proc" / "disease_df.parquet")
    target_df = pd.read_parquet(ROOT.parent / "code" / "copy_proc" / "target_df.parquet")

    test_raw = add_historical_score(history_raw, build_temporal_test_set(history_raw, future_raw))
    test_raw["score_past"] = test_raw["score_past"].fillna(0.0)
    history_df = merge_df_dis_target(history_raw, disease_df, target_df)
    test_df = merge_df_dis_target(test_raw, disease_df, target_df)
    print("history", history_df.shape, "test", test_df.shape,
          "test pos", int(test_df.label.sum()), flush=True)

    train_pool = Pool(data=history_df[FEATURES], label=history_df["label"],
                      text_features=TEXT_FEATURES, cat_features=CAT_FEATURES)
    test_pool = Pool(data=test_df[FEATURES], label=test_df["label"],
                     text_features=TEXT_FEATURES, cat_features=CAT_FEATURES)
    y = test_df["label"].to_numpy()

    rows = []
    for i, cfg in enumerate(GRID, 1):
        params = {"eval_metric": "AUC", "random_seed": SEED, "verbose": False, **cfg}
        model = CatBoostClassifier(**params)
        model.fit(train_pool)
        pred = model.predict_proba(test_pool)[:, 1]
        roc, pr = roc_auc_score(y, pred), average_precision_score(y, pred)
        rows.append({**cfg, "ROC_AUC": roc, "PR_AUC": pr})
        print(f"[{i}/{len(GRID)}] {cfg}  ROC {roc:.4f}  PR {pr:.4f}", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "ottree_hparam_sweep.csv", index=False)
    print("\nbest ROC", df.ROC_AUC.max(), "| best PR", df.PR_AUC.max())
    print("saved", OUT / "ottree_hparam_sweep.csv", flush=True)


if __name__ == "__main__":
    main()

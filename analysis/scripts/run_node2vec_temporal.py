"""Run the EXISTING OTRec Node2Vec baseline under the temporal (2022 -> 2025) protocol.

Reuses, without modification:
  - run_baselines.fit_node2vec_predict / BaselineConfig
  - run_temporal_repeated.build_temporal_test_set / merge_df_dis_target
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BASELINES = str(Path(__file__).resolve().parents[2] / "baselines")
CODE = Path(__file__).resolve().parents[3] / "code"
OUT = Path(__file__).resolve().parents[1] / "results"      # OTRec/analysis/results
sys.path.insert(0, BASELINES)

from run_baselines import BaselineConfig, fit_node2vec_predict  # noqa: E402
from run_temporal_repeated import (  # noqa: E402
    build_temporal_test_set,
    merge_df_dis_target,
)

SEED = 42

t0 = time.time()
history_raw = pd.read_parquet(CODE / "history_df.parquet")
future_raw = pd.read_parquet(CODE / "final_df.parquet")
disease_df = pd.read_parquet(CODE / "copy_proc" / "disease_df.parquet")
target_df = pd.read_parquet(CODE / "copy_proc" / "target_df.parquet")
print(f"loaded history={history_raw.shape} future={future_raw.shape} "
      f"({time.time()-t0:.1f}s)", flush=True)

test_raw = build_temporal_test_set(history_raw, future_raw)
history_df = merge_df_dis_target(history_raw, disease_df, target_df)
test_df = merge_df_dis_target(test_raw, disease_df, target_df)

cols = ["diseaseId", "targetId", "label", "disease_text", "target_text"]
n_test = len(test_df)
n_pos = int(test_df["label"].sum())
n_hist_pos = int((history_df["label"] == 1).sum())
print(f"test pairs={n_test} positives={n_pos} rate={n_pos/n_test:.6f}", flush=True)
print(f"history rows={len(history_df)} history positives={n_hist_pos}", flush=True)

config = BaselineConfig(models=("node2vec",), random_state=SEED)
t_fit = time.time()
preds = fit_node2vec_predict(
    train_df=history_df[cols],
    test_df=test_df[cols],
    config=config,
)
fit_secs = time.time() - t_fit
print(f"fit+predict done in {fit_secs:.1f}s", flush=True)

y = test_df["label"].to_numpy()
roc = float(roc_auc_score(y, preds))
pr = float(average_precision_score(y, preds))

out_df = test_df[["diseaseId", "targetId", "label"]].copy()
out_df["pred"] = np.asarray(preds, dtype=np.float32)
out_df.to_parquet(OUT / "node2vec_temporal_preds.parquet", index=False)

result = {
    "model": "Node2Vec (OTRec baselines.fit_node2vec_predict)",
    "protocol": "temporal 2022.02 history -> 2025.06 future (anti-join test set)",
    "seed": SEED,
    "roc_auc": roc,
    "pr_auc": pr,
    "n_test_pairs": n_test,
    "n_positives": n_pos,
    "positive_rate": n_pos / n_test,
    "n_history_rows": len(history_df),
    "n_history_positive_edges": n_hist_pos,
    "n_unique_preds": int(pd.Series(out_df["pred"]).nunique()),
    "n_unique_test_diseases": int(test_df["diseaseId"].nunique()),
    "fit_seconds": fit_secs,
    "total_seconds": time.time() - t0,
    "params": {
        "dimensions": config.node2vec_dimensions,
        "walk_length": config.node2vec_walk_length,
        "num_walks": config.node2vec_num_walks,
        "window": config.node2vec_window,
        "epochs": config.node2vec_epochs,
        "p": 1.0,
        "q": 1.0,
        "workers": 4,
        "sg": 1,
        "min_count": 1,
    },
}
print(json.dumps(result, indent=2), flush=True)
(OUT / "node2vec_temporal_result.json").write_text(json.dumps(result, indent=2))
print(f"ROC-AUC={roc:.4f} PR-AUC={pr:.4f}", flush=True)

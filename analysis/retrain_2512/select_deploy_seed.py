"""Pick the deployment seed by VALIDATION metrics only, then report per-disease
held-out ranking for the winner (reported, not used for selection).

Selection metric: last-epoch val_cls_pr_auc from the training logs (the metric
the training callbacks monitor a proxy of). Test-set numbers are printed for
the record but play no part in the choice.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/mnt/d/Research/OpenTargetsTransfer/retrain_2512")

LOGS = {42: R / "train_main_2512.log", 43: R / "train_s43.log", 44: R / "train_s44.log"}


def last_val_pr(log_path):
    vals = re.findall(r"val_cls_pr_auc: ([0-9.]+)", log_path.read_text())
    return float(vals[-1]) if vals else None


print(f"{'seed':>5} {'val_cls_pr_auc(last)':>22}")
scores = {}
for seed, log in LOGS.items():
    v = last_val_pr(log)
    scores[seed] = v
    print(f"{seed:>5} {v if v is not None else 'missing':>22}")

winner = max((s for s in scores if scores[s] is not None), key=lambda s: scores[s])
print(f"\nWINNER by validation: seed {winner}")

suffix = "" if winner == 42 else f"_s{winner}"
weights = R / f"model{suffix}.weights.h5"
assert weights.exists(), weights
print(f"weights: {weights.name}")

# Held-out per-disease ranking for the winner -- reported only.
sys.path.insert(0, "/mnt/d/Research/OpenTargetsTransfer/OTRec/gradio")
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from dl_model_def import build_two_tower_model
from vocab_io import load_vocabularies, apply_vocabularies

full = pd.read_parquet(R / "df_learn_2512.parquet")
train_tids, test_tids = train_test_split(
    full["targetId"].unique(), test_size=0.2, random_state=42, shuffle=True,
    stratify=full.drop_duplicates(subset=["targetId"])["label"])
test = full[full["targetId"].isin(test_tids)]

keras.backend.clear_session()
m = build_two_tower_model(full)
apply_vocabularies(m, load_vocabularies(R / "vocabs_2512.json.gz"))
m.load_weights(str(weights))
ds = tf.data.Dataset.from_tensor_slices({
    "query": {"disease_text": test["disease_text"], "diseaseId": test["diseaseId"]},
    "candidate": {"target_text": test["target_text"], "targetId": test["targetId"]}}).batch(2048)
p = m.predict(ds, verbose=0)["cls"].ravel()
print(f"winner held-out: ROC {roc_auc_score(test.label, p):.4f} PR {average_precision_score(test.label, p):.4f}")

work = test[["diseaseId", "label"]].copy()
work["p"] = p
hits1, mrrs = [], []
posd = work.groupby("diseaseId").label.transform("sum") > 0
for _, g in work[posd].groupby("diseaseId"):
    order = np.argsort(-g["p"].to_numpy(), kind="stable")
    best = int(np.argmax(g.label.to_numpy()[order])) + 1
    hits1.append(best <= 1)
    mrrs.append(1 / best)
print(f"winner held-out per-disease: Hit@1 {np.mean(hits1):.3f} MRR {np.mean(mrrs):.3f} "
      f"(25.06 fixed reference: 0.800 / 0.856; seed-42 25.12: 0.748 / 0.812)")
print(f"cls_head kernel sign: {m.cls_head.get_weights()[0].ravel()[0]:+.2f}")
print(f"\nDEPLOY = seed {winner}: package swaps only the weights file "
      f"(vocab + df_learn_sub cover are frame-derived, seed-independent)")

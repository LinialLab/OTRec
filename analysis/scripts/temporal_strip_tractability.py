"""Leakage robustness check for the temporal experiment.

Identical to the seed-42 temporal reproduction, except that clinical-precedence
tractability tokens (which encode post-2022 drug/clinical status in the 25.06
annotation snapshot) are stripped from target_text before training.

Compare against the unmodified seed-42 run: ROC 0.876933 / PR 0.286191.
No repo file is modified.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]                 # .../OTRec
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines"))

import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

from dl_model_def import build_two_tower_model
from run_temporal_repeated import (
    add_historical_score, build_temporal_test_set, make_ds, merge_df_dis_target, set_global_seed,
)

OUT = Path(__file__).resolve().parents[1] / "results"      # OTRec/analysis/results
SEED, EPOCHS = 42, 6

# Clinical-precedence tractability buckets = the only post-2022 OUTCOME signal in the text.
LEAK_TOKENS = ["Approved Drug", "Advanced Clinical", "Phase 1 Clinical", "Clinical Precedence"]
LEAK_RE = re.compile("|".join(re.escape(t) for t in LEAK_TOKENS), flags=re.IGNORECASE)

history_raw = pd.read_parquet(ROOT.parent / "code" / "history_df.parquet")
future_raw = pd.read_parquet(ROOT.parent / "code" / "final_df.parquet")
disease_df = pd.read_parquet(ROOT.parent / "code" / "copy_proc" / "disease_df.parquet")
target_df = pd.read_parquet(ROOT.parent / "code" / "copy_proc" / "target_df.parquet")

n_before = target_df["target_text_embed"].astype(str).str.contains(LEAK_RE).sum()
target_df = target_df.copy()
target_df["target_text_embed"] = (
    target_df["target_text_embed"].astype(str).str.replace(LEAK_RE, " ", regex=True)
)
n_after = target_df["target_text_embed"].str.contains(LEAK_RE).sum()
print(f"targets carrying clinical-precedence tokens: {n_before} -> {n_after} after strip", flush=True)

test_raw = add_historical_score(history_raw, build_temporal_test_set(history_raw, future_raw))
test_raw["score_past"] = test_raw["score_past"].fillna(0.0)
history_df = merge_df_dis_target(history_raw, disease_df, target_df)
test_df = merge_df_dis_target(test_raw, disease_df, target_df)
print("history", history_df.shape, "test", test_df.shape, "test pos", int(test_df.label.sum()), flush=True)

set_global_seed(SEED)
keras.backend.clear_session()
train_tids, val_tids = train_test_split(
    history_df["targetId"].unique(), test_size=0.01, random_state=SEED, shuffle=True
)
train_df = history_df[history_df["targetId"].isin(train_tids)].copy()
val_df = history_df[history_df["targetId"].isin(val_tids)].copy()

model = build_two_tower_model(history_df)
model.compile(
    optimizer=keras.optimizers.Adam(7e-3),
    loss={"cls": keras.losses.BinaryCrossentropy(from_logits=False),
          "score": keras.losses.MeanSquaredError()},
    loss_weights={"cls": 1.0, "score": 0.1},
    metrics={"cls": [keras.metrics.AUC(name="auc"), keras.metrics.AUC(curve="PR", name="pr_auc")],
             "score": [keras.metrics.RootMeanSquaredError(name="rmse")]},
)
train_ds = (make_ds(train_df).shuffle(200_000, seed=SEED, reshuffle_each_iteration=True)
            .batch(512).prefetch(tf.data.AUTOTUNE))
val_ds = make_ds(val_df).batch(2048).prefetch(tf.data.AUTOTUNE)
test_ds = make_ds(test_df[train_df.columns.to_list()]).batch(1024).prefetch(tf.data.AUTOTUNE)
callbacks = [
    keras.callbacks.ReduceLROnPlateau("val_cls_loss", mode="min", factor=0.2, patience=1),
    keras.callbacks.EarlyStopping("val_cls_loss", mode="min", patience=2, restore_best_weights=False),
]
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)
pred = model.predict(test_ds, verbose=0)["cls"].ravel()
y = test_df["label"].to_numpy()
roc, pr = roc_auc_score(y, pred), average_precision_score(y, pred)
print(f"STRIPPED seed42 ROC {roc:.6f} PR {pr:.6f}   (unstripped seed42: 0.876933 / 0.286191)", flush=True)

hist = history_raw
ind = hist[hist.label == 1].groupby("targetId")["diseaseId"].nunique()
out = test_df[["diseaseId", "targetId", "label"]].copy()
out["otrec_stripped"] = pred
out["n_ind_2022"] = out.targetId.map(ind).fillna(0).astype(int)
out["bin"] = np.where(out.n_ind_2022 == 0, "0", np.where(out.n_ind_2022 == 1, "1", ">=2"))
for b in ["0", "1", ">=2"]:
    d = out[out.bin == b]
    print(f"  bin {b:>3}: n={len(d):>6} pos={int(d.label.sum()):>5} "
          f"ROC {roc_auc_score(d.label, d.otrec_stripped):.4f} "
          f"PR {average_precision_score(d.label, d.otrec_stripped):.4f}", flush=True)
out.to_parquet(OUT / "temporal_preds_seed42_stripped.parquet", index=False)
print("saved", OUT / "temporal_preds_seed42_stripped.parquet", flush=True)

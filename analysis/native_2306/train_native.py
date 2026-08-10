"""Fully release-native temporal replication: train 23.06 -> eval 25.12.

Protocol identical to the paper's temporal experiment (and to
rebuttal_scratch/temporal_2202_features.py): same architecture, optimizer,
callbacks, epochs, split construction, anti-join test definition. The only
differences are the release pair and that ALL inputs (labels, auxiliary score,
annotation text) come from the training release, 23.06 — no era mixing by
construction.

Usage: python3 train_native.py [seed]   (default 42)
Outputs: native2306_preds_s{seed}.parquet + metrics printed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/d/Research/OpenTargetsTransfer/OTRec")
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

OUT = Path("/mnt/d/Research/OpenTargetsTransfer/rebuttal_scratch/native_2306")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
EPOCHS = 6

history_raw = pd.read_parquet(OUT / "native2306_train_frame.parquet")
future_raw = pd.read_parquet(OUT / "native2306_eval_frame.parquet")
disease_df = pd.read_parquet(OUT / "native2306_disease_df.parquet")
target_df = pd.read_parquet(OUT / "native2306_target_df.parquet")

test_raw = add_historical_score(history_raw, build_temporal_test_set(history_raw, future_raw))
test_raw["score_past"] = test_raw["score_past"].fillna(0.0)
history_df = merge_df_dis_target(history_raw, disease_df, target_df)
test_df = merge_df_dis_target(test_raw, disease_df, target_df)
print(f"train {history_df.shape} | test {test_df.shape} | test pos {int(test_df.label.sum()):,} "
      f"({test_df.label.mean():.2%})", flush=True)
empty_d = (test_df["disease_text"].str.strip() == "").mean()
empty_t = (test_df["target_text"].str.strip() == "").mean()
print(f"native coverage on test set: empty disease_text={empty_d:.4f} empty target_text={empty_t:.4f}", flush=True)

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

sp = test_df["score_past"].to_numpy()
roc_b, pr_b = roc_auc_score(y, sp), average_precision_score(y, sp)
print(f"NATIVE 23.06->25.12 seed{SEED} ROC {roc:.6f} PR {pr:.6f} | "
      f"OT-Score temporal baseline ROC {roc_b:.6f} PR {pr_b:.6f}", flush=True)

ind = history_raw[history_raw.label == 1].groupby("targetId")["diseaseId"].nunique()
out = test_df[["diseaseId", "targetId", "label", "score_past"]].copy()
out["otrec_native"] = pred
out["n_ind_2306"] = out.targetId.map(ind).fillna(0).astype(int)
out["bin"] = np.where(out.n_ind_2306 == 0, "0", np.where(out.n_ind_2306 == 1, "1", ">=2"))
for b in ["0", "1", ">=2"]:
    d = out[out.bin == b]
    if len(d) and d.label.sum():
        print(f"  bin {b:>3}: n={len(d):>6} pos={int(d.label.sum()):>5} "
              f"ROC {roc_auc_score(d.label, d.otrec_native):.4f} "
              f"PR {average_precision_score(d.label, d.otrec_native):.4f}", flush=True)
out.to_parquet(OUT / f"native2306_preds_s{SEED}.parquet", index=False)
print("saved", OUT / f"native2306_preds_s{SEED}.parquet", flush=True)

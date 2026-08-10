"""Retrain the deployed OTRec main model on Release 25.12.

Ports the exact main-model training cell of code/1-Train-DL-Retriever.ipynb
(cells ~8-14, lines 800-1031 of the nbconverted script): 80/20 target-holdout
split (seed 42, stratified by per-target label), 5% of train targets carved as
validation (seed 42), Adam(8e-3), loss_weights {cls:1.0, score:0.1}, batch 1024
train / 2048 val, ReduceLROnPlateau + EarlyStopping on val_cls_loss, 7 epochs.

Uses OTRec/gradio/dl_model_def.py (byte-identical to code/dl_model_def.py
except trailing commented-out code) -- the exact module that will load these
weights at serve time, to eliminate any drift between train-time and
serve-time architecture.

model = build_two_tower_model(df_learn) is built on the FULL label frame (not
just train_df), exactly as the notebook does and as OTRec/gradio/app.py does
at cold start -- this fixes the vocabulary that must ship as a matched set
with the weights.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/d/Research/OpenTargetsTransfer/OTRec/gradio")

import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

from dl_model_def import build_two_tower_model

OUT = Path("/mnt/d/Research/OpenTargetsTransfer/retrain_2512")
import sys as _sys
SEED = int(_sys.argv[1]) if len(_sys.argv) > 1 else 42
EPOCHS = 7
SUFFIX = "" if SEED == 42 else f"_s{SEED}"

df_learn = pd.read_parquet(OUT / "df_learn_2512.parquet")
print(f"df_learn {df_learn.shape}, positives {int(df_learn.label.sum()):,} ({df_learn.label.mean():.2%})",
      flush=True)

train_tids, test_tids = train_test_split(
    df_learn["targetId"].unique(), test_size=0.2, random_state=42, shuffle=True,
    stratify=df_learn.drop_duplicates(subset=["targetId"])["label"],
)
train_df_full = df_learn[df_learn["targetId"].isin(train_tids)].copy()
test_df = df_learn[df_learn["targetId"].isin(test_tids)].copy()

train_tids, val_tids = train_test_split(train_tids, test_size=0.05, random_state=42, shuffle=True)
val_df = train_df_full[train_df_full["targetId"].isin(val_tids)].copy()
train_df = train_df_full[train_df_full["targetId"].isin(train_tids)].copy()

assert set(train_tids).isdisjoint(val_tids)
assert set(train_tids).isdisjoint(test_tids)
assert set(val_tids).isdisjoint(test_tids)
assert len(train_df) + len(val_df) + len(test_df) == len(df_learn)
print(f"train {train_df.shape} | val {val_df.shape} | test {test_df.shape}", flush=True)
print(f"label rate: train {train_df.label.mean():.4f} val {val_df.label.mean():.4f} "
      f"test {test_df.label.mean():.4f}", flush=True)


def make_ds(df: pd.DataFrame) -> tf.data.Dataset:
    feats = {
        "query": {"disease_text": df["disease_text"], "diseaseId": df["diseaseId"]},
        "candidate": {"target_text": df["target_text"], "targetId": df["targetId"]},
    }
    y = {"cls": df["label"].astype("float32"), "score": df["score"].astype("float32")}
    return tf.data.Dataset.from_tensor_slices((feats, y))


tf.keras.utils.set_random_seed(SEED)
keras.backend.clear_session()

model = build_two_tower_model(df_learn)  # full label frame -> vocab fixed here
model.compile(
    optimizer=keras.optimizers.Adam(8e-3),
    loss={"cls": keras.losses.BinaryCrossentropy(from_logits=False),
          "score": keras.losses.MeanSquaredError()},
    loss_weights={"cls": 1.0, "score": 0.1},
    metrics={"cls": [keras.metrics.AUC(name="auc"), keras.metrics.AUC(curve="PR", name="pr_auc")],
             "score": [keras.metrics.RootMeanSquaredError(name="rmse")]},
)

train_ds = make_ds(train_df).shuffle(300_000, seed=SEED).batch(1024).prefetch(tf.data.AUTOTUNE)
val_ds = make_ds(val_df).batch(2048).prefetch(tf.data.AUTOTUNE)
test_ds = make_ds(test_df).batch(2048).prefetch(tf.data.AUTOTUNE)

callbacks = [
    keras.callbacks.ReduceLROnPlateau("val_cls_loss", mode="min", factor=0.2, patience=1),
    keras.callbacks.EarlyStopping("val_cls_loss", mode="min", patience=2, restore_best_weights=False),
]
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

test_pred = model.predict(test_ds, verbose=0)["cls"].ravel()
y_test = test_df["label"].to_numpy()
roc, pr = roc_auc_score(y_test, test_pred), average_precision_score(y_test, test_pred)
print(f"MAIN MODEL 25.12 -- held-out target test: ROC-AUC {roc:.4f}  PR-AUC {pr:.4f}", flush=True)
print("Notebook's logged 25.06 reference (same protocol, different release): "
      "Test AUC 0.9454 | PR-AUC 0.8504 (docstring comment, cell ~984)", flush=True)

model.save_weights(str(OUT / f"model{SUFFIX}.weights.h5"))
# The matched vocabulary frame is df_learn_2512.parquet itself (written by
# build_2512_frames.py); no separate copy needed.
print(f"saved model{SUFFIX}.weights.h5 (vocab frame: df_learn_2512.parquet)", flush=True)

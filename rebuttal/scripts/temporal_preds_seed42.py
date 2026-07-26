"""Reproduce the seed-42 temporal run of baselines/run_temporal_repeated.py and SAVE per-pair predictions.

Mirrors train_otrec_once / train_ottree_once verbatim (same split, model, optimizer,
callbacks, epochs); the only change is keeping y_pred instead of discarding it.
No repo file is modified.
"""
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
    add_historical_score,
    build_temporal_test_set,
    make_ds,
    merge_df_dis_target,
    set_global_seed,
)

OUT = Path(__file__).resolve().parents[1] / "results"      # OTRec/rebuttal/results
SEED, EPOCHS = 42, 6

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

# ---- OTRec: verbatim body of train_otrec_once, plus y_pred retained ----
set_global_seed(SEED)
keras.backend.clear_session()
train_tids, val_tids = train_test_split(
    history_df["targetId"].unique(), test_size=0.01, random_state=SEED, shuffle=True
)
train_df = history_df[history_df["targetId"].isin(train_tids)].copy()
val_df = history_df[history_df["targetId"].isin(val_tids)].copy()
assert set(train_tids).isdisjoint(val_tids)

model = build_two_tower_model(history_df)
model.compile(
    optimizer=keras.optimizers.Adam(7e-3),
    loss={"cls": keras.losses.BinaryCrossentropy(from_logits=False),
          "score": keras.losses.MeanSquaredError()},
    loss_weights={"cls": 1.0, "score": 0.1},
    metrics={"cls": [keras.metrics.AUC(name="auc"), keras.metrics.AUC(curve="PR", name="pr_auc")],
             "score": [keras.metrics.RootMeanSquaredError(name="rmse")]},
)
train_ds = (make_ds(train_df)
            .shuffle(200_000, seed=SEED, reshuffle_each_iteration=True)
            .batch(512).prefetch(tf.data.AUTOTUNE))
val_ds = make_ds(val_df).batch(2048).prefetch(tf.data.AUTOTUNE)
test_ds = make_ds(test_df[train_df.columns.to_list()]).batch(1024).prefetch(tf.data.AUTOTUNE)
callbacks = [
    keras.callbacks.ReduceLROnPlateau("val_cls_loss", mode="min", factor=0.2, patience=1),
    keras.callbacks.EarlyStopping("val_cls_loss", mode="min", patience=2, restore_best_weights=False),
]
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)
otrec_pred = model.predict(test_ds, verbose=0)["cls"].ravel()
y = test_df["label"].to_numpy()
print(f"OTRec  seed42 ROC {roc_auc_score(y, otrec_pred):.6f} "
      f"PR {average_precision_score(y, otrec_pred):.6f}  (reported seed42: 0.869770 / 0.280826)",
      flush=True)

# ---- OTTree: verbatim body of train_ottree_once, plus y_pred retained ----
from catboost import CatBoostClassifier, Pool

feats = ["disease_text", "target_text", "diseaseId"]
train_pool = Pool(data=history_df[feats], label=history_df["label"],
                  text_features=["disease_text", "target_text"], cat_features=["diseaseId"])
test_pool = Pool(data=test_df[feats], label=test_df["label"],
                 text_features=["disease_text", "target_text"], cat_features=["diseaseId"])
params = {"depth": 8, "eval_metric": "AUC", "random_seed": SEED, "verbose": False}
if tf.config.list_physical_devices("GPU"):
    params["task_type"] = "GPU"
cb = CatBoostClassifier(**params)
cb.fit(train_pool)
ottree_pred = cb.predict_proba(test_pool)[:, 1]
print(f"OTTree seed42 ROC {roc_auc_score(y, ottree_pred):.6f} "
      f"PR {average_precision_score(y, ottree_pred):.6f}  (reported seed42: 0.859945 / 0.228965)",
      flush=True)

out = test_df[["diseaseId", "targetId", "label", "score_past"]].copy()
out["otrec"] = otrec_pred
out["ottree"] = ottree_pred
out.to_parquet(OUT / "temporal_preds_seed42.parquet", index=False)
print("saved", OUT / "temporal_preds_seed42.parquet", out.shape, flush=True)

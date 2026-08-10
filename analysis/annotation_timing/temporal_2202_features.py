"""22.02-native-features robustness check for the temporal experiment.

Identical to the seed-42 temporal reproduction, except disease_text/target_text
are rebuilt entirely from Open Targets Release 22.02 raw tables (via
rebuttal_scratch/build_2202_features.py) instead of the newer OTP snapshot.
This eliminates ALL post-2022 information from annotation text by
construction, not just the clinical-precedence tractability tokens targeted
by the strip ablation (temporal_strip_tractability.py).

Caveat baked into interpretation: 22.02 annotation coverage is much sparser
than the modern snapshot (58.5% of test pairs have non-empty 22.02 disease
text vs ~100% under the newer snapshot; target text coverage 99.8%). Any
delta here conflates leak removal with this coverage loss -- report next to
the strip ablation, which isolates the outcome-encoding channel alone.

Compare against:
  unstripped (newer-snapshot features) seed42: ROC 0.876933 / PR 0.286191
  stripped (clinical tokens removed)   seed42: ROC 0.870100 / PR 0.282100 (approx, see temporal_strip_tractability.py)

No repo file is modified.
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

OUT = Path("/mnt/d/Research/OpenTargetsTransfer/rebuttal_scratch")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
EPOCHS = 6

history_raw = pd.read_parquet(ROOT.parent / "code" / "history_df.parquet")
future_raw = pd.read_parquet(ROOT.parent / "code" / "final_df.parquet")
disease_df = pd.read_parquet(OUT / "disease_df_2202.parquet")
target_df = pd.read_parquet(OUT / "target_df_2202.parquet")

test_raw = add_historical_score(history_raw, build_temporal_test_set(history_raw, future_raw))
test_raw["score_past"] = test_raw["score_past"].fillna(0.0)
history_df = merge_df_dis_target(history_raw, disease_df, target_df)
test_df = merge_df_dis_target(test_raw, disease_df, target_df)
print("history", history_df.shape, "test", test_df.shape, "test pos", int(test_df.label.sum()), flush=True)

empty_dtext = (test_df["disease_text"].str.strip() == "").mean()
empty_ttext = (test_df["target_text"].str.strip() == "").mean()
print(f"22.02-native coverage on test set: empty disease_text={empty_dtext:.4f} empty target_text={empty_ttext:.4f}", flush=True)

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
print(f"22.02-NATIVE seed{SEED} ROC {roc:.6f} PR {pr:.6f}   (unstripped seed42: 0.876933 / 0.286191)", flush=True)

hist = history_raw
ind = hist[hist.label == 1].groupby("targetId")["diseaseId"].nunique()
out = test_df[["diseaseId", "targetId", "label"]].copy()
out["otrec_2202native"] = pred
out["n_ind_2022"] = out.targetId.map(ind).fillna(0).astype(int)
out["bin"] = np.where(out.n_ind_2022 == 0, "0", np.where(out.n_ind_2022 == 1, "1", ">=2"))
for b in ["0", "1", ">=2"]:
    d = out[out.bin == b]
    print(f"  bin {b:>3}: n={len(d):>6} pos={int(d.label.sum()):>5} "
          f"ROC {roc_auc_score(d.label, d.otrec_2202native):.4f} "
          f"PR {average_precision_score(d.label, d.otrec_2202native):.4f}", flush=True)
suffix = "" if SEED == 42 else f"_s{SEED}"
out.to_parquet(OUT / f"temporal_preds_seed42_2202native{suffix}.parquet", index=False)
print("saved", OUT / f"temporal_preds_seed42_2202native{suffix}.parquet", flush=True)

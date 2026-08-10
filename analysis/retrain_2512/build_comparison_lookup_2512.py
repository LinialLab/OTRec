"""Build comparison_lookup_2512.parquet for the app's "comparison" panel.

SCOPE REDUCTION (documented, not silent): the notebook's full 5x5x5 CV
(run_groupwise_cv) builds a bespoke per-fold model subclass; faithfully
porting it is a large effort for a feature secondary to the actual ask. This
script instead reuses train_main_2512.py's target-disjoint held-out test split
(20% held-out targets, single split) for OTRec, and trains one OTTree
(CatBoost, same recipe as
OTRec/baselines/run_temporal_repeated.py::train_ottree_once) on the matching
train split, scored on the same held-out test set. Single-split numbers, not
a CV replacement -- fine for the app's illustrative comparison panel, not
suitable for benchmarking claims.

Schema matches OTRec/gradio/runtime_data.py::COMPARISON_COLUMNS exactly:
diseaseId, targetId, ot_score, known_label, otrec_oof_pred, ottree_pred.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/mnt/d/Research/OpenTargetsTransfer/OTRec/gradio")
import tensorflow as tf
from tensorflow import keras

from dl_model_def import build_two_tower_model

REPO = Path("/mnt/d/Research/OpenTargetsTransfer")
OUT = REPO / "retrain_2512"
SEED = 42

df_learn = pd.read_parquet(OUT / "df_learn_2512.parquet")
train_tids, test_tids = train_test_split(
    df_learn["targetId"].unique(), test_size=0.2, random_state=SEED, shuffle=True,
    stratify=df_learn.drop_duplicates(subset=["targetId"])["label"],
)
train_df_full = df_learn[df_learn["targetId"].isin(train_tids)].copy()
test_df = df_learn[df_learn["targetId"].isin(test_tids)].copy()
train_tids, val_tids = train_test_split(train_tids, test_size=0.05, random_state=SEED, shuffle=True)
val_df = train_df_full[train_df_full["targetId"].isin(val_tids)].copy()
train_df = train_df_full[train_df_full["targetId"].isin(train_tids)].copy()
print(f"train {train_df.shape} | val {val_df.shape} | test {test_df.shape}", flush=True)


def make_ds(df):
    feats = {"query": {"disease_text": df["disease_text"], "diseaseId": df["diseaseId"]},
              "candidate": {"target_text": df["target_text"], "targetId": df["targetId"]}}
    y = {"cls": df["label"].astype("float32"), "score": df["score"].astype("float32")}
    return tf.data.Dataset.from_tensor_slices((feats, y))


# --- OTRec: reload the already-trained weights (train_main_2512.py) and score the held-out test set ---
tf.keras.utils.set_random_seed(SEED)
keras.backend.clear_session()
model = build_two_tower_model(df_learn)
model.load_weights(str(OUT / "model.weights.h5"))
test_ds = make_ds(test_df).batch(2048).prefetch(tf.data.AUTOTUNE)
otrec_pred = model.predict(test_ds, verbose=0)["cls"].ravel()
roc, pr = roc_auc_score(test_df.label, otrec_pred), average_precision_score(test_df.label, otrec_pred)
print(f"OTRec held-out (target-disjoint, single split): ROC {roc:.4f} PR {pr:.4f}", flush=True)

# --- OTTree: 5-fold target-grouped OUT-OF-FOLD predictions over ALL pairs ---
# Full OOF coverage (not just the 20% held-out split) so the app's OTTree
# column exists for every pair and OTTree-first sorting is meaningful, while
# every prediction is still made by a model that never saw that pair's target.
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold

feats = ["disease_text", "target_text", "diseaseId"]
params = {"depth": 8, "eval_metric": "AUC", "random_seed": SEED, "verbose": False}
if tf.config.list_physical_devices("GPU"):
    params["task_type"] = "GPU"

ottree_oof = pd.Series(np.nan, index=df_learn.index)
gkf = GroupKFold(n_splits=5)
for fold, (tr_idx, te_idx) in enumerate(
        gkf.split(df_learn, groups=df_learn["targetId"])):
    tr, te = df_learn.iloc[tr_idx], df_learn.iloc[te_idx]
    m_fold = CatBoostClassifier(**params)
    m_fold.fit(Pool(tr[feats], tr["label"], text_features=["disease_text", "target_text"],
                     cat_features=["diseaseId"]))
    ottree_oof.iloc[te_idx] = m_fold.predict_proba(
        Pool(te[feats], text_features=["disease_text", "target_text"], cat_features=["diseaseId"]))[:, 1]
    print(f"  OTTree OOF fold {fold+1}/5 done ({len(te_idx):,} rows)", flush=True)

assert ottree_oof.notna().all()
roc_t = roc_auc_score(df_learn.label, ottree_oof)
pr_t = average_precision_score(df_learn.label, ottree_oof)
print(f"OTTree 5-fold OOF (all pairs): ROC {roc_t:.4f} PR {pr_t:.4f}", flush=True)
ottree_pred = ottree_oof.loc[test_df.index].to_numpy()

held_out = test_df[["diseaseId", "targetId", "label", "score"]].copy()
held_out["otrec_score"] = otrec_pred
held_out["ottree_score"] = ottree_pred
held_out.to_parquet(OUT / "Outputs" / "held_out_preds_2512.parquet", index=False)

# Full-coverage lookup: ot_score + known_label for EVERY df_learn pair (so the
# app's comparison panel keeps its coverage), model predictions only where the
# pair was genuinely held out from training (NaN elsewhere -- honest, and the
# app already renders missing values as "--").
comparison_lookup = df_learn[["diseaseId", "targetId", "score", "label"]].rename(
    columns={"score": "ot_score", "label": "known_label"})
preds = held_out[["diseaseId", "targetId", "otrec_score"]].rename(
    columns={"otrec_score": "otrec_oof_pred"})
comparison_lookup = comparison_lookup.merge(preds, on=["diseaseId", "targetId"], how="left")
comparison_lookup["ottree_pred"] = ottree_oof.to_numpy()  # full OOF coverage
comparison_lookup.to_parquet(OUT / "Outputs" / "comparison_lookup_2512.parquet", index=False)
print(f"saved held_out_preds_2512.parquet ({len(held_out):,} rows) and "
      f"comparison_lookup_2512.parquet ({len(comparison_lookup):,} rows, "
      f"{comparison_lookup.otrec_oof_pred.notna().mean():.0%} with held-out model preds)", flush=True)

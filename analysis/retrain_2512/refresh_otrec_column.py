"""Refresh only the OTRec held-out prediction column for the deployed seed.

The OTTree 5-fold OOF column is seed-independent (CatBoost, fixed seed, fixed
data) and expensive (~25 min); the OTRec column is cheap (~2 min). After a
deployment-seed change, recompute only the latter in both
held_out_preds_2512.parquet and comparison_lookup_2512.parquet.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/d/Research/OpenTargetsTransfer/OTRec/gradio")
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from dl_model_def import build_two_tower_model
from vocab_io import load_vocabularies, apply_vocabularies

R = Path("/mnt/d/Research/OpenTargetsTransfer/retrain_2512")

full = pd.read_parquet(R / "df_learn_2512.parquet")
train_tids, test_tids = train_test_split(
    full["targetId"].unique(), test_size=0.2, random_state=42, shuffle=True,
    stratify=full.drop_duplicates(subset=["targetId"])["label"])
test = full[full["targetId"].isin(test_tids)].copy()

keras.backend.clear_session()
m = build_two_tower_model(full)
apply_vocabularies(m, load_vocabularies(R / "vocabs_2512.json.gz"))
m.load_weights(str(R / "model.weights.h5"))
ds = tf.data.Dataset.from_tensor_slices({
    "query": {"disease_text": test["disease_text"], "diseaseId": test["diseaseId"]},
    "candidate": {"target_text": test["target_text"], "targetId": test["targetId"]}}).batch(2048)
pred = m.predict(ds, verbose=0)["cls"].ravel()
print(f"deployed-seed held-out: ROC {roc_auc_score(test.label, pred):.4f} "
      f"PR {average_precision_score(test.label, pred):.4f}")

held = pd.read_parquet(R / "Outputs" / "held_out_preds_2512.parquet")
key = test[["diseaseId", "targetId"]].reset_index(drop=True)
key["otrec_new"] = pred
held = held.drop(columns=["otrec_score"]).merge(key.rename(columns={"otrec_new": "otrec_score"}),
                                                 on=["diseaseId", "targetId"], how="left")
assert held.otrec_score.notna().all()
held.to_parquet(R / "Outputs" / "held_out_preds_2512.parquet", index=False)

lookup = pd.read_parquet(R / "Outputs" / "comparison_lookup_2512.parquet")
lookup = lookup.drop(columns=["otrec_oof_pred"]).merge(
    key.rename(columns={"otrec_new": "otrec_oof_pred"}), on=["diseaseId", "targetId"], how="left")
lookup.to_parquet(R / "Outputs" / "comparison_lookup_2512.parquet", index=False)
print(f"refreshed otrec columns: held_out ({len(held):,}) + comparison_lookup "
      f"({len(lookup):,}, {lookup.otrec_oof_pred.notna().mean():.0%} coverage)")

"""Per-feature-group ablation, TEMPORAL protocol (2022.02 -> 2025.06).

One rung per invocation (`--rung`), run as a separate process so GPU memory does not
accumulate across rungs. Everything except the named factor is held identical to the
seed-42 temporal reference run:

  seed 42 | 6 epochs | Adam 7e-3 | batch 512 | shuffle 200k | val = 1% of history targets
  (train_test_split random_state=42, so the SAME split in every rung)
  ReduceLROnPlateau(val_cls_loss, 0.2, patience 1) | EarlyStopping(val_cls_loss, patience 2)
  loss_weights {cls 1.0, score 0.1}   [rung R5 sets score -> 0.0]

Feature groups are rebuilt from the component columns of disease_df / target_df. The
reconstruction is TOKEN-identical to the stored *_text_embed strings (verified 200/200),
which is what the count-mode unigram TextVectorization actually consumes.

Nested ladder (each rung is a superset of the previous):
  R1 text-only    disease: name, ExactSynonyms, description
                  target : sym, approvedName, synonyms, functionDescriptions
  R2 +ontology    disease: + dbXRefs, therapeuticAreas, parents, phenotypes
                  target : + targetClass
  R3 +GO/pathway  target : + go, pathways
  R4 +tractability (= FULL) target: + tractability, constraint
  R5 -aux head    R4 features, auxiliary score head weight 0.0

Note: the learned disease-ID embedding is part of the architecture and is present in
every rung, so R1 is "no annotation text", not "no disease identity".
"""
import argparse
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

DIS_TEXT = ["name", "ExactSynonyms", "description"]
DIS_ONTO = ["dbXRefs", "therapeuticAreas", "parents"]          # + phenotypes remainder
TGT_TEXT = ["sym", "approvedName", "synonyms", "functionDescriptions"]
TGT_ONTO = ["targetClass"]
TGT_GOPW = ["go", "pathways"]
TGT_TRACT = ["tractability", "constraint"]

RUNGS = {
    "R1": dict(label="text-only",        dis=DIS_TEXT, dis_pheno=False, tgt=TGT_TEXT, aux=0.1),
    "R2": dict(label="+ontology",        dis=DIS_TEXT + DIS_ONTO, dis_pheno=True,
               tgt=TGT_TEXT + TGT_ONTO, aux=0.1),
    "R3": dict(label="+GO/pathway",      dis=DIS_TEXT + DIS_ONTO, dis_pheno=True,
               tgt=TGT_TEXT + TGT_ONTO + TGT_GOPW, aux=0.1),
    "R4": dict(label="+tractability (full)", dis=DIS_TEXT + DIS_ONTO, dis_pheno=True,
               tgt=TGT_TEXT + TGT_ONTO + TGT_GOPW + TGT_TRACT, aux=0.1),
    "R5": dict(label="full, -auxiliary head", dis=DIS_TEXT + DIS_ONTO, dis_pheno=True,
               tgt=TGT_TEXT + TGT_ONTO + TGT_GOPW + TGT_TRACT, aux=0.0),
}

ap = argparse.ArgumentParser()
ap.add_argument("--rung", required=True, choices=list(RUNGS))
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="ablation_temporal_results.csv")
args = ap.parse_args()
cfg = RUNGS[args.rung]
SEED = args.seed  # val split uses random_state=SEED, matching run_temporal_repeated.py
print(f"=== rung {args.rung}: {cfg['label']} | aux weight {cfg['aux']} | seed {SEED} ===", flush=True)

history_raw = pd.read_parquet(ROOT.parent / "code" / "history_df.parquet")
future_raw = pd.read_parquet(ROOT.parent / "code" / "final_df.parquet")
disease_df = pd.read_parquet(ROOT.parent / "code" / "copy_proc" / "disease_df.parquet")
target_df = pd.read_parquet(ROOT.parent / "code" / "copy_proc" / "target_df.parquet")

# --- rebuild the two text fields for this rung ---
FULL_DIS = DIS_TEXT + DIS_ONTO
def join(row, cols):
    return " ".join(str(row[c]) for c in cols)

pheno = []
for _, r in disease_df.iterrows():
    pre = join(r, FULL_DIS)
    s = r["disease_text_embed"]
    pheno.append(s[len(pre):].strip() if isinstance(s, str) and s.startswith(pre) else "")
disease_df = disease_df.copy()
disease_df["_pheno"] = pheno

dis_txt = disease_df.apply(lambda r: join(r, cfg["dis"]), axis=1)
if cfg["dis_pheno"]:
    dis_txt = dis_txt + " " + disease_df["_pheno"]
disease_df["disease_text_embed"] = dis_txt.str.strip()
target_df = target_df.copy()
target_df["target_text_embed"] = target_df.apply(lambda r: join(r, cfg["tgt"]), axis=1).str.strip()

print("  mean disease_text chars %.0f | mean target_text chars %.0f" % (
    disease_df.disease_text_embed.str.len().mean(), target_df.target_text_embed.str.len().mean()), flush=True)

test_raw = add_historical_score(history_raw, build_temporal_test_set(history_raw, future_raw))
test_raw["score_past"] = test_raw["score_past"].fillna(0.0)
history_df = merge_df_dis_target(history_raw, disease_df, target_df)
test_df = merge_df_dis_target(test_raw, disease_df, target_df)

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
    loss_weights={"cls": 1.0, "score": cfg["aux"]},
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
print(f"RUNG {args.rung} ({cfg['label']}): ROC {roc:.6f}  PR {pr:.6f}", flush=True)

# per-stratum (2022 indication count) for the same rung
ind = history_raw[history_raw.label == 1].groupby("targetId")["diseaseId"].nunique()
res = test_df[["diseaseId", "targetId", "label"]].copy()
res["pred"] = pred
res["n_ind"] = res.targetId.map(ind).fillna(0).astype(int)
strat = {}
for name, mask in [("bin0", res.n_ind == 0), ("bin1", res.n_ind == 1), ("bin2plus", res.n_ind >= 2)]:
    d = res[mask]
    strat[name + "_roc"] = roc_auc_score(d.label, d.pred)
    strat[name + "_pr"] = average_precision_score(d.label, d.pred)
    print(f"    {name}: n={len(d)} pos={int(d.label.sum())} ROC {strat[name+'_roc']:.4f} PR {strat[name+'_pr']:.4f}", flush=True)

row = dict(rung=args.rung, variant=cfg["label"], seed=SEED, aux_weight=cfg["aux"], roc_auc=roc, pr_auc=pr, **strat)
f = OUT / args.out
pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)
res.to_parquet(OUT / f"ablation_preds_{args.rung}_s{SEED}.parquet", index=False)
print("appended ->", f, flush=True)

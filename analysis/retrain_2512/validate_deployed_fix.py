"""Validate the vocabulary fix on the EXISTING deployed 25.06 model.

Three serving configurations, same HF weights, same evaluation sample:
  broken : vocab re-adapted from the shipped df_learn_sub (today's live Space)
  fixed  : vocab re-adapted from df_learn_sub, then apply_vocabularies(saved)
  oracle : vocab adapted from the full 25.06 training frame (training-time)

Metrics beyond ROC-AUC: PR-AUC, precision/recall/F1 at the paper's 0.65
threshold, per-disease Hit@1/5/10 + MRR (worst-rank ties), and agreement of
top-1 recommendations with the oracle. Also writes the saved-vocab artifact
for the 25.06 model (vocabs_2506.json.gz).

Evaluation rows are the full 25.06 frame's held-out 20% target split (seed 42,
same recipe as the training notebook) so labels are genuinely held out.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/mnt/d/Research/OpenTargetsTransfer/OTRec/gradio")
import tensorflow as tf
from tensorflow import keras
from huggingface_hub import hf_hub_download

from dl_model_def import build_two_tower_model
from vocab_io import extract_vocabularies, save_vocabularies, load_vocabularies, apply_vocabularies

REPO = Path("/mnt/d/Research/OpenTargetsTransfer")
R = REPO / "retrain_2512"

WEIGHTS = hf_hub_download(repo_id="GrimSqueaker/OTRec", filename="model.weights.h5")
full = pd.read_parquet(REPO / "code" / "copy_proc" / "df_learn.parquet")
sub = pd.read_parquet(REPO / "OTRec" / "gradio" / "data" / "proc" / "df_learn_sub.parquet")

# Held-out targets, exactly the training notebook's split recipe.
_, test_tids = train_test_split(
    full["targetId"].unique(), test_size=0.2, random_state=42, shuffle=True,
    stratify=full.drop_duplicates(subset=["targetId"])["label"],
)
test_df = full[full["targetId"].isin(test_tids)].copy()
print(f"eval rows {len(test_df):,}, positives {int(test_df.label.sum()):,} ({test_df.label.mean():.2%})",
      flush=True)


def _score(model, sample):
    ds = tf.data.Dataset.from_tensor_slices({
        "query": {"disease_text": sample["disease_text"], "diseaseId": sample["diseaseId"]},
        "candidate": {"target_text": sample["target_text"], "targetId": sample["targetId"]},
    }).batch(1024)
    return model.predict(ds, verbose=0)["cls"].ravel()


def per_disease_rank(df, col):
    hits = {1: [], 5: [], 10: []}
    mrrs = []
    for _, g in df[df.groupby("diseaseId")["label"].transform("sum") > 0].groupby("diseaseId"):
        s, y = g[col].to_numpy(), g.label.to_numpy()
        order = np.argsort(-s, kind="stable")
        srt, ysrt = s[order], y[order]
        worst = np.empty(len(s), dtype=int)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and srt[j + 1] == srt[i]:
                j += 1
            worst[i:j + 1] = j + 1
            i = j + 1
        best = worst[ysrt == 1].min()
        mrrs.append(1.0 / best)
        for k in hits:
            hits[k].append(1.0 if best <= k else 0.0)
    return {f"Hit@{k}": float(np.mean(v)) for k, v in hits.items()} | {"MRR": float(np.mean(mrrs))}


def metric_row(y, p, name):
    yhat = (p >= 0.65).astype(int)
    return {
        "config": name,
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "P@0.65": precision_score(y, yhat, zero_division=0),
        "R@0.65": recall_score(y, yhat, zero_division=0),
        "F1@0.65": f1_score(y, yhat, zero_division=0),
    }


configs = {}

keras.backend.clear_session()
oracle = build_two_tower_model(full)
oracle.load_weights(WEIGHTS)
vocabs = extract_vocabularies(oracle)
save_vocabularies(vocabs, R / "vocabs_2506.json.gz")
print(f"saved 25.06 vocabularies -> {R/'vocabs_2506.json.gz'} "
      f"({(R/'vocabs_2506.json.gz').stat().st_size/1e6:.2f} MB)", flush=True)
configs["oracle (full-frame adapt)"] = _score(oracle, test_df)

keras.backend.clear_session()
broken = build_two_tower_model(sub)
broken.load_weights(WEIGHTS)
configs["broken (live Space today)"] = _score(broken, test_df)

keras.backend.clear_session()
fixed = build_two_tower_model(sub)
apply_vocabularies(fixed, load_vocabularies(R / "vocabs_2506.json.gz"))
fixed.load_weights(WEIGHTS)
configs["fixed (sub + saved vocab)"] = _score(fixed, test_df)

y = test_df["label"].to_numpy()
rows = [metric_row(y, p, name) for name, p in configs.items()]
print("\n" + pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

work = test_df[["diseaseId", "label"]].copy()
for name, p in configs.items():
    work[name] = p
print("\nPer-disease ranking (diseases with >=1 positive in held-out rows):")
for name in configs:
    r = per_disease_rank(work, name)
    print(f"  {name:<28} Hit@1 {r['Hit@1']:.3f}  Hit@5 {r['Hit@5']:.3f}  "
          f"Hit@10 {r['Hit@10']:.3f}  MRR {r['MRR']:.3f}")

fo = np.corrcoef(configs["fixed (sub + saved vocab)"], configs["oracle (full-frame adapt)"])[0, 1]
delta = np.max(np.abs(configs["fixed (sub + saved vocab)"] - configs["oracle (full-frame adapt)"]))
print(f"\nfixed vs oracle: corr {fo:.6f}, max |delta| {delta:.8f}")
assert delta < 1e-5, "fixed serving does not reproduce the oracle"
print("PASS: fix validated on the deployed 25.06 model")

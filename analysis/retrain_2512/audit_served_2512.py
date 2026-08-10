"""Serve audit for the 25.12 package, exactly as the Space will run it.

Uses the packaged artifacts (gradio_artifacts/) + the 25.12 weights through the
app's load path (build from packaged df_learn_sub, apply packaged vocabs).
Checks: known-positive recovery, paper-example diseases, and extreme bias
(how often pseudogene hubs appear in served top-10s).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/d/Research/OpenTargetsTransfer/OTRec/gradio")
import tensorflow as tf
from tensorflow import keras

from dl_model_def import build_two_tower_model
from vocab_io import load_vocabularies, apply_vocabularies

REPO = Path("/mnt/d/Research/OpenTargetsTransfer")
R = REPO / "retrain_2512"
PKG = R / "gradio_artifacts"
PSEUDO = {"GUCY1B2", "CLCA3P", "GLRA4", "TUBB8", "TUBB8B"}

sub = pd.read_parquet(PKG / "df_learn_sub.parquet")
disease_df = pd.read_parquet(PKG / "disease_df.parquet")
target_df = pd.read_parquet(PKG / "target_df.parquet")
full = pd.read_parquet(R / "df_learn_2512.parquet")
positives = set(zip(full.query("label==1").diseaseId, full.query("label==1").targetId))

keras.backend.clear_session()
m = build_two_tower_model(sub)
apply_vocabularies(m, load_vocabularies(PKG / "vocabs.json.gz"))
m.load_weights(str(R / "model.weights.h5"))

t_text = target_df["target_text"].astype(str).to_numpy()
t_ids = target_df["targetId"].astype(str).to_numpy()
t_sym = target_df["approvedSymbol"].astype(str).to_numpy()
embs = []
for i in range(0, len(t_text), 1024):
    embs.append(tf.nn.l2_normalize(m.encode_k(t_text[i:i+1024], t_ids[i:i+1024]), axis=1, epsilon=1e-16))
cand_embs = tf.concat(embs, axis=0)
print(f"candidate universe: {len(t_ids):,} targets")

d_lookup = disease_df.set_index("diseaseId")


def top10(did):
    if did not in d_lookup.index:
        return None
    row = d_lookup.loc[[did]]
    q = tf.nn.l2_normalize(m.encode_q(row["disease_text"].to_numpy(), np.array([did])), axis=1)
    sims = tf.matmul(q, cand_embs, transpose_b=True).numpy().ravel()
    probs = m.cls_head(tf.constant(sims.reshape(-1, 1))).numpy().ravel()
    order = np.argsort(-probs)[:10]
    return [(t_sym[i], float(probs[i]), (did, t_ids[i]) in positives) for i in order]


print("\n--- paper-example diseases (served top-10, 25.12 model) ---")
for did, desc in [("MONDO_0100039", "CDKL5 disorder"), ("MONDO_0000515", "bone chondrosarcoma")]:
    r = top10(did)
    if r is None:
        print(f"  {desc}: not in disease frame")
        continue
    txt = " ".join(f"{s}({p:.2f}{',K' if k else ''}{',PSEUDO' if s in PSEUDO else ''})" for s, p, k in r)
    print(f"  {desc}: {txt}")

rng = np.random.default_rng(0)
pos_dis = [d for d in full.query("label==1").diseaseId.unique() if d in d_lookup.index]
sample = rng.choice(pos_dis, size=200, replace=False)
recov, pseudo_hits, pseudo_top1 = [], [], []
for did in sample:
    r = top10(did)
    recov.append(any(k for _, _, k in r))
    pseudo_hits.append(sum(1 for s, _, _ in r if s in PSEUDO))
    pseudo_top1.append(r[0][0] in PSEUDO)
print(f"\nknown-positive in served top-10 (200 positive-bearing diseases): {np.mean(recov):.3f}")
print(f"pseudogene-hub rows per served top-10: mean {np.mean(pseudo_hits):.2f} "
      f"(max {max(pseudo_hits)}); pseudogene at rank 1: {np.mean(pseudo_top1):.1%}")
print("(25.06 fixed-serving reference: recovery 0.750)")

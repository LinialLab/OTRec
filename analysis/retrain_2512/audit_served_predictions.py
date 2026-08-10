"""What does the Space actually show? Fixed vs broken serving, per disease.

For a panel of diseases (well-characterised, paper examples, orphan), rank ALL
candidates in the app's own target_df with (a) today's broken serving and
(b) fixed serving, and report top-10 with known-positive markers and pseudogene
flags. Sanity = known positives should rank top for well-characterised
diseases; bias = pseudogene hubs appearing across unrelated diseases.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/d/Research/OpenTargetsTransfer/OTRec/gradio")
import tensorflow as tf
from tensorflow import keras
from huggingface_hub import hf_hub_download

from dl_model_def import build_two_tower_model
from vocab_io import load_vocabularies, apply_vocabularies

REPO = Path("/mnt/d/Research/OpenTargetsTransfer")
R = REPO / "retrain_2512"

DISEASES = [
    ("EFO_0003885", "multiple sclerosis (well-characterised)"),
    ("EFO_0001360", "type 2 diabetes (well-characterised)"),
    ("MONDO_0100039", "CDKL5 disorder (paper example)"),
    ("MONDO_0000515", "bone chondrosarcoma (paper example, orphan)"),
]
PSEUDO = {"GUCY1B2", "CLCA3P", "GLRA4", "TUBB8", "TUBB8B"}

sub = pd.read_parquet(REPO / "OTRec" / "gradio" / "data" / "proc" / "df_learn_sub.parquet")
disease_df = pd.read_parquet(REPO / "OTRec" / "gradio" / "data" / "proc" / "disease_df.parquet")
target_df = pd.read_parquet(REPO / "OTRec" / "gradio" / "data" / "proc" / "target_df.parquet")
full = pd.read_parquet(REPO / "code" / "copy_proc" / "df_learn.parquet")
positives = set(zip(full.query("label==1").diseaseId, full.query("label==1").targetId))

WEIGHTS = hf_hub_download(repo_id="GrimSqueaker/OTRec", filename="model.weights.h5")

t_text = target_df["target_text"].astype(str).to_numpy()
t_ids = target_df["targetId"].astype(str).to_numpy()
t_sym = target_df["approvedSymbol"].astype(str).to_numpy()


def build(fixed: bool):
    keras.backend.clear_session()
    m = build_two_tower_model(sub)
    if fixed:
        apply_vocabularies(m, load_vocabularies(R / "vocabs_2506.json.gz"))
    m.load_weights(WEIGHTS)
    embs = []
    for i in range(0, len(t_text), 1024):
        embs.append(tf.nn.l2_normalize(m.encode_k(t_text[i:i+1024], t_ids[i:i+1024]), axis=1, epsilon=1e-16))
    return m, tf.concat(embs, axis=0)


def top10(m, cand_embs, did):
    row = disease_df.loc[disease_df.diseaseId == did]
    if row.empty:
        return None
    q = tf.nn.l2_normalize(m.encode_q(row["disease_text"].to_numpy(), row["diseaseId"].to_numpy()), axis=1)
    sims = tf.matmul(q, cand_embs, transpose_b=True).numpy().ravel()
    probs = m.cls_head(tf.constant(sims.reshape(-1, 1))).numpy().ravel()
    order = np.argsort(-probs)[:10]
    out = []
    for idx in order:
        sym = t_sym[idx]
        marks = []
        if (did, t_ids[idx]) in positives:
            marks.append("KNOWN+")
        if sym in PSEUDO:
            marks.append("PSEUDO/HUB")
        out.append(f"{sym}({probs[idx]:.2f}{',' + '+'.join(marks) if marks else ''})")
    return out


for fixed, label in [(False, "BROKEN (live today)"), (True, "FIXED")]:
    m, cand_embs = build(fixed)
    print(f"\n================ {label} ================")
    for did, desc in DISEASES:
        r = top10(m, cand_embs, did)
        print(f"\n  {desc} [{did}]")
        print("   ", " ".join(r) if r else "disease not in app frame")

# Known-positive recovery rate across a random disease sample (sanity at scale)
rng = np.random.default_rng(0)
pos_diseases = full.query("label==1").diseaseId.unique()
sample_dids = rng.choice(pos_diseases, size=200, replace=False)
for fixed, label in [(False, "BROKEN"), (True, "FIXED")]:
    m, cand_embs = build(fixed)
    hits = []
    for did in sample_dids:
        r10 = top10(m, cand_embs, did)
        if r10 is None:
            continue
        hits.append(1.0 if any("KNOWN+" in x for x in r10) else 0.0)
    print(f"\n{label}: fraction of 200 positive-bearing diseases with >=1 known positive in top-10: "
          f"{np.mean(hits):.3f}")

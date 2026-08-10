"""Precompute per-gene top-K predicted diseases for ALL candidate targets.

One-time chunked matrix product over the packaged embeddings (17,073 targets x
46,960 diseases), calibrated with the deployed cls_head. Feeds the gene-family
overlap analysis so it never has to touch TensorFlow again.

Output: Outputs/gene_topk.npz
  topk_idx    (n_targets, 200) int32   disease row indices, best first
  topk_score  (n_targets, 200) float32 calibrated probabilities
  n_ge_065    (n_targets,)     int32   count of diseases with score >= 0.65
  w, b        cls_head calibration scalars
Run: python3 retrain_2512/precompute_gene_topk.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/d/Research/OpenTargetsTransfer/OTRec/gradio")
from tensorflow import keras

from dl_model_def import build_two_tower_model
from vocab_io import load_vocabularies, apply_vocabularies

R = Path("/mnt/d/Research/OpenTargetsTransfer/retrain_2512")
PKG = R / "gradio_artifacts"
K = 200

emb = np.load(PKG / "embeddings.npz", allow_pickle=False)
cand, dis = emb["candidate_embs"], emb["disease_embs"]

keras.backend.clear_session()
model = build_two_tower_model(pd.read_parquet(PKG / "df_learn_sub.parquet"))
apply_vocabularies(model, load_vocabularies(PKG / "vocabs.json.gz"))
model.load_weights(str(R / "model.weights.h5"))
w, b = (float(x) for x in np.concatenate([a.ravel() for a in model.cls_head.get_weights()]))
print(f"cand {cand.shape} dis {dis.shape} w={w:+.2f} b={b:+.2f}")

n = cand.shape[0]
topk_idx = np.empty((n, K), np.int32)
topk_score = np.empty((n, K), np.float32)
n_ge = np.empty(n, np.int32)
thr_cos = (np.log(0.65 / 0.35) - b) / w  # prob >= 0.65 in cosine units (w > 0)
assert w > 0, "top-k by cosine assumes positive calibration slope"
for i in range(0, n, 2048):
    cos = cand[i:i + 2048] @ dis.T
    part = np.argpartition(-cos, K, axis=1)[:, :K]
    rows = np.arange(cos.shape[0])[:, None]
    order = np.argsort(-cos[rows, part], axis=1)
    idx = part[rows, order]
    topk_idx[i:i + 2048] = idx
    topk_score[i:i + 2048] = 1.0 / (1.0 + np.exp(-(cos[rows, idx] * w + b)))
    n_ge[i:i + 2048] = (cos >= thr_cos).sum(axis=1)
    print(f"  {min(i + 2048, n)}/{n}", flush=True)

np.savez_compressed(R / "Outputs" / "gene_topk.npz",
                    topk_idx=topk_idx, topk_score=topk_score, n_ge_065=n_ge,
                    w=np.float64(w), b=np.float64(b))
print(f"saved gene_topk.npz; median diseases >=0.65 per gene: {np.median(n_ge):.0f}, "
      f"max {n_ge.max()}, zero-count genes {(n_ge == 0).sum()}")

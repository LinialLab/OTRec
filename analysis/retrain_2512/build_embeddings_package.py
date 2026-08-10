"""Precompute candidate + disease embeddings for the Space package.

Computed with the exact serving path (packaged df_learn_sub + vocabs + deployed
weights), stamped with the weights md5 so app.py's loader can refuse stale
files and fall back to live computation. Cuts the free-tier cold start's
~1-minute embedding precompute to a file read.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/d/Research/OpenTargetsTransfer/OTRec/gradio")
import tensorflow as tf
from tensorflow import keras

from dl_model_def import build_two_tower_model
from vocab_io import load_vocabularies, apply_vocabularies

R = Path("/mnt/d/Research/OpenTargetsTransfer/retrain_2512")
PKG = R / "gradio_artifacts"

sub = pd.read_parquet(PKG / "df_learn_sub.parquet")
target_df = pd.read_parquet(PKG / "target_df.parquet")
disease_df = pd.read_parquet(PKG / "disease_df.parquet")

keras.backend.clear_session()
model = build_two_tower_model(sub)
apply_vocabularies(model, load_vocabularies(PKG / "vocabs.json.gz"))
weights_path = R / "model.weights.h5"
model.load_weights(str(weights_path))
weights_md5 = hashlib.md5(weights_path.read_bytes()).hexdigest()


def embed(texts, ids, encode):
    out = []
    for i in range(0, len(texts), 1024):
        out.append(tf.nn.l2_normalize(encode(texts[i:i+1024], ids[i:i+1024]), axis=1))
    return tf.concat(out, axis=0).numpy().astype(np.float32)


cand = embed(target_df["target_text"].astype(str).to_numpy(),
             target_df["targetId"].astype(str).to_numpy(), model.encode_k)
dis = embed(disease_df["disease_text"].astype(str).to_numpy(),
            disease_df["diseaseId"].astype(str).to_numpy(), model.encode_q)

np.savez_compressed(PKG / "embeddings.npz",
                    candidate_embs=cand, disease_embs=dis, weights_md5=weights_md5)
size_mb = (PKG / "embeddings.npz").stat().st_size / 1e6
print(f"saved embeddings.npz: candidates {cand.shape}, diseases {dis.shape}, "
      f"{size_mb:.1f} MB, weights_md5={weights_md5[:10]}...")

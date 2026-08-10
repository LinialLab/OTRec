"""Frozen BioClinical-ModernBERT + MLP on the native 23.06 -> 25.12 split.

Runs the repo's own OTRec/baselines/run_frozen_encoder_temporal.py unmodified,
pointed at the native frames via its existing CLI flags. The only addition is a
version shim: sentence-transformers 5.x renamed models.Transformer's
`model_kwargs` argument to `model_args`, which is what blocked this baseline
previously. We patch build_sentence_transformer in-process; no repo file is
modified.

Must run under an environment with sentence-transformers >= 4 and torch+CUDA
(e.g. ~/anaconda3/envs/ag/bin/python).
"""
import inspect
import sys
from pathlib import Path

ROOT = Path("/mnt/d/Research/OpenTargetsTransfer/OTRec")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines"))

import torch
from sentence_transformers import SentenceTransformer, models

import run_frozen_encoder_mlp as rfem
import run_frozen_encoder_temporal as rfet


def build_sentence_transformer(model_name: str, max_seq_length: int) -> SentenceTransformer:
    """Version-tolerant rebuild of the repo helper (model_kwargs -> model_args in ST 5.x)."""
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": "sdpa",
    }
    params = inspect.signature(models.Transformer.__init__).parameters
    key = "model_kwargs" if "model_kwargs" in params else "model_args"
    transformer = models.Transformer(model_name, max_seq_length=max_seq_length, **{key: kwargs})
    pooling = models.Pooling(transformer.get_embedding_dimension(), pooling_mode="mean")
    return SentenceTransformer(modules=[transformer, pooling, models.Normalize()])


rfem.build_sentence_transformer = build_sentence_transformer
rfet.build_sentence_transformer = build_sentence_transformer

if __name__ == "__main__":
    OUT = Path("/mnt/d/Research/OpenTargetsTransfer/rebuttal_scratch/native_2306")
    sys.argv = [
        "run_frozen_encoder_temporal.py",
        "--history", str(OUT / "native2306_train_frame.parquet"),
        "--future", str(OUT / "native2306_eval_frame.parquet"),
        "--disease", str(OUT / "native2306_disease_df.parquet"),
        "--target", str(OUT / "native2306_target_df.parquet"),
        "--output-dir", str(OUT / "modernbert"),
    ] + sys.argv[1:]
    rfet.main()

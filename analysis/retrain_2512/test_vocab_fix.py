"""Vocabulary-fix tests -- RED before GREEN.

Seam: (any bootstrap frame, saved vocabularies, weights) -> scoring model whose
predictions are identical to the training-time model. This is the fix for the
silent-permutation defect: the Space re-adapts TextVectorization from a
deduplicated frame whose token frequencies (hence vocab ORDER) differ from
training, so identically-shaped weights are applied to permuted features.

Oracle: a model built by adapt on the FULL training frame with the same
weights -- the configuration verified to reproduce training-time behaviour
(ROC 0.9855 for the deployed 25.06 pair; in-session equality for 25.12).

The stress case deliberately bootstraps from the DEDUP COVER -- the exact
frame shape that corrupts predictions today -- and requires that after
apply_vocabularies() the predictions match the oracle anyway.

Run: python3 retrain_2512/test_vocab_fix.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/d/Research/OpenTargetsTransfer/OTRec/gradio")
import tensorflow as tf
from tensorflow import keras

from dl_model_def import build_two_tower_model
from vocab_io import extract_vocabularies, save_vocabularies, load_vocabularies, apply_vocabularies

R = Path("/mnt/d/Research/OpenTargetsTransfer/retrain_2512")
WEIGHTS = R / "model.weights.h5"


def _make_dedup_cover(full: pd.DataFrame) -> pd.DataFrame:
    by_disease = full.drop_duplicates(subset=["diseaseId"], keep="first")
    by_target = full.drop_duplicates(subset=["targetId"], keep="first")
    return (pd.concat([by_disease, by_target], ignore_index=True)
            .drop_duplicates(subset=["diseaseId", "targetId"]).reset_index(drop=True))


def _score(model, sample: pd.DataFrame) -> np.ndarray:
    ds = tf.data.Dataset.from_tensor_slices({
        "query": {"disease_text": sample["disease_text"], "diseaseId": sample["diseaseId"]},
        "candidate": {"target_text": sample["target_text"], "targetId": sample["targetId"]},
    }).batch(512)
    return model.predict(ds, verbose=0)["cls"].ravel()


def test_roundtrip_and_dedup_bootstrap_reproduce_oracle():
    full = pd.read_parquet(R / "df_learn_2512.parquet")
    sample = full.sample(n=4096, random_state=0)

    # Oracle: adapt on the full frame (training-time configuration).
    keras.backend.clear_session()
    oracle = build_two_tower_model(full)
    oracle.load_weights(str(WEIGHTS))
    preds_oracle = _score(oracle, sample)
    vocabs = extract_vocabularies(oracle)
    assert set(vocabs) == {"disease_text", "target_text", "diseaseId"}, sorted(vocabs)

    with tempfile.TemporaryDirectory() as td:
        vpath = Path(td) / "vocabs.json.gz"
        save_vocabularies(vocabs, vpath)
        loaded = load_vocabularies(vpath)
        for k in vocabs:
            assert list(loaded[k]) == list(vocabs[k]), f"vocabulary {k} did not roundtrip"

        # Stress case: bootstrap from the dedup cover (the corrupting frame),
        # then apply the saved vocabularies and load the same weights.
        cover = _make_dedup_cover(full)
        keras.backend.clear_session()
        fixed = build_two_tower_model(cover)
        apply_vocabularies(fixed, loaded)
        fixed.load_weights(str(WEIGHTS))
        preds_fixed = _score(fixed, sample)

    max_abs = float(np.max(np.abs(preds_oracle - preds_fixed)))
    corr = float(np.corrcoef(preds_oracle, preds_fixed)[0, 1])
    print(f"dedup-bootstrap + saved vocab: max |delta| = {max_abs:.8f}, corr = {corr:.6f}")
    assert max_abs < 1e-5, f"fix does not reproduce oracle predictions (max delta {max_abs})"
    print("PASS: saved vocabularies make the dedup bootstrap exactly reproduce the oracle")


if __name__ == "__main__":
    test_roundtrip_and_dedup_bootstrap_reproduce_oracle()

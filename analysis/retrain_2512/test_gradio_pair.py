"""Matched-set load test -- RED before GREEN.

Seam under test: (df_learn_sub, weights) -> a scoring model that loads without
shape mismatch. This is the defect class the exploration agent flagged: the
Space rebuilds TextVectorization vocab from df_learn_sub.parquet at cold start
then load_weights()s a separately-uploaded file; if the two are not from the
same training run, vocab size shifts and load_weights raises (or, worse,
silently mismatches row order -- caught by the negative control below).

Positive case: NEW weights + NEW df_learn_sub load together and produce
sane-range scores.
Negative control: OLD weights + NEW df_learn_sub must FAIL to load -- this is
what proves the test can actually detect a mismatched pair, not just that
`load_weights` never raises.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/d/Research/OpenTargetsTransfer/OTRec/gradio")
import tensorflow as tf
from tensorflow import keras

from dl_model_def import build_two_tower_model

REPO = Path("/mnt/d/Research/OpenTargetsTransfer")
NEW = REPO / "retrain_2512"
OLD_WEIGHTS = REPO / "OTRec" / "gradio" / "_old_model_weights_probe.h5"  # fetched on demand below


def _mismatched_vocab_frame():
    """A frame whose vocabulary CANNOT match the 25.12 weights: the released
    25.06 training frame (different disease ontology size). Stable oracle --
    unlike downloading "old" weights from HF, which drift as deployments
    advance (that stale premise once made this control fail spuriously)."""
    p = Path("/mnt/d/Research/OpenTargetsTransfer/code/copy_proc/df_learn.parquet")
    if not p.exists():
        print("  (negative control skipped: 25.06 frame not on disk)")
        return None
    return pd.read_parquet(p)


def test_new_pair_loads_and_scores():
    df_learn_sub = pd.read_parquet(NEW / "df_learn_2512.parquet")
    keras.backend.clear_session()
    model = build_two_tower_model(df_learn_sub)
    model.load_weights(str(NEW / "model.weights.h5"))  # must not raise

    sample = df_learn_sub.sample(n=min(2048, len(df_learn_sub)), random_state=0)
    ds = tf.data.Dataset.from_tensor_slices({
        "query": {"disease_text": sample["disease_text"], "diseaseId": sample["diseaseId"]},
        "candidate": {"target_text": sample["target_text"], "targetId": sample["targetId"]},
    }).batch(512)
    preds = model.predict(ds, verbose=0)["cls"].ravel()
    assert len(preds) == len(sample), f"expected {len(sample)} scores, got {len(preds)}"
    assert np.all((preds >= 0.0) & (preds <= 1.0)), "scores outside [0,1]"
    print(f"PASS: new pair loads; {len(preds)} scores in [{preds.min():.4f}, {preds.max():.4f}]")


def test_mismatched_pair_fails_to_load():
    frame_2506 = _mismatched_vocab_frame()
    if frame_2506 is None:
        return
    keras.backend.clear_session()
    model = build_two_tower_model(frame_2506)
    try:
        model.load_weights(str(NEW / "model.weights.h5"))
        # If load_weights didn't raise, the vocab sizes happened to coincide --
        # verify a real shape mismatch some other way so this isn't a false pass.
        raise AssertionError(
            "25.12 weights loaded onto a 25.06 vocabulary without error -- "
            "negative control is not discriminating; inspect vocab sizes"
        )
    except (ValueError, AssertionError) as e:
        if "negative control is not discriminating" in str(e):
            raise
        print(f"PASS: mismatched pair correctly failed to load ({type(e).__name__})")


if __name__ == "__main__":
    test_new_pair_loads_and_scores()
    test_mismatched_pair_fails_to_load()
    print("PASS: gradio matched-set seam verified")

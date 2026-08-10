"""Does the packaged df_learn_sub reproduce the full frame's model?

Seam: (df_learn_sub, weights) -> predictions, versus (full df_learn, weights)
-> predictions. The Space ships a deduplicated *cover* of df_learn (one row per
unique disease / target) rather than the full frame, because the full frame is
1.28 GB. The model's vocabulary is re-adapted from whatever frame is shipped.

The risk this test exists to settle: TextVectorization builds its vocabulary by
token frequency. Deduplicating rows changes those frequencies, which can
reorder the vocabulary, which permutes feature indices -- and the trained dense
weights would then be applied to the wrong features, silently producing garbage
while still "loading" without error.

Oracle: the full-frame model's own predictions. If the shipped subset is a
faithful stand-in, predictions must match to numerical tolerance.

Run: python3 retrain_2512/test_df_learn_sub_equivalence.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/d/Research/OpenTargetsTransfer/OTRec/gradio")
import tensorflow as tf
from tensorflow import keras

from dl_model_def import build_two_tower_model

R = Path("/mnt/d/Research/OpenTargetsTransfer/retrain_2512")
WEIGHTS = R / "model.weights.h5"


def _score(frame_for_vocab: pd.DataFrame, sample: pd.DataFrame) -> np.ndarray:
    keras.backend.clear_session()
    model = build_two_tower_model(frame_for_vocab)
    model.load_weights(str(WEIGHTS))
    ds = tf.data.Dataset.from_tensor_slices({
        "query": {"disease_text": sample["disease_text"], "diseaseId": sample["diseaseId"]},
        "candidate": {"target_text": sample["target_text"], "targetId": sample["targetId"]},
    }).batch(512)
    return model.predict(ds, verbose=0)["cls"].ravel()


def make_dedup_cover(full: pd.DataFrame) -> pd.DataFrame:
    """One row per unique diseaseId plus one per unique targetId -- the
    packaging strategy the deployed Space uses (13,856 rows covering all
    12,337 diseases and 1,522 targets of the 25.06 frame)."""
    by_disease = full.drop_duplicates(subset=["diseaseId"], keep="first")
    by_target = full.drop_duplicates(subset=["targetId"], keep="first")
    cover = pd.concat([by_disease, by_target], ignore_index=True)
    return cover.drop_duplicates(subset=["diseaseId", "targetId"], keep="first").reset_index(drop=True)


def test_dedup_cover_reproduces_full_frame_predictions():
    """The decisive test: does the SHIPPABLE (small) frame preserve predictions?"""
    full = pd.read_parquet(R / "df_learn_2512.parquet")
    cover = make_dedup_cover(full)
    print(f"full {full.shape} -> dedup cover {cover.shape} "
          f"(diseases {cover.diseaseId.nunique():,}/{full.diseaseId.nunique():,}, "
          f"targets {cover.targetId.nunique():,}/{full.targetId.nunique():,})")

    sample = full.sample(n=4096, random_state=0)
    preds_full = _score(full, sample)
    preds_cover = _score(cover, sample)
    max_abs = float(np.max(np.abs(preds_full - preds_cover)))
    corr = float(np.corrcoef(preds_full, preds_cover)[0, 1])
    print(f"dedup cover: max |delta| = {max_abs:.6f}, correlation = {corr:.6f}")
    assert max_abs < 1e-4, (
        f"dedup cover changes predictions (max delta {max_abs:.6f}) -- "
        "re-adapted vocabulary differs from the training vocabulary"
    )
    print("PASS: dedup cover is a safe stand-in for the full frame")


def test_subset_reproduces_full_frame_predictions():
    full = pd.read_parquet(R / "df_learn_2512.parquet")
    subset = pd.read_parquet(R / "df_learn_sub_2512.parquet")
    print(f"full {full.shape} vs packaged subset {subset.shape}")

    sample = full.sample(n=4096, random_state=0)
    preds_full = _score(full, sample)
    preds_subset = _score(subset, sample)

    max_abs = float(np.max(np.abs(preds_full - preds_subset)))
    corr = float(np.corrcoef(preds_full, preds_subset)[0, 1])
    print(f"max |delta| = {max_abs:.6f}, correlation = {corr:.6f}")
    assert max_abs < 1e-4, (
        f"packaged subset changes predictions (max delta {max_abs:.6f}); "
        "the re-adapted vocabulary does not match the training vocabulary"
    )
    print("PASS: packaged frame reproduces full-frame predictions")


if __name__ == "__main__":
    test_dedup_cover_reproduces_full_frame_predictions()
    test_subset_reproduces_full_frame_predictions()

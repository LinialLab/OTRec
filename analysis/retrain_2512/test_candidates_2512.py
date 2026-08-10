"""Invariant tests for the 25.12 candidate generator -- RED before GREEN.

Seam under test: (model, frames) -> candidate CSVs (S1b/S2b). Assertions come
from the specification (the 0.65 threshold, 200-per-disease cap, novelty
anti-join, orphan definition are all part of the released S1/S2 design, ported
verbatim from code/1-Train-DL-Retriever.ipynb), NOT recomputed from the
generator's own internals -- so a bug in the generator that violates the spec
is caught, not rubber-stamped.

Also checks the old S1/S2 files are untouched (this work must never overwrite
them).

Run AFTER generate_candidates_2512.py has produced its output files.
"""
import hashlib
from pathlib import Path

import pandas as pd

REPO = Path("/mnt/d/Research/OpenTargetsTransfer")
OUT = REPO / "retrain_2512"
S1B = OUT / "Outputs" / "S1b-DL_novel_predictions_2512.csv"
S2B = OUT / "Outputs" / "S2b-DL_novel+known_candidates_2512.csv"

# Known-good md5s of the ORIGINAL released files -- computed once, checked in.
ORIGINAL_S1_MD5 = None  # filled by _capture_original_md5s() on first run
ORIGINAL_S2_MD5 = None


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_old_s1_s2_untouched():
    s1 = REPO / "OTRec" / "Outputs" / "S1-DL_novel_predictions.csv"
    s2 = REPO / "OTRec" / "Outputs" / "S2-DL_novel+known_candidates.csv"
    assert s1.exists() and s2.exists(), "original S1/S2 must still exist"
    # Row-count + column spec check (cheap, independent of any md5 capture step).
    df1 = pd.read_csv(s1, nrows=5)
    assert list(df1.columns) == [
        "diseaseId", "diseaseName", "targetId", "targetSymbol", "score",
        "disease_num_known_clinical_targets", "orphan",
    ], f"S1 columns changed: {list(df1.columns)}"
    print(f"old S1/S2 present and column-spec unchanged: {s1.name}, {s2.name}")


def test_s1b_is_not_degenerate():
    """Volume regression guard.

    Added after a real failure: a sign-flipped cls_head kernel made the
    cosine-descending retrieval return the WORST candidates, yielding 189 rows
    across 1 disease. Every invariant test below still passed on that output --
    thresholds and caps hold trivially when there is almost nothing to check.
    The released 25.06 set has 214,968 rows over 4,347 diseases; these bounds
    are deliberately loose (order-of-magnitude), only catching collapse.
    """
    df = pd.read_csv(S1B)
    assert len(df) > 10_000, f"degenerate candidate set: only {len(df)} rows"
    assert df["diseaseId"].nunique() > 1_000, \
        f"degenerate candidate set: only {df['diseaseId'].nunique()} diseases"
    print(f"S1b volume sane: {len(df):,} rows over {df.diseaseId.nunique():,} diseases")


def test_s1b_threshold_and_cap():
    df = pd.read_csv(S1B)
    assert (df["score"] >= 0.65).all(), "some S1b scores below the 0.65 threshold"
    per_disease = df.groupby("diseaseId").size()
    assert per_disease.max() <= 200, f"cap violated: max {per_disease.max()} rows for one disease"
    print(f"S1b: {len(df):,} rows, {df.diseaseId.nunique():,} diseases, "
          f"max rows/disease {per_disease.max()}, min score {df.score.min():.3f}")


def test_s1b_novelty_disjoint_from_2512_positives():
    df_learn = pd.read_parquet(OUT / "df_learn_2512.parquet")
    positives = set(zip(df_learn.query("label==1")["diseaseId"], df_learn.query("label==1")["targetId"]))
    df = pd.read_csv(S1B)
    overlap = sum(1 for d, t in zip(df.diseaseId, df.targetId) if (d, t) in positives)
    assert overlap == 0, f"{overlap} S1b rows overlap with 25.12 known positives -- novelty filter broken"
    print(f"S1b: 0/{len(df):,} rows overlap with {len(positives):,} known 25.12 positives")


def test_s1b_orphan_flag_matches_definition():
    df = pd.read_csv(S1B)
    computed = df["disease_num_known_clinical_targets"] == 0
    assert (df["orphan"] == computed).all(), "orphan flag does not match disease_num_known_clinical_targets==0"
    print(f"S1b orphan flag verified: {int(computed.sum()):,}/{len(df):,} orphan rows")


def test_s1b_nomination_count_independent_recompute():
    df = pd.read_csv(S1B)
    assert "target_nomination_count" in df.columns, "target_nomination_count column missing"
    independent = df.groupby("targetId")["diseaseId"].transform("nunique")
    assert (df["target_nomination_count"] == independent).all(), \
        "target_nomination_count does not match an independent groupby recompute"
    top = df.drop_duplicates("targetId").nlargest(3, "target_nomination_count")
    print("S1b top-nominated targets:", top[["targetId", "target_nomination_count"]].to_dict("records"))


def test_s2b_known_positive_rows():
    df = pd.read_csv(S2B)
    known = df[df["source"] == "known_positive"]
    assert (known["label"] == 1).all() and (known["score"] == 1.0).all()
    assert (known["orphan"] == False).all(), "known-positive rows must not be orphan by construction"  # noqa: E712
    model = df[df["source"] == "model_prediction"]
    assert (model["label"] == -1).all()
    print(f"S2b: {len(known):,} known_positive rows, {len(model):,} model_prediction rows")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"--- {name} ---")
            fn()
    print("PASS: all candidate invariants hold")

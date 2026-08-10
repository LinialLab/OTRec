"""Per-model prediction sanity checks for the 25.12 retrain.

Seams under test:
  1. retrain_2512/Outputs/held_out_preds_2512.parquet -- labeled target-disjoint
     held-out predictions, one column per model (otrec_score, ottree_score).
     This is the only artifact where discrimination can be measured against
     ground truth.
  2. retrain_2512/Outputs/S1b-DL_novel_predictions_2512.csv -- the released
     novel-candidate scores (OTRec `score`, plus the OTTree second opinion).

Oracles are independent of the code under test:
  - ground-truth labels derive from Open Targets known_drug (phase non-NA),
  - probabilities must lie in [0,1] by definition of the sigmoid head,
  - the manuscript's published target-disjoint CV performance (OTRec
    0.950 ROC / 0.844 PR; OTTree 0.947 / 0.772) gives loose lower bounds --
    the thresholds below sit far under those so they flag breakage, not
    ordinary release-to-release drift,
  - the degeneracy checks encode a real failure mode already documented in
    analysis/HANDOFF_REPORT.md (the temporal Node2Vec baseline emitted one
    distinct value per disease, i.e. it never used the target at all).

Why these tests earn their place: the sign-flip defect found in this session
(cls_head kernel -5.52, positives encoded as NEGATIVE cosine) passed every
invariant test we had. test_positives_outrank_negatives and
test_models_agree_with_each_other are the artifact-level detectors for that
class of bug.

Run: python3 retrain_2512/test_prediction_sanity.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = Path("/mnt/d/Research/OpenTargetsTransfer/retrain_2512/Outputs")
HELD_OUT = OUT / "held_out_preds_2512.parquet"
S1B = OUT / "S1b-DL_novel_predictions_2512.csv"

MODELS = {"otrec_score": "OTRec", "ottree_score": "OTTree"}

# Loose floors: published CV values are ~0.95 ROC / ~0.77-0.84 PR. A broken or
# inverted model lands near 0.5 ROC and near the base rate for PR.
MIN_ROC = 0.80
MIN_PR_LIFT = 3.0  # PR-AUC must beat the positive base rate by at least this factor


def _held_out() -> pd.DataFrame:
    return pd.read_parquet(HELD_OUT)


def test_scores_are_probabilities():
    df = _held_out()
    for col, name in MODELS.items():
        s = df[col]
        assert s.notna().all(), f"{name}: {int(s.isna().sum())} NaN predictions"
        assert s.between(0.0, 1.0).all(), \
            f"{name}: predictions outside [0,1] (min {s.min()}, max {s.max()})"
        print(f"  {name:<7} range [{s.min():.4f}, {s.max():.4f}], no NaN")


def test_each_model_discriminates_on_labels():
    df = _held_out()
    base_rate = df.label.mean()
    for col, name in MODELS.items():
        roc = roc_auc_score(df.label, df[col])
        pr = average_precision_score(df.label, df[col])
        lift = pr / base_rate
        print(f"  {name:<7} ROC {roc:.4f}  PR {pr:.4f}  (base rate {base_rate:.4f}, lift {lift:.1f}x)")
        assert roc > MIN_ROC, f"{name}: ROC {roc:.4f} below floor {MIN_ROC} -- model may be broken/inverted"
        assert lift > MIN_PR_LIFT, f"{name}: PR-AUC lift {lift:.1f}x below floor {MIN_PR_LIFT}x"


def test_positives_outrank_negatives():
    """Sign-flip detector. An inverted head still yields a high ROC under
    metric conventions that take |AUC|, but scores positives BELOW negatives."""
    df = _held_out()
    for col, name in MODELS.items():
        pos_mean = df.loc[df.label == 1, col].mean()
        neg_mean = df.loc[df.label == 0, col].mean()
        print(f"  {name:<7} mean score: positives {pos_mean:.4f} vs negatives {neg_mean:.4f}")
        assert pos_mean > neg_mean, \
            f"{name}: positives score LOWER than negatives ({pos_mean:.4f} < {neg_mean:.4f}) -- inverted"


def test_predictions_are_not_degenerate():
    """Guards the documented Node2Vec failure mode: constant-per-disease
    predictions, i.e. the model ignoring the candidate entirely."""
    df = _held_out()
    for col, name in MODELS.items():
        n_distinct = df[col].nunique()
        assert n_distinct > 1_000, f"{name}: only {n_distinct} distinct prediction values"
        per_disease_distinct = df.groupby("diseaseId")[col].nunique()
        multi = per_disease_distinct[df.groupby("diseaseId").size() > 1]
        frac_constant = (multi <= 1).mean() if len(multi) else 0.0
        print(f"  {name:<7} {n_distinct:,} distinct values; "
              f"{frac_constant:.1%} of multi-candidate diseases constant")
        assert frac_constant < 0.05, \
            f"{name}: {frac_constant:.1%} of diseases have a constant score -- candidate ignored?"


def test_models_agree_with_each_other():
    """Two independently-trained models on the same task must correlate
    positively. Strong anti-correlation means one of them is inverted."""
    df = _held_out()
    rho = spearmanr(df["otrec_score"], df["ottree_score"]).statistic
    print(f"  Spearman(OTRec, OTTree) = {rho:.4f}")
    assert rho > 0.2, f"models disagree (rho {rho:.4f}) -- suspect an inverted or broken model"


def test_released_candidate_scores_are_sane():
    df = pd.read_csv(S1B)
    assert df["score"].between(0.65, 1.0).all(), "S1b OTRec scores outside [0.65, 1.0]"
    ot = df["ottree_score"]
    assert ot.notna().all(), f"{int(ot.isna().sum())} novel rows missing the OTTree second opinion"
    assert ot.between(0.0, 1.0).all(), "S1b ottree_score outside [0,1]"
    print(f"  S1b OTRec score  [{df.score.min():.3f}, {df.score.max():.3f}] over {len(df):,} rows")
    print(f"  S1b OTTree score [{ot.min():.3f}, {ot.max():.3f}], median {ot.median():.3f}")
    # The two models should not be in outright opposition on the released set.
    rho = spearmanr(df["score"], ot).statistic
    print(f"  Spearman(OTRec, OTTree) on released candidates = {rho:.4f}")
    assert rho > -0.1, f"released candidate scores anti-correlate across models (rho {rho:.4f})"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"--- {name} ---")
            fn()
    print("\nPASS: per-model prediction sanity verified")

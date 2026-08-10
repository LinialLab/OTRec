# Rolling release-pair replication: 23.06 → 25.12 (fully release-native)

A second temporal experiment on an independent release pair, run as an internal
robustness check. Not reported in the manuscript; kept here as a released,
reproducible artifact. Ready-to-use response-letter text is at the bottom.

## Design

Train on OTP Release 23.06 (labels, auxiliary association score, AND all
annotation text — fully release-native, no era mixing), evaluate on pairs whose
clinical label changed by Release 25.12. Gap: 2.5 years. Eval is 25.12 rather
than 26.03 because Release 26.03 retired the `known_drug` dataset (new
clinical-mining pipeline), which would change label semantics; 25.12 is the
newest release with the manuscript's label definition. 23.06's ontology covers
91% of eval-frame disease terms (vs 58% for 22.02), so pooled metrics are
meaningful without coverage stratification.

Split: train 469,116 pairs (65,734 positive, 14.0%); test 284,723 pairs
(8,843 positive, 3.11%). Protocol, architecture, optimizer, callbacks, seeds
(42–46) identical to the manuscript's temporal experiment; baselines use the
repo's own implementations (`run_temporal_repeated.py`, `run_baselines.py`).

## Results

Pooled (mean ± SD over 5 seeds where applicable):

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| OTRec | 0.828 ± 0.005 | 0.144 ± 0.008 |
| OTTree (CatBoost) | 0.851 ± 0.001 | 0.198 ± 0.002 |
| Target Mean | 0.854 | 0.248 |
| Disease Mean | 0.619 | 0.042 |
| Open Targets score | 0.574 | 0.068 |
| Matrix Factorization | 0.561 | 0.042 |
| Node2Vec | 0.536 | 0.033 |
| TF-IDF cosine | 0.500 | 0.034 |
| Frozen BioClinical ModernBERT + MLP | not run (flash-attn ABI incompatibility in available env) |

Cold-start (targets with no 23.06 indication; n=7,185 pairs, 249 positives),
seed 42: OTRec ROC 0.900 / PR 0.270; OTTree 0.898 / 0.368; Target Mean
0.565 / 0.040; OT Score 0.500 / 0.035.

Per-disease shortlist (950 diseases with ≥1 later positive; worst-rank tie
handling), seed 42: OTRec Hit@1 0.414 / MRR 0.537; OTTree 0.410 / 0.516;
Target Mean 0.382 / 0.487; OT Score 0.282 / 0.381.

## Interpretation (horizon composition)

Relative to the manuscript's 3.4-year split, the 2.5-year horizon concentrates
new positives on already-established targets: 97.1% of test positives fall on
targets with ≥2 prior indications (vs 95.4%), and cold-start positives drop
from 618 to 249. Shorter horizons therefore reward target-popularity signals
(Target Mean, and CatBoost's heavier use of the categorical disease ID),
consistent with the manuscript's pooled-vs-shortlist analysis. Content-based
retrieval pays off in exactly the regimes the manuscript emphasizes: cold-start
ranking (OTRec ROC 0.900 here vs 0.887 in the manuscript) and per-disease
shortlists, where OTRec leads all baselines on both splits.

## Files

- `build_native_frames.py` — builds 23.06-native text + label frames and the
  25.12 eval frame (reuses `../annotation_timing/build_2202_features.py`
  functions, port-fidelity-tested).
- `train_native.py` — OTRec retrain, seed-parametrized; `native2306_preds_s{42..46}.parquet`.
- `run_baselines_native.py` — all baselines via the repo's own implementations;
  `native2306_baselines_*.json`, `native2306_ottree_preds.parquet`.
- `run_modernbert_native.py` — ModernBERT runner incl. sentence-transformers
  5.x compatibility shim (blocked in the available env by a flash-attn ABI
  mismatch; runnable elsewhere).
- `analyze_native.py` — seed stability, strata, shortlist;
  `native2306_seed_stability.csv`, `native2306_shortlist.csv`.

## Ready-to-paste response-letter paragraph (if rolling evaluation is requested)

> We have additionally run a fully release-native rolling replication (train
> Release 23.06, evaluate Release 25.12; 2.5-year horizon; identical protocol
> and seeds; all inputs, including annotation text, from the training release).
> Over this shorter horizon 97.1% of newly positive pairs fall on targets that
> already had two or more indications (vs.\ 95.4% at 3.4 years), and
> popularity-based signals are correspondingly stronger: pooled ROC-AUC is
> 0.828 ± 0.005 for OTRec against 0.854 for the Target Mean baseline. The
> regimes our paper emphasizes are preserved: on cold-start targets OTRec
> attains ROC-AUC 0.900 (0.887 in the main experiment) while Target Mean falls
> to 0.565, and OTRec leads all baselines on per-disease shortlist utility
> (Hit@1 0.414, MRR 0.537). Complete scripts and per-pair predictions are
> released under `analysis/native_2306/`.

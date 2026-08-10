# OTRec 25.12 refresh (deployed-model retrain)

Retrains the deployed OTRec model on Open Targets Platform Release 25.12 (the
newest release with the manuscript's label definition; 26.03 retired the
`known_drug` dataset) and regenerates the Space runtime artifacts and a new
novel-candidate set. The manuscript's experiments and released S1/S2 files are
untouched; this is a deployment refresh, not a paper change.

## Held-out evaluation (informational)

Target-disjoint 20% held-out split, identical recipe to the training notebook
(seed 42, stratified by per-target label). Single split, not repeated CV.

| Model | Release | ROC-AUC | PR-AUC | P@0.65 | R@0.65 |
|---|---|---|---|---|---|
| OTRec seed 42 | 25.12 | 0.944 | 0.801 | -- | -- |
| OTRec seed 43 (DEPLOYED; picked on validation PR-AUC) | 25.12 | 0.947 | 0.824 | -- | -- |
| OTRec seed 44 | 25.12 | 0.947 | 0.818 | -- | -- |
| OTRec (deployed weights) | 25.06 | 0.954 | 0.845 | 0.905 | 0.634 |
| OTRec (training-session log) | 25.06 | 0.945 | 0.850 | -- | -- |
| OTTree 5-fold OOF (all pairs) | 25.12 | 0.946 | 0.765 | -- | -- |
| OTTree 5x5 CV (manuscript Table 1) | 25.06 | 0.947 | 0.772 | -- | -- |

The 25.12 frame is larger (705,239 pairs / 67,013 positives vs 663,351 /
67,532) with a lower positive rate (9.5% vs 10.2%); the small metric
differences are consistent with release growth plus run-to-run variance.

## The serving-vocabulary defect and fix

`build_two_tower_model` re-adapts TextVectorization from whatever frame it is
given; vocabularies are frequency-ordered, so any frame other than the full
training frame yields a same-sized but permuted vocabulary -- weights load
without error and predictions silently degrade to chance (measured on the
deployed 25.06 pair: held-out ROC-AUC 0.52 vs 0.95; known-positive-in-top-10
recovery 1.5% vs 75%). Fixed by serializing the training-time vocabularies
(`OTRec/gradio/vocab_io.py`, `vocabs.json.gz`) and applying them before
`load_weights`. Tests: `test_vocab_fix.py` (bit-exact reproduction from the
corrupting frame shape), `validate_deployed_fix.py` (full metric panel),
`test_df_learn_sub_equivalence.py` (documents the failure mode).

Two further defects found and fixed along the way:
- Candidate generation ranked by raw cosine, assuming a positive cls_head
  kernel; this run learned a negative kernel, inverting retrieval. Fixed by
  ranking on the calibrated probability (sign-agnostic);
  `generate_candidates_2512.py`.
- The packaged `target_df` must be filtered to the app's candidate universe
  (truthy tractability or known drug): 17,073 targets in 25.12 vs 17,065 in
  25.06 (+12/-4) -- recomputed per release, not inherited.

Deployment seed selection used validation PR-AUC only (0.837 / 0.868 / 0.846
for seeds 42/43/44); test-set numbers above are reported for the record.
Served-ranking audit of the deployed seed: known-positive-in-top-10 recovery
0.875 (25.06 fixed serving: 0.750), pseudogene at rank 1 in 14.5% of top-10s
(25.06: 27.5%).

## New candidate set (separate release, S1/S2 untouched)

`Outputs/S1b-DL_novel_predictions_2512.csv` (deployed seed 43): 166,275 novel
pairs over 4,903 diseases (old S1: 214,968 / 4,347). Adds
`target_nomination_count` and an OTTree second-opinion score. The popularity
bias persists but is milder than in the seed-42 run: GUCY1B2 remains the top
hub (79.7% of diseases) while CLCA3P/GLRA4 dropped out of the top nominees;
`Outputs/S1b_curated_2512.csv` (protein-coding, non-hub; 132,334 rows) is the
clean secondary view.
Six old novel predictions became known positives by 25.12 (five spinal-cord
-injury GABA/calcium-channel pairs; PPIA x keratitis).

## Reproduction

```
python3 test_build_2512_frames.py     # oracle gate vs released 25.06 artifacts
python3 build_2512_frames.py          # frames from 25.12 raw tables
python3 train_main_2512.py            # main model + matched df_learn_sub + weights
python3 test_gradio_pair.py           # matched-set load + negative control
python3 generate_candidates_2512.py   # S1b/S2b
python3 test_candidates_2512.py       # invariants + volume regression
python3 test_prediction_sanity.py     # per-model discrimination/degeneracy
python3 build_comparison_lookup_2512.py  # OTTree 5-fold OOF + app lookup
python3 prepare_gradio_artifacts_2512.py # Space artifact set
```

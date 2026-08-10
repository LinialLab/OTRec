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

## Gene-family overlap analysis (2026-08)

Question (coauthor): do gene-family members share their top predicted diseases,
and is the sharing biology or popularity? Case study: RAB GTPases; scale-up:
all 471 HGNC gene groups with >= 10 members in the candidate universe.

Design (externally audited before running): per-gene top-50 predicted-disease
lists (equal sizes; threshold >= 0.65 kept as sensitivity); family statistic =
mean pairwise shared diseases; null = random gene sets drawn from the real
per-gene lists, which conditions EXACTLY on disease popularity (exact null
mean = sum_d C(n_d,2)/C(N,2)); 10,000 draws/size, (r+1)/(n+1) p, BH FDR.
A degree-preserving curveball null was considered and rejected as redundant
and anti-conservative. Non-circular check: sibling-transfer test (known
clinical pair (g,d) -> is d in siblings' top-50 more than n_d/N predicts).

Results (K=50 primary):
- Exact popularity baseline: two RANDOM druggable genes already share
  24.6/50 top predicted diseases.
- 320/471 families (68%) exceed it (BH q < 0.05); RAB: obs 33.25/50,
  excess +8.65, p 1e-4, 54th excess percentile — a typical family.
  Stable at K=20/100 and with known pairs masked (novel-only).
- Negative control: "MicroRNA protein coding host genes" (arbitrary grouping,
  n=822) shows no excess (+0.38, p 0.19).
- Top excess: Ig/TCR loci, collagens, H2B histones, eIF3, ribosomal proteins —
  families with duplicated annotation text.
- Sibling transfer: pooled lift 1.20 (465k observed vs 386k expected),
  median family lift 1.71, 83% of families > 1; complexes reach lift 7-20
  (ribosome, proteasome, MMPs). Family structure carries real held-out signal.
- Curated control arm: curated clinical sets are THEMSELVES family-structured
  (38/115 families significant; Tubulin beta shares 345.8 curated diseases per
  gene pair vs null 3.7) — trials annotate whole complexes/families, so
  family-coherent predictions mirror the label space.

Scripts: precompute_gene_topk.py -> family_overlap_analysis.py,
curated_control_overlap.py, plot_family_volcano.py, rab_overlap_analysis.py.
Outputs: family_overlap_results.csv, family_overlap_sibling_transfer.csv,
family_overlap_curated_control.csv, family_overlap_volcano.png,
rab_overlap_{heatmap,recurrence}.{png,csv}, gene_topk.npz.
Literature grounding saved in sources/papers_20260810_gene_family_overlap_null_models.md
(paralog buffering: Hsiao & Vitkup 2008; degree baseline: Zietz et al. 2024).

Post-review corrections (2026-08-10, claims checked against data):
- Top-50 lists draw on only 2,604/46,960 diseases; the 100 most popular fill
  83% of all slots. "HIV-Associated Lipodystrophy" is in the top-50 of 88% of
  ALL genes (rank 10) — the RAB "61/62" recurrence is a global-hub property,
  not family biology. Top hubs include non-disease terms (drug toxicity #1,
  drug interaction #3): candidate-disease filtering for the app is worth
  considering.
- RETRACTED: "close RAB paralogs overlap less than the family mean"
  (threshold-set-size artifact; equal-size lists show paralogs ~ family mean)
  and "selective RAB genes carry distinctive predictions" (RAB3A's Rh
  isoimmunization is itself the #14 genome-wide hub; RAB27A melanoma does not
  reproduce). After hub-discounting no distinctive RAB prediction remains.
PI-facing review page: rab_family_overlap_review.html (Claude artifact,
built from scratchpad; all numbers recomputed from Outputs/ CSVs).

RAB excess decomposition (rab_excess_decomposition.py -> Outputs/
rab_excess_decomposition.csv): the +8.65 family excess is a family-wide
ONCOLOGY tilt — top-10 contributing diseases (83% of excess) are all cancers
that are modestly popular genome-wide (8-58% of genes) but near-universal in
RAB (52-64/64); 46/50 top contributors are oncology terms; RAB conversely
avoids non-cancer hubs (Rh isoimmunization 17/64 vs 80% global). PI page
updated with this table + method-provenance box (deployed 25.12 model, not
the manuscript's 25.06).

Pair-level + cutoff follow-up (rab_pair_overlap.py -> rab_pair_stats.csv,
rab_pair_overlap_heatmap.png, rab_overlap_disease_dumbbell.png):
- Random-pair null (200k pairs): mean 24.6/50 shared, 95th pct 39. Only 19% of
  the 2,016 RAB pairs exceed the 95th pct (context: Ig-lambda 100%, beta-
  tubulins 67%, interleukin receptors 1%). RAB3A shares LESS than random with
  most of its own family.
- Cutoff/stringency on held-out (143,898 pairs, base rate 10.1%): OTRec>=0.65
  precision 0.908/recall 62%; >=0.90 0.970/46%. Two-model agreement is the
  efficient knob: OTRec>=0.65 AND OTTree>=0.5 gives 0.964/47% (S1b: 95,305
  novel rows vs 66,100 at OTRec>=0.90 alone); OTRec>=0.90 AND OTTree>=0.5:
  0.984/40%. Suggested tiers documented in the PI page; both scores already
  ship in released rows. Cutoffs do not change hub composition.
- Top-K choice is scale-free: baseline ~49% of list at K=20/50/100.

K=30 primary rerun (family_overlap_k30.py; user request — matches app reading
depth): all conclusions replicate. Null 14.74/30 (49% of list — scale-free
across K); 297/471 families significant (63%); RAB obs 18.43, excess +3.69,
p 2e-4, 40th percentile; sibling transfer lift 1.23 (z=131); pair test: 6% of
RAB pairs above the random 95th pct (24/30) vs Ig-lambda 100% / beta-tubulin
71% / interleukin receptors 0%; decomposition top-10 all oncology (HER2-
breast 21x, adult ALL 21x, anus cancer 14x enrichment). Outputs:
family_overlap_results_k30.csv, rab_excess_decomposition_k30.csv,
rab_pair_stats_k30.csv. Figures regenerated at K=30 (volcano, pair heatmap,
dumbbell); RAB score heatmap redesigned transposed — diseases as readable rows
with per-disease genome-wide >=0.65 rate in the label. PI page now K=30
primary throughout; dual-model cutoff section unchanged (threshold-based,
K-independent).

OTTree popularity check (rab_ottree_overlap.py; single CatBoost fit on full
25.12 frame, 64 RAB + 499 random genes x all 46,960 diseases): OTTree is MORE
popularity-concentrated than OTRec in prediction space, not less. Its top-30
lists use only 195 distinct diseases (top-100 fill 99% of slots vs OTRec's
89%); random-pair baseline 21.9/30 (73% of list, vs OTRec 14.7/30 = 49%);
RAB pairs mean 25/30 with 10% above the 95th pct — no RAB-specific structure
visible, every gene gets the same trial-heavy leukemia/carcinoma hubs
(diseaseId categorical + label frequency; unseen diseases fall back to text).
Implication: OTTree discriminates well within the labeled space but collapses
as a full-space ranker; OTRec's text embeddings generalize to unlabeled
diseases. Known-only vs novel-only overlap: novel-masked variant ~identical
to raw (310 vs 320/471 sig; RAB pct 56 vs 54); known-only = curated control
(38/115 sig; complexes extreme).

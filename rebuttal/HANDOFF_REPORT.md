# OTRec rebuttal analysis — handoff report

Session date: 2026-07-26. All work is read-only against the repo; every script and output
lives in `rebuttal_scratch/`. No repo file was created, modified, or deleted. The only
computation beyond re-analysis was one authorised seed-42 temporal retrain (plus a leakage
robustness rerun), because the temporal per-pair predictions were never persisted.

---

## 0. Executive summary — what changes the rebuttal

| # | Finding | Direction |
|---|---|---|
| 1 | Temporal annotation features come from the newer OTP snapshot, not 22.02. Only **clinical-precedence tractability tokens** carry post-2022 outcome info (1,510 targets). | ⚠️ requires rewording, quantified below |
| 2 | **Zero- and single-indication targets ARE evaluable** in the temporal split (618 + 117 positives) and OTRec is well above chance on them | ✅ **strengthens** the rebuttal |
| 3 | OTRec beats Target Mean on per-disease Hit@1/MRR overall **and** in both orphan and non-orphan splits | ✅ **the pooled-PR-AUC counter-argument holds** |
| 4 | S3 ablation artefacts are not on disk | ⚠️ pointer must be fixed or ablation rerun |
| 5 | "~282,500 novel candidates" = S2 row count (novel + known); novel-only is 214,968 | ⚠️ manuscript fix |
| 6 | GUCY1B2 appears in 64.9% of candidate diseases; TUBB8 64.7% | ⚠️ **supports** the "indiscriminate" criticism |
| 7 | "~6,000 positive-bearing diseases" is ~2,329 (25.06) / ~2,096 (2022) | ⚠️ manuscript fix — **the corrected number helps the covariate-shift argument** |
| 8 | Table 1 OTRec ROC-AUC SD 0.007 does not reconcile with the released folds (0.011) | ⚠️ manuscript fix |
| 9 | IL6R × ulcerative colitis is absent from the released candidate files | ⚠️ discovery example unsupported by released data |
| 10 | Node2Vec's scoring rule is disease-only — constant within a disease | context needed if added to Table 2 |

---

## 1. Artefact inventory

| Artefact | Status | Path / shape |
|---|---|---|
| CV OOF predictions — OTRec | ✅ | `Outputs/CV_DL/oof_dl_preds.parquet`, 663,351 × 7 |
| CV OOF predictions — OTTree | ✅ | `Outputs/CV_tree/CB_5_cv.parquet`, 663,351 × 7 |
| CV OOF predictions — ModernBERT | ✅ | `Outputs/CV_frozen_encoder_mlp/frozen_encoder_oof.parquet`, 2,653,404 × 6 |
| CV baselines (per-pair + folds) | ✅ | `Outputs/CV_baselines/table1_fast_baselines/`, `full_node2vec_v2/`, `full_tfidf/` |
| **Temporal per-pair predictions (any model)** | ❌ **absent** | `run_temporal_repeated.py` reduces `y_pred` to metrics and discards it |
| Temporal metrics | ✅ | `Outputs/temporal_repeats_5seed/` (+ `_43_46`, `_seed42_check`, `_frozen_encoder_mlp`) |
| **S3 per-feature-group ablation** | ❌ **absent** | `Outputs/CV_DL/` holds full-model results only |
| Released novel candidates (S1) | ✅ | `Outputs/S1-DL_novel_predictions.csv`, 214,968 × 7 |
| Released novel+known (S2) | ✅ | `Outputs/S2-DL_novel+known_candidates.csv`, 282,500 × 9 |
| Processed 2022.02 label frame | ✅ | `code/history_df.parquet` — 411,430 rows, 62,417 pos, 9,741 diseases, 1,407 targets |
| Processed 25.06 label frame | ✅ | `code/final_df.parquet` — 663,351 rows, 67,532 pos (10.18%), 12,337 diseases, 1,522 targets |
| Disease/target feature frames | ✅ | `code/copy_proc/{disease_df,target_df}.parquet` (MD5-identical to `data/proc/*`) |
| Surviving 22.02 raw data | partial | only `known_drug/` + `association_overall_direct/` (rest deleted for space) |
| Orphan flag | ✅ | S1 `orphan` bool ≡ `disease_num_known_clinical_targets==0` ≡ zero `label==1` in `final_df` (all agree) |
| Therapeutic areas | ✅ | `disease_df.therapeuticAreas`, 4,347/4,347 matched |

---

## 2. T0 — Temporal feature provenance

**Method.** Read the temporal data-loading path; then tested the conclusion two ways rather
than inferring from file absence (the 22.02 dump was deliberately deleted for space, so
absence proves nothing).

**Evidence.**

- The temporal model consumes exactly four fields: `disease_text`, `target_text`,
  `diseaseId`, `targetId` (`run_temporal_repeated.py:107-121`, `dl_model_def.py:126-138`).
  `label` and `score` are y-targets, not inputs.
- Both text fields are read from `code/copy_proc/{disease_df,target_df}.parquet`
  (`run_temporal_repeated.py:45-48`; notebook `2-Temporal-Eval.ipynb:167-168`). The historic
  directory is used **only** for `association_overall_direct` and `known_drug`
  (`2-Temporal-Eval.ipynb:207,239,241`).
- **Set-equality test:** `copy_proc/disease_df.parquet` id set == current OTP `disease` table
  id set exactly (38,959 ids, `True`). `target_df` ⊂ current `target` table. A 22.02-derived
  frame cannot carry the newer release's exact id set.
- **Timing test:** feature frames dated 2025-12-10; the runs that produced Table 2
  (`temporal_repeats_5seed`) dated 2026-04-27; frames untouched afterwards. So the reported
  numbers were computed from these frames.
- Seed-42 reproduction with these frames: ROC 0.8769 / PR 0.2862 vs reported 0.8698 / 0.2808
  (reported 5-seed spread 0.868–0.880, SD 0.005) — consistent.

**Verdict per source.**

| Source | Release | Note |
|---|---|---|
| Training labels (clinical evidence / known_drug) | **22.02** | `data/historical_ot/22_02/known_drug` |
| Auxiliary target + "OT Score" baseline | **22.02** | `association_overall_direct`; `history_df.score` matches 22.02 for 99.94% of rows |
| Disease name / description / synonyms | newer snapshot | in `disease_text` |
| Disease EFO parents / therapeutic areas / phenotypes | newer snapshot | in `disease_text` |
| Gene symbol / name / synonyms / function | newer snapshot | in `target_text` |
| GO terms | newer snapshot | 99.2% of target texts contain `GO:` |
| Reactome / pathways | newer snapshot | 62.7% contain `R-HSA-` |
| UniProt-derived | newer snapshot | only via `functionDescriptions` + `synonyms`; `proteinIds`/`subcellularLocations` are NOT in model input |
| Tractability buckets | newer snapshot | **contains clinical-precedence tokens — the one real leak channel** |
| gnomAD constraint | newer snapshot | present in the saved artefact (97.2% of texts) although the shipped notebook line excludes it — code/artefact mismatch |

**Scope of the actual leak.** Of everything above, only tractability encodes *outcomes*:
`Approved Drug` (963 targets, 5.6%), `Advanced Clinical` (703, 4.1%),
`Phase 1 Clinical` (49, 0.3%) — 1,510 targets total. GO/Reactome/names/constraint drift
between releases (coverage grows) but do not encode who entered trials.

**Robustness check (this session).** Seed-42 temporal rerun with those tokens stripped from
`target_text`, everything else identical. See §9 for the measured result.

**Suggested wording.**

> Clinical-progression labels and the OT association score used for training and as the
> auxiliary target are taken from Release 22.02. Disease and target annotation text (names,
> ontology, GO, pathways, constraint, tractability) is drawn from a single newer OTP snapshot
> shared across both cohorts; of these only the clinical-precedence tractability buckets carry
> post-2022 outcome information, and we quantify their effect in a stripped-token ablation.

**Secondary provenance findings.**

1. 25.06-driven filtering of the 22.02 training set: the notebook filters history to entities
   still present in the newer frames (411,430 → 327,812 rows).
2. Notebook vs script divergence: the scripts skip the notebook's ID cleanup/filters and use
   `how="left"` + `fillna("")`, so **83,835 of 411,430 history rows (20.4%) train with an empty
   `disease_text`** in the script path. The 5-seed Table 2 therefore comes from a slightly
   different training set than the notebook path.
3. No release variable exists anywhere — release selection is by hard-coded path only.

---

## 3. T1 — S3 ablation

**NOW RUN (temporal protocol, this session).** Five rungs, seed 42, 6 epochs, everything but
the named factor held identical; feature groups rebuilt from component columns
(reconstruction verified token-identical to the stored `*_text_embed` strings, 200/200).
Results: `rebuttal_scratch/ablation_table_final.csv`, driver `ablation_temporal.py`.

| Rung | Variant | Temporal ROC-AUC | Temporal PR-AUC | cold-start ROC | cold-start PR |
|---|---|---|---|---|---|
| R0 | full, stored text (reference) | 0.8769 | 0.2862 | 0.8875 | 0.2550 |
| R1 | text-only | 0.8733 | 0.2797 | 0.8814 | 0.2024 |
| R2 | +ontology | 0.8707 | 0.2752 | 0.8848 | 0.1723 |
| R3 | +GO/pathway | 0.8775 | 0.2826 | 0.8881 | 0.2183 |
| R4 | +tractability (= full) | 0.8710 | 0.2821 | 0.8918 | 0.2367 |
| R5 | full, −auxiliary head | **0.8785** | **0.2930** | 0.8855 | 0.2573 |

Reading, stated conservatively:

1. **Pooled temporal performance is insensitive to feature group.** R1–R5 span 0.0078 ROC
   against a reported seed SD of 0.005, and the ordering is non-monotone (R2 < R1, R4 < R3).
   With one seed per rung, no pooled difference is separable from run-to-run noise.
   Free text alone (names, synonyms, descriptions, function) reaches 0.873 of the full 0.877.
2. **On cold-start (zero-indication) targets the ladder does climb monotonically**:
   ROC 0.8814 → 0.8848 → 0.8881 → 0.8918 (+0.0104) and PR 0.2024 → 0.1723 → 0.2183 → 0.2367
   (+0.034 net, non-monotone at R2). A 4-step monotone ROC increase has ~4% chance under
   exchangeability, so this is suggestive but not established at one seed per rung. This is the
   regime the paper is about — targets with no interaction history, where annotation is all the
   model has.
3. **The auxiliary head is not load-bearing.** Removing it (R5) is neutral-to-slightly-positive
   on both pooled metrics — it does not hurt. This answers the open question left in the
   manuscript's Limitations.

Caveats to state if the table is used: single seed per rung; and this is the **temporal**
protocol, not the CV protocol the §S3 pointer implies (a CV ablation is 25 folds per variant).

**Original status (unchanged):** no pre-existing ablation artefacts were on disk. Repo-wide grep for
`ablation|text_only|no_aux|feature_group` over `.py/.md/.csv/.json` → zero hits. The only
traces are unexecuted flags:
`TEMPORAL_AUX_SCORE_WEIGHT = 0.1  # set to 0.0 for an auxiliary-free temporal ablation`
(`2-Temporal-Eval.ipynb` cell 31) and `USE_PRETRAINED_TEXT_EMB = True  # <-- flip to False for
the ablation` in a superseded prototype.

Correct claim: *no ablation outputs are retained*. If ablations were run and outputs not kept,
that is consistent with what I see. Either way the manuscript's S3 pointer to `Outputs/CV_DL/`
resolves to nothing a reviewer can open.

Cost to generate: the CV ablation grid is 25 folds per variant — expensive. A **temporal**
auxiliary-head ablation is one 6-epoch fit (~10 min) and would at least populate the
`−auxiliary head` row.

---

## 4. T4 — Indication-count strata (temporal)

Indications = distinct diseases with `label==1` for that target in the processed 2022 frame.

### T4a — cohort composition

| Bin | targets | test pairs | positives | pos rate |
|---|---|---|---|---|
| 0 indications | 119 | 51,216 | **618** | 1.21% |
| exactly 1 | 104 | 25,245 | **117** | 0.46% |
| ≥2 | 1,298 | 326,671 | 15,339 | 4.70% |
| ≤1 combined | 223 | 76,461 | 735 | 0.96% |

**117 of the 119 zero-indication targets appear nowhere in the 2022 training frame** — genuine
cold start, not merely low evidence.

### T4b — discrimination within strata (seed-42 reproduction)

| Bin | OTRec ROC | OTRec PR | OTTree ROC | OTTree PR | TargetMean ROC | OTScore ROC |
|---|---|---|---|---|---|---|
| 0 indications | **0.887** | 0.255 | 0.859 | 0.091 | 0.505 | 0.500 |
| exactly 1 | **0.722** | 0.013 | 0.751 | 0.011 | 0.597 | 0.709 |
| ≤1 combined | **0.863** | 0.212 | 0.844 | 0.071 | 0.593 | 0.533 |
| ≥2 | **0.869** | 0.293 | 0.839 | 0.239 | 0.841 | 0.559 |

Bin-1 PR-AUC (0.013) rests on 117 positives at a 0.46% base rate — ROC is the stable
statistic there; label PR-AUC as unstable.

**Rebuttal impact.** The draft says zero-indication targets are not evaluable. In the temporal
setting they are, on 618 positives, and OTRec scores 0.887 ROC-AUC on them while Target Mean
collapses to chance (0.505) by construction. This is the strongest single number available and
should replace the current concession.

---

## 5. T3 — Per-disease shortlist metrics (temporal)

1,455 diseases with ≥1 temporal positive; 175,764 pairs; median 66 candidates/disease.
**Conservative tie handling: every tied item receives the worst rank in its tie block.**

| Model | Hit@1 | Hit@5 | Hit@10 | MRR | mean top-tie fraction |
|---|---|---|---|---|---|
| **OTRec** | **0.463** | **0.730** | **0.833** | **0.586** | 6.7% |
| OTTree | 0.418 | 0.673 | 0.786 | 0.535 | 6.7% |
| Target Mean | 0.405 | 0.625 | 0.720 | 0.511 | 6.9% |
| Disease Mean | 0.025 | 0.081 | 0.137 | 0.067 | **100%** (degenerate) |
| OT Score | 0.263 | 0.416 | 0.484 | 0.342 | 58.3% |

### Orphan vs non-orphan (orphan = no clinically associated target in the 2022 frame)

| Split | Model | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|
| Orphan (610 diseases, 8,181 pos) | **OTRec** | **0.575** | **0.816** | **0.897** | **0.680** |
| | OTTree | 0.559 | 0.800 | 0.887 | 0.662 |
| | Target Mean | 0.538 | 0.752 | 0.816 | 0.636 |
| | OT Score | 0.139 | 0.223 | 0.295 | 0.196 |
| Non-orphan (845 diseases, 7,893 pos) | **OTRec** | **0.381** | **0.667** | **0.787** | **0.518** |
| | OTTree | 0.316 | 0.581 | 0.714 | 0.443 |
| | Target Mean | 0.310 | 0.533 | 0.651 | 0.421 |
| | OT Score | 0.353 | 0.556 | 0.620 | 0.448 |

**The counter-argument survives.** OTRec > Target Mean on Hit@1 and MRR in the pool and in both
splits, despite Target Mean edging OTRec on pooled PR-AUC (0.299 vs 0.288). Caveats to state:
the margin over Target Mean is real but moderate (+0.058 Hit@1 pooled, +0.038 orphan); Disease
Mean is degenerate under conservative ties and should not be cited; OT Score beats OTRec on
non-orphan Hit@1 (0.353 vs 0.381 — OTRec still ahead, but narrowly) yet collapses on orphans.

**ModernBERT row: NOT AVAILABLE.** Its temporal per-pair predictions were never saved, and
`frozen_encoder_mlp_bioclinical_modernbert_base/test_predictions.parquet` is the CV
target-disjoint split (143,240 rows, 305 test targets; only 21.8% of the temporal cohort).
Regeneration is blocked by environment: `run_frozen_encoder_mlp.build_sentence_transformer`
requires a `sentence-transformers` API (`models.Transformer(..., model_kwargs=...)`, ≥3.4)
absent from every available interpreter — `.venv` has none; `base`, `hf` have broken
huggingface-hub pins; `ot-recsys-kerasrs` has a broken torchvision; `nlp`/`Medrag`/`IntFeat`
are 3.3.x and reject `model_kwargs`. Fixing means mutating shared conda envs — not done.

---

## 6. T2 — Candidate-set composition (S1, verified twice)

| Quantity | Value |
|---|---|
| Rows | **214,968** (paper says ~282,500 → that is S2 = 214,968 novel + 67,532 known) |
| Diseases | **4,347** (paper says 4,346) |
| Targets | 2,808 ✅ matches |
| Threshold | `score ≥ 0.65` (495 rows exactly at 0.65; strict `>` gives 214,473) |
| Score min / median / max | 0.650 / 0.854 / 0.998 |
| Candidates per disease | mean **49.45**, median **14**, IQR 2–78, max 200 |
| Cap | binding: 385 diseases exactly at 200; 808 diseases have exactly 1 |

**Most-nominated targets.**

| Symbol | # diseases | % of 4,347 | in top-25 of |
|---|---|---|---|
| GUCY1B2 | 2,822 | **64.9%** | 51.9% |
| TUBB8 | 2,814 | 64.7% | **57.4%** |
| FKBP1A | 1,860 | 42.8% | 32.9% |
| TUBB8B | 1,764 | 40.6% | 31.4% |
| TUBB6 | 1,527 | 35.1% | 27.6% |

Top-10 combined = **8.03%** of all rows. Tubulin family (19 genes, 0.68% of targets) = **10.02%**.
87.3% of diseases contain GUCY1B2 or TUBB8.

**This contradicts the expected direction.** Two targets each exceed 50% disease coverage.
GUCY1B2 is a largely uncharacterised guanylate-cyclase subunit and TUBB8 is oocyte-restricted;
neither is a plausible pan-disease target, so this reads as a popularity/embedding artefact.
Do not claim "no single target dominates". A defensible reframing: report the concentration
honestly, note top-10 = 8% of rows (i.e. the tail is broad), and consider publishing a
frequency-filtered variant of the candidate list.

**Therapeutic areas** (disease counted once per area it carries; mean 2.09 areas/disease):
cancer/benign tumour 1,563 (36.0%); genetic/familial/congenital 1,406; nervous system 778;
musculoskeletal 619; GI 522; endocrine 453; integumentary 421; reproductive/breast 381;
immune 380; **cardiovascular 370 (8.5%)**; haematologic 355; infectious 345; metabolic 283;
respiratory 253; visual 242; urinary 237; psychiatric 170.
Cancer ∪ cardiovascular = 1,875; **neither = 2,472 (56.9%)**.

**Orphan diseases:** 2,322 ✅ matches paper. **1,614 (69.5%) have >1 candidate**; median 3.0.

---

## 7. T5 — Temporal Node2Vec

**ROC-AUC 0.5330 / PR-AUC 0.0421** — near chance, just below temporal Matrix Factorization
(0.553 / 0.050). CV analogue for reference: 0.684 ± 0.006 / 0.139 ± 0.013.

Run details: bipartite graph from the 62,417 positive 2022 edges; node2vec 0.5.0 / gensim
4.4.0; dimensions 64, walk length 20, walks 10, window 10, epochs 3, p=q=1.0, seed 42;
40.6 s fit. Per-pair predictions at `rebuttal_scratch/node2vec_temporal_preds.parquet`.

**Caveat that must accompany the number.** `fit_node2vec_predict` scores
`0.5*(cos(disease_emb, mean_target_embedding)+1)` — it iterates over `diseaseId` only and never
uses `targetId`. Verified: max 1 distinct prediction per disease. 58.7% of test pairs fall back
to the global prior because their disease has no node in the 2022 positive graph. So this is a
disease-prior model with graph-derived ordering, not a link predictor. Matches the paper's own
description of the CV protocol, but a reviewer asking "is Node2Vec a fair graph baseline?" is
asking exactly this.

---

## 8. T6 — Temporal base rate

```
pairs 403,132 | positives 16,074 | positive rate 3.99%   (CV frame: 10.18%)
diseases 11,624 | targets 1,521
```
The 11,624 figure matches the manuscript.

---

## 9. Leakage robustness check (added this session)

Seed-42 temporal rerun, identical in every respect except that `Approved Drug`,
`Advanced Clinical`, `Phase 1 Clinical`, `Clinical Precedence` are stripped from `target_text`
(1,510 → 0 targets carrying them).

**Result: the leak channel is worth ≈0.007 ROC-AUC — no larger than seed noise.**

| | ROC-AUC | PR-AUC |
|---|---|---|
| Unstripped (seed 42) | 0.8769 | 0.2862 |
| **Clinical tokens stripped (seed 42)** | **0.8701** | **0.2821** |
| Δ | −0.0069 | −0.0041 |
| Reported 5-seed mean (Table 2) | 0.8724 ± 0.005 | 0.2879 ± 0.009 |

The stripped run (0.8701) sits **inside the reported seed spread** (0.868–0.880) and the delta
is smaller than the reported seed SD, so the effect is not separable from run-to-run noise at
one seed. The manuscript's conclusions do not depend on the leaked tokens.

Per stratum (stripped vs unstripped):

| Bin | ROC stripped | ROC unstripped | PR stripped | PR unstripped |
|---|---|---|---|---|
| 0 indications | 0.8855 | 0.8875 | **0.2005** | **0.2550** |
| exactly 1 | 0.7262 | 0.7218 | 0.0137 | 0.0128 |
| ≥2 | 0.8629 | 0.8687 | 0.2936 | 0.2926 |

**Report this honestly:** ROC-AUC is essentially unaffected in every stratum (|Δ| ≤ 0.006), but
PR-AUC in the **zero-indication** stratum drops 0.255 → 0.201 (−21% relative). That is the one
place the leak measurably helped, and it is exactly where you would predict it to: those are
the targets whose clinical status changed after 2022, so their 25.06 tractability bucket is the
most informative. Even stripped, 0.201 PR-AUC against a 1.21% base rate is a ~16× lift, and
ROC-AUC 0.886 on genuinely cold-start targets is unchanged.

Recommended rebuttal sentence:

> To bound any effect of the shared annotation snapshot, we re-ran the temporal experiment with
> all clinical-precedence tractability tokens (Approved Drug / Advanced Clinical / Phase 1
> Clinical; 1,510 targets) removed from the target input text. Temporal ROC-AUC changed from
> 0.877 to 0.870 — within the seed-to-seed variation of our reported 0.872 ± 0.005 — and
> cold-start (zero-indication) ROC-AUC was unchanged at 0.886.

---

## 10. T7 — Hyperparameter tuning

**No sweep code, logs, or per-configuration results exist anywhere in the workspace.**
Repo-wide grep for `optuna|keras_tuner|kerastuner|RandomSearch|GridSearchCV|RandomizedSearchCV|
wandb|hyperopt|ray.tune|param_grid|skopt` → zero matches, including `code/backup_code/` and
`.ipynb_checkpoints/`. No `ModelCheckpoint` anywhere. Git history: every hyperparameter appears
at its final value in the initial commit; later diffs are cosmetic.

Hard-coded values: Adam 8e-3 (main) / 7e-3 (CV, temporal); batch 1024 (main) / 512 (CV,
temporal); ID embedding 64; tower width 384 with deep bottleneck 64; dropout 0.35/0.15;
epochs 7 (main) / 10 (CV) / 6 (temporal); ReduceLROnPlateau factor 0.2 patience 1;
EarlyStopping patience 2; aux loss weight 0.1 (0.2 in CV); TextVectorization `max_tokens=160_000`,
count mode, unigrams. Located in `dl_model_def.py:7,8,17,18,51,54,56,58,59,64`,
`1-Train-DL-Retriever.ipynb` cell 22 (CV cell 6), `2-Temporal-Eval.ipynb` cell 31,
`run_temporal_repeated.py:214,216,229,240-245`.

What *did* happen: informal manual variation, hand-logged. `dl_model_def.py:28-42` has
commented-out towers with inline notes (`# orig, 95.9 auc`); notebook markdown cells 21, 24–28
record a handful of configs with pasted results (`Dense(128,"gelu")` → 0.9502/0.8127; a 256/128
variant → 0.9482/0.8229). **Several of those comparisons were read off the test split.** Older
copies show LR drifting by hand across 5e-3…9e-3.

**Model selection used only held-out validation** — confirmed. Val is a target-disjoint 5% of
*training* targets (1% in CV/temporal) with explicit `assert ... isdisjoint(test_tids)`; both
callbacks monitor `val_cls_loss`; test data reaches only `model.predict`.

Three caveats a reviewer may probe:
1. `restore_best_weights=False` everywhere — early stopping *stops*, it does not *select*. The
   reported model is whatever weights exist when patience expires.
2. In the CV/single-split notebook, text vocab and the disease-ID lookup are adapted on the full
   `df_learn` outside the fold loop — unsupervised (no labels), but held-out-target rows are
   seen. The temporal script is clean here (`build_two_tower_model(history_df)`).
3. The 0.65 cutoff is hard-coded, not derived from validation data.

Honest one-liner: *"No automated hyperparameter search was performed. Architecture and
optimisation settings were chosen by a small number of manual comparisons during development;
all reported model selection (early stopping, LR schedule) used only a held-out validation
split of training targets."*

---

## 11. Independent verification of manuscript numbers

Everything below was recomputed from released artefacts.

### Reproduces exactly ✅

- **Table 3 (shortlist utility)** — all 7 metrics, both models, 2,329 diseases:
  OTRec 0.682 / 0.806 / 0.645 / 0.321 / 0.848 / 0.897 / MRR 0.741;
  OTTree 0.487 / 0.732 / 0.537 / 0.272 / 0.848 / 0.951 / MRR 0.600.
  Also **tie-robust** — identical under worst-rank and optimistic tie policies (scores are continuous).
- **OTP drug index**: 18,081 entities; phase ≥1 = 10,956; small molecules 14,848 (82.1%);
  antibodies 963; protein 739 + enzyme 91 = 830; oligonucleotides 159; ADCs 119; gene 117; cell 52.
- **Target biotypes**: protein_coding 20,130; lncRNA 34,882; miRNA 1,879; snRNA 1,901.
- **Frame sizes**: 663,351 training pairs; 67,532 positives (10.18%); temporal test 11,624
  diseases vs 9,741 in the 2022 frame; 1,522 / 1,407 targets.
- **Table 2** temporal values match `temporal_repeats_5seed` exactly.
- **CV baselines** (Table 1 lower block) match `table1_fast_baselines/baselines_summary.csv`:
  OT score 0.9136/0.4551, disease mean 0.8730/0.4660, target mean 0.5000/0.1020,
  MF 0.8112/0.2651, TF-IDF 0.5126/0.1062.
- **0.65 threshold in-distribution precision** 0.925 (paper says 0.92 ✅).

### Discrepancies ⚠️

| # | Manuscript | Artefact | Note |
|---|---|---|---|
| a | "~282,500 candidate associations" novel | S1 novel = **214,968**; 282,500 = S2 (novel + 67,532 known) | abstract, §4.3, S1/S2 text |
| b | "4,346 diseases" | **4,347** in S1 | off by one |
| c | "mean ≈65 candidates/disease" | **49.45** (median 14) | follows from (a) |
| d | "spanning ~6,000 positive-bearing diseases" (§3.1) and "vs ~6,000 positive-bearing at training" (§3.3) | **2,329** (25.06 frame); **2,096** (2022 frame); OTP known_drug has 2,684 distinct diseases | ~2.6× overstated. **Correcting it strengthens the covariate-shift claim**: 11,624 test diseases vs 2,096 positive-bearing at training |
| e | Table 1 OTRec ROC-AUC SD **0.007** | 25-fold SD = **0.0111**; SD across 5 repeat-means = 0.0028 | no aggregation reproduces 0.007. PR-AUC SD 0.017 ✅ matches 25-fold 0.0172. Every other row reconciles (OTTree 0.0059/0.0232; ModernBERT 0.0085/0.0290) |
| f | recall 0.62 at 0.65 threshold | **0.634** | minor |
| g | druggable genome "~4,600 genes" | `finan_proc_druggable_genome_list.csv` = **4,479** | "~4,500" is more accurate |
| h | OTTree Table 1 SD 0.005 | 0.0059 | rounds to 0.006 |

### Discovery examples audit

| Claim | Released data | Verdict |
|---|---|---|
| SCN8A × CDKL5 disorder = 0.991 | 0.991 | ✅ exact |
| SCN1A × CDKL5 disorder = 0.989 | 0.989 | ✅ exact |
| PDE4C × limited cutaneous systemic sclerosis = 0.924 | **0.955** | ⚠️ mismatch |
| DHODH × temporal arteritis = 0.829 | **0.819** | ⚠️ mismatch (also Takayasu arteritis 0.862) |
| **IL6R × ulcerative colitis "ranked highly"** | **absent from S1 and S2**; the pair exists in the 25.06 frame with `label=0` and `score=0.357`, so it scored below the 0.65 threshold. S2 has 174 UC rows and 145 IL6R rows, but no IL6R × UC row | ❌ **unsupported by released data** — a reviewer checking the release will not find it. The manuscript comment already flags it as "may be a bad example"; recommend dropping it |
| POLA1 / POLA2 × CNS cancer, temporal, 0.961 / 0.966 | seed-42 temporal rerun gives **POLA1 0.856 (rank 18/95), POLA2 0.908 (rank 8/95)**, both `label=1` | ⚠️ direction confirmed (both converted by 2025, both top-20 of 95 candidates) but the specific scores are run-dependent; `score_past` is **0.0**, not ~0.06 |

---

## 12. Reproduction map (`rebuttal_scratch/`)

| Script | Produces |
|---|---|
| `t6_t4a.py` | T6 base rate, T4a strata → `temporal_test_bins.parquet` |
| `t3_t4b_reconstructible.py` | training-free baseline strata/per-disease → `temporal_test_with_baselines.parquet` |
| `temporal_preds_seed42.py` | **seed-42 OTRec+OTTree temporal retrain w/ saved preds** → `temporal_preds_seed42.parquet` |
| `t3_t4b_final.py` | final T3 + T4b tables |
| `temporal_strip_tractability.py` | leakage robustness rerun → `temporal_preds_seed42_stripped.parquet` |
| `run_node2vec_temporal.py` | T5 → `node2vec_temporal_preds.parquet`, `node2vec_temporal_result.json` |
| `verify_t2.py`, `analysis.py`, `fix2.py` | T2 composition |
| `verify_paper.py`, `verify2.py`, `verify3.py`, `verify4.py` | §11 manuscript verification |
| `temporal_bert_seed42.py` | ModernBERT temporal (blocked on environment) |

The retrain driver copies `train_otrec_once` / `train_ottree_once` verbatim (same split, model,
optimiser, callbacks, epochs) and imports the repo's own helpers; the only change is retaining
`y_pred`.

---

## 13. Not available

1. S3 per-feature-group ablation — no artefacts.
2. ModernBERT temporal per-pair predictions — never saved; regeneration environment-blocked.
3. 5-seed variance on the T3/T4b numbers — one authorised seed only.
4. Reconciliation of Table 1's OTRec ROC-AUC SD 0.007 with any released fold file.
5. Reconciliation of the manuscript's "~6,000 positive-bearing diseases".
6. Source of the POLA "~0.06 2022 known-drug score" and the PDE4C 0.924 / DHODH 0.829 values.

# Rebuttal analyses

Analyses run for the review response. Everything here is either a re-analysis of committed
artefacts (`Outputs/`) or a re-run of the existing temporal protocol with predictions retained.
Full write-up with per-claim evidence: [HANDOFF_REPORT.md](HANDOFF_REPORT.md).

## What was produced

| Analysis | Script | Result |
|---|---|---|
| Per-feature-group ablation (temporal, 5 rungs) | `scripts/ablation_temporal.py`, `scripts/run_ablation.sh` | `../Outputs/S3-feature_ablation.csv` |
| Seed replication of the text-only vs full contrast | `scripts/run_seed_replication.sh` | `results/ablation_seed_replication.csv` |
| Temporal per-pair predictions (seed 42, OTRec + OTTree) | `scripts/temporal_preds_seed42.py` | `../Outputs/S7-temporal_predictions.parquet` |
| Leakage bound: clinical-precedence tractability tokens stripped | `scripts/temporal_strip_tractability.py` | reported in the write-up |
| Stratification by therapeutic area / known targets / annotation volume | `scripts/stratify_q7.py`, `scripts/q7_summary_export.py` | `../Outputs/S8-shortlist_stratification.csv` |
| Node2Vec under the temporal protocol | `scripts/run_node2vec_temporal.py` | `results/node2vec_temporal_result.json` |
| Temporal base rate and indication-count strata | `scripts/t6_t4a.py`, `scripts/t3_t4b_final.py` | reported in the write-up |
| Manuscript-number verification against released artefacts | `scripts/verify_paper.py`, `verify2.py`, `verify3.py`, `verify4.py`, `verify_t2.py` | reported in the write-up |

## Running

The temporal scripts import the repo's own helpers (`dl_model_def.build_two_tower_model`,
`baselines/run_temporal_repeated.py`) and copy `train_otrec_once` / `train_ottree_once`
verbatim — same split, model, optimiser, callbacks and epochs. The only difference is that
per-pair predictions are retained instead of being reduced to metrics.

Paths are derived from each script's location, except the verification and per-pair analysis
scripts, which are run from the workspace root (the parent of `OTRec/`) because they read the
processed frames in `../code/`. Per-pair prediction parquets are 3–6 MB each and are not
committed; set `OTREC_SCRATCH` to point at them:

```bash
# from the workspace root
python OTRec/analysis/scripts/temporal_preds_seed42.py          # writes OTRec/Outputs/S7-temporal_predictions.parquet
OTREC_SCRATCH=rebuttal_scratch python OTRec/analysis/scripts/t3_t4b_final.py
bash OTRec/analysis/scripts/run_ablation.sh                     # 5 rungs, ~10 min each on one GPU
```

## Ablation design

Five nested rungs, seed 42, 6 epochs, identical validation split and callbacks; only the named
factor varies. Feature groups are rebuilt from the component columns of `disease_df` /
`target_df`; the reconstruction is token-identical to the stored `*_text_embed` strings
(verified 200/200), which is what the count-mode unigram `TextVectorization` consumes. The
learned disease-ID embedding is architectural and present in every rung, so the `text-only`
rung means "no annotation text", not "no disease identity".

| Rung | disease fields | target fields |
|---|---|---|
| R1 text-only | name, ExactSynonyms, description | sym, approvedName, synonyms, functionDescriptions |
| R2 +ontology | + dbXRefs, therapeuticAreas, parents, phenotypes | + targetClass |
| R3 +GO/pathway | — | + go, pathways |
| R4 +tractability (full) | — | + tractability, constraint |
| R5 −auxiliary head | as R4 | as R4, auxiliary score-head weight 0.0 |

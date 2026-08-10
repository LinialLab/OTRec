---
title: OTRec
emoji: 🦀
app_file: app.py
sdk: gradio
sdk_version: 6.13.0
python_version: "3.10"
license: mit
short_description: 'OTRec: prediction of druggable target–disease associations'
---

# OTRec

Interactive demo for OTRec, a deep learning recommender that ranks druggable-genome targets for a disease by predicted likelihood of clinical relevance. Open Targets and OTTree scores are shown for context where available.

This is a research screening tool, not clinical evidence. Most predicted candidates will not progress in development.
The app prewarms in the background on startup; after a Space wake-up the first query typically takes a few seconds.

## Use

The app has two tabs:

**Disease → Targets** (forward query):

1. Search a disease by name, synonym, or Open Targets / EFO ID.
2. Select it explicitly from the dropdown.
3. Inspect the ranked targets, including a tractability summary column. By default, packaged known hits are hidden so the list focuses on candidate novel pairs. The summary header links out to Open Targets and, when the ontology is supported, to OLS.
4. Read the coverage note below the summary. It explains packaged-label semantics and whether Open Targets / OTTree comparison columns are available for that disease.
5. Download either the full filtered ranking or the full druggable-genome ranking as CSV.

**Target → Diseases** (reverse query):

1. Search a gene by symbol or Ensembl ID.
2. Inspect the ranked diseases for that target and download the full disease ranking.

## Runtime layout

Expected Space layout:

```text
.
├── app.py
├── dl_model_def.py
├── vocab_io.py
├── runtime_data.py
├── prepare_runtime_artifacts.py
├── requirements.txt
└── data/
    └── proc/
        ├── df_learn_sub.parquet
        ├── disease_df.parquet
        ├── target_df.parquet
        ├── comparison_lookup.parquet
        ├── disease_metadata.csv
        ├── vocabs.json.gz
        └── embeddings.npz
```

`vocabs.json.gz` holds the training-time vocabularies and is required for
correct predictions (re-adapting from `df_learn_sub.parquet` permutes the
vocabulary order and silently degrades predictions to chance). Produce it with
`vocab_io.save_vocabularies(extract_vocabularies(model), path)` in the same
training session as `model.weights.h5`; always update the two together.

The model weights are downloaded at runtime.

## Free-tier behaviour and limits

Runs on Spaces `cpu-basic` (2 vCPU, 16 GB). Practical implications:

- **After ~48 h without visitors the Space sleeps.** The next visit pays a
  container boot (~1-2 min, outside the app's control) plus the app cold start
  (~5-15 s: 0.54 GB weights download + packaged embeddings; the app prewarms
  in the background so the first query is usually fast once the page loads).
- **One worker, queued requests**: simultaneous users are served in turn
  (warm queries take a few seconds each).
- Weights ship without optimizer state (1.63 GB -> 0.54 GB, bit-exact
  predictions). The full training checkpoint remains in the model repo's git
  history if training ever needs to resume. By default the app uses `OTREC_MODEL_REPO_ID=GrimSqueaker/OTRec` and `OTREC_MODEL_FILENAME=model.weights.h5`, but both can be overridden with environment variables for reviewer-safe or local deployments.

## Packaging comparison data

The reviewer filters and model-comparison columns rely on packaged artifacts in `data/proc`. To regenerate them from the repository outputs before pushing the Gradio folder to Spaces:

```bash
python prepare_runtime_artifacts.py
```

That script packages:

- `comparison_lookup.parquet`: disease-target rows with OTRec out-of-fold predictions, OTTree predictions, packaged label, and Open Targets score
- `disease_metadata.csv`: disease-level counts and orphan flags when available

If those files are missing, the app still runs, but comparison columns and known-vs-novel metadata will be unavailable. If a packaged comparison file is present but empty, the app falls back to repository outputs when they are available locally.

Packaged label caveat:

- `packaged known hit` means the disease-target pair has a positive packaged comparison label.
- `packaged novel` means the row is labeled `0` in the packaged comparison data.
- `unlabeled` means the pair is still rankable by OTRec, but it is outside the packaged comparison subset.

## Local smoke test

```bash
pip install -r requirements.txt
python prepare_runtime_artifacts.py
python - <<'PY'
import app
summary, note, table_html, csv_path, full_csv_path = app.recommend_targets("DOID_0050890", top_k=10)
print(summary.splitlines()[0])
print("forward table:", "Open Targets" in table_html, csv_path, full_csv_path)
target_id, message = app._resolve_target_id("TNF", None)
summary, table_html, csv_path = app.recommend_diseases(target_id, top_k=10)
print(summary.splitlines()[0])
print("reverse table:", "Open Targets" in table_html, csv_path)
PY
```

## Reviewer caveats surfaced in the UI

- Predictions are screening scores, not clinical evidence
- Open Targets and OTTree columns are shown only where the packaged comparison data covers a pair
- The CSV download contains the full filtered ranking, not just the visible rows


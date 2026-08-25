# OTRec

OTRec (Open Targets Recommender) ranks druggable genes for a disease using a two-tower recommender trained on Open Targets data. It predicts disease–target prioritization rather than compound–protein binding or drug–target interaction. This repository contains the code, released predictions, and the interactive demo for the OTRec manuscript.

## Try OTRec

* **Web demo:** explore disease → target and target → disease rankings in the [OTRec Hugging Face app](https://huggingface.co/spaces/GrimSqueaker/OTRec).
* **Released predictions:** `Outputs/S1-DL_novel_predictions.csv` (novel candidates) and `Outputs/S2-DL_novel+known_candidates.csv` (novel + known).
* **Simple local query**, no model install needed:

```bash
pip install pandas
python examples/query_predictions.py --disease obesity
python examples/query_predictions.py --target GIPR
```

See [examples/README.md](examples/README.md).

## Released paper data

All manuscript data files live in `Outputs/`:

| File | Contents |
|---|---|
| `S1-DL_novel_predictions.csv` | 214,968 novel disease–target predictions (score ≥ 0.65; 4,347 diseases) |
| `S2-DL_novel+known_candidates.csv` | S1 plus 67,532 known clinical associations (282,500 rows) |
| `S3-feature_ablation.csv` | Per-feature-group ablation (temporal setting) |
| `S5-ottree_hyperparameter_sweep.csv` | CatBoost baseline hyperparameter sweep |
| `S7-temporal_predictions.parquet` | Per-pair temporal predictions behind Tables 4, 5, and S8 |
| `S8-shortlist_stratification.csv` | Shortlist metrics by therapeutic area / known targets / annotation volume |
| `S9-gene_family_overlap.csv` | Per-family gene-family overlap statistics (25.12 model) |
| `Table 1/2/3 ... .csv` | Main benchmark tables |
| `InterFeat_reranked_candidates/` | InterFeat annotations for selected candidates |
| `CV_*`, `temporal_*` directories | Per-fold and per-run metrics behind Tables 1–3 |

**Versions:** the manuscript benchmarks and the S1/S2 prediction files are frozen to the paper analysis (Open Targets 22.02 and 25.06). The interactive web app has since been refreshed to Open Targets 25.12 (`analysis/retrain_2512/`), so live app scores can differ from S1/S2.

## Citation

> Ofer D., Linial M. *OTRec: A Deep Learning Recommender for Druggable Disease–Target Prioritization.* bioRxiv 2025.12.21.695803. doi: [10.64898/2025.12.21.695803](https://doi.org/10.64898/2025.12.21.695803)

---

## Reproducing the analysis

### Repository structure

```text
OTRec/
├── 0-OT-PreProcess_Recc.ipynb  # Data preprocessing and feature engineering
├── 1-Train-DL-Retriever.ipynb  # Trains the two-tower model; writes S1/S2
├── 2-Temporal-Eval.ipynb       # Temporal split validation (2022 vs 2025)
├── dl_model_def.py             # Keras two-tower model definition
├── utils.py                    # Data loading and metric helpers
├── examples/                   # Query the released predictions with pandas
├── gradio/                     # Standalone interactive web demo
├── analysis/                   # Supplementary and rebuttal analyses
│   ├── scripts/                # Reproduction scripts (S3, S5, S7, S8, ...)
│   └── retrain_2512/           # Open Targets 25.12 deployment refresh
├── baselines/                  # Baseline reproduction (run_baselines.py, run_temporal_repeated.py)
└── Outputs/                    # Released predictions, tables, figures, metrics
```

### Data prerequisites

The repository does not contain the raw Open Targets Platform (OTP) data. To retrain or re-evaluate, download the parquet exports from [Open Targets Data Downloads](https://platform.opentargets.org/downloads):

* Core: `associationByOverallDirect` (rename folder to `association_overall_direct`), `associationByDatatypeDirect` (→ `association_by_datatype_direct`), `target`, `disease`, `knownDrugsAggregated` (→ `known_drug`)
* Features: `targetPrioritisation` (→ `target_prioritisation`), `targetEssentiality` (→ `target_essentiality`), `diseasePhenotypes` (→ `disease_phenotype`), optional `mousePhenotypes` (→ `mouse_phenotype`)
* Temporal validation additionally needs the Release **22.02** core datasets.

Expected layout (siblings of this repository; set `Config.DATA_DIR` in the notebooks to change it):

```text
project_root/
├── OTRec/                 # this repository
└── data/
    ├── opentargets/       # current release (paper: 25.06)
    │   ├── association_overall_direct/ ... mouse_phenotype/
    └── historical_ot/
        └── 22_02/         # historical release for the temporal split
```

### Environment

Python 3.10+:

```bash
pip install tensorflow pandas scikit-learn catboost gradio pyarrow fastparquet
```

### Run order

1. `0-OT-PreProcess_Recc.ipynb` — filters raw data, applies the druggable-genome constraint, builds the training frame, trains the CatBoost baseline.
2. `1-Train-DL-Retriever.ipynb` — trains the two-tower model; writes weights and the S1/S2 prediction files.
3. `2-Temporal-Eval.ipynb` — temporal validation: 2022 model vs 2025 clinical outcomes.

Baselines: `baselines/run_baselines.py` (CV) and `baselines/run_temporal_repeated.py` (temporal, five runs). Supplementary analyses: see [analysis/README.md](analysis/README.md).

### Interactive demo (local)

```bash
cd gradio
python app.py
```

The app downloads model weights at runtime; see [gradio/README.md](gradio/README.md) for details and caveats.

## Key results

* **Temporal validation** (predict 2025 clinical-trial entries from 2022 data; mean of five runs): OTRec ROC-AUC **0.872** / PR-AUC **0.288** vs Open Targets score 0.559 / 0.082.
* **Target-disjoint 5×5 CV** (targets unseen in training): OTRec ROC-AUC **0.950** / PR-AUC **0.844**.

## License

MIT. Open Targets Platform data are CC0 1.0; other evidence sources retain their own licenses.

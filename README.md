
# OTRec: A deep learning recommender for prospective druggable disease–target associations

**OTRec** (Open Targets Recommender) ranks druggable genes for a disease using a two-tower neural architecture trained on Open Targets Platform data. The repository contains the notebooks, model code, outputs, and the interactive Gradio app used for manuscript figures and reviewer-facing artifacts.

For blind-review builds, public identity-bearing URLs are intentionally omitted from this README. The supplementary materials can point reviewers to the appropriate snapshot or deployment.

If you want to retrain or reevaluate the model, you will need the Open Targets parquet exports and the historical 22.02 release described below.

---

## 📂 Repository Structure

```text
OTRec/
├── 2-Temporal-Eval.ipynb       # Main evaluation notebook: Temporal split (2022 vs 2025)
├── 1-Train-DL-Retriever.ipynb  # Training loop for the Deep Learning Recommender
├── 0-OT-PreProcess_Recc.ipynb  # Data preprocessing and feature engineering
├── dl_model_def.py             # Keras definition of the Two-Tower model architecture
├── utils.py                    # Helper functions for data loading and metrics
├── gradio/                     # Standalone interactive web demo
│   ├── app.py                  # Gradio application entry point
│   └── ...                     # App-specific assets
└── Outputs/                    # Saved model weights, logs, and evaluation metrics
```

---


### Important: Data & Prerequisites

**Note:** This repository **does not** contain the raw Open Targets Platform (OTP) data due to file size limitations. You must download the data separately to run the training or evaluation scripts.

### 1. Download Open Targets Data

The code expects raw Parquet files from Open Targets in a specific directory structure relative to this repo (typically `../data/opentargets/`).

You will need to download the following datasets via `wget` or rsync from the [Open Targets Data Downloads](https://platform.opentargets.org/downloads). Specifically, the code requires:

* **Core Datasets:**
* `associationByOverallDirect` (Note: Rename folder to `association_overall_direct` after download)
* `associationByDatatypeDirect` (Note: Rename folder to `association_by_datatype_direct`)
* `target`
* `disease`
* `knownDrugsAggregated` (Note: Rename folder to `known_drug`)


* **Additional Feature Datasets:**
* `targetPrioritisation` (Note: Rename folder to `target_prioritisation`)
* `targetEssentiality` (Note: Rename folder to `target_essentiality`)
* `diseasePhenotypes` (Note: Rename folder to `disease_phenotype`)
* optional: `mousePhenotypes` (Note: Rename folder to `mouse_phenotype`)


* **Historical Data (For Temporal Validation):**
* For the temporal split (2022 vs 2025), you also need the Open Targets Release **22.02** versions of the core datasets above, placed in `../data/historical_ot/22_02/`.


### 2. Directory Structure Expectation

The scripts assume the following directory layout by default. If you use a different path, please update `Config.DATA_DIR` in the notebooks/scripts.

```text
project_root/
├── OTRec/                 # This repository
│   ├── 0-OT-PreProcess_Recc.ipynb
│   └── ...
└── data/                  # Data folder (sibling to OTRec)
    ├── opentargets/       # Current Data (2025.xx)
    │   ├── association_overall_direct/
    │   ├── association_by_datatype_direct/
    │   ├── target/
    │   ├── disease/
    │   ├── known_drug/
    │   ├── target_prioritisation/
    │   ├── target_essentiality/
    │   ├── disease_phenotype/
    │   └── mouse_phenotype/
    └── historical_ot/     # Historical Data (22.02)
        └── 22_02/
            ├── association_overall_direct/
            └── ... (other core datasets)

```

### 3. Environment

Install the required dependencies (Python 3.10+ recommended). We use tensorflow and keras:

```bash
pip install tensorflow pandas scikit-learn catboost gradio pyarrow fastparquet

```

---

##  Usage

### Running the Analysis

The notebooks are numbered sequentially:

1. **`0-OT-PreProcess_Recc.ipynb`**: Filters raw data, applies "druggable genome" constraints, and generates the training dataset. Trains the CatBoost baseline.
2. **`1-Train-DL-Retriever.ipynb`**: Trains the Two-Tower model . Saves model weights to `output/`.
3. **`2-Temporal-Eval.ipynb`**: Performs the rigorous temporal split validation, comparing 2022 predictions against 2025 clinical outcomes.

### Interactive Demo (Gradio)

We provide a standalone web interface to explore predictions.

```bash
cd gradio
python app.py

```

The Gradio app downloads model weights at runtime and precomputes embeddings on first use. See [gradio/README.md](gradio/README.md) for packaged comparison caveats, environment variables, and reviewer-facing behavior.

---

## 📊 Key Results

* **Temporal validation:** Predicting 2025 clinical-trial entries from 2022 data.
* **OTRec:** ROC-AUC **0.863**, PR-AUC **0.303**
* **Open Targets Score:** ROC-AUC **0.558**, PR-AUC **0.082**


* **Target-Disjoint Generalization:** 5-fold CV on targets unseen during training.
* **OTRec:** ROC-AUC **0.950**, PR-AUC **0.844**



---

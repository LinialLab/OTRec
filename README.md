
# OTRec: A deep learning recommender for prospective druggable disease–target associations

**OTRec** (Open Targets Recommender) is a deep learning recommender system designed to prioritize druggable targets for repurposing and novel drug discovery, based on Open Targets Platform (https://platform.opentargets.org/). Unlike retrospective evidence aggregation, OTRec uses a **Two-Tower Neural Network** architecture to learn latent representations from disease and target text descriptions, ontologies, and biological annotations.

Given a disease, the model will reccommend targets (genes), out of the druggable genome (~4,600 targets). The model was trained to predict for a given disease, which targets will enter clinical trials. 

This repository contains the source code, analysis notebooks, outputs and the interactive Gradio application code for the paper:
> **OTRec: prospective prediction of druggable target–disease associations via deep learning** > *Dan Ofer and Michal Linial* 

You can view the app and get recommendations for any disease here: https://huggingface.co/spaces/GrimSqueaker/OTRec
Trained model weights available at: https://huggingface.co/GrimSqueaker/OTRec

If you want to train the model yourself, you'll need to download the Open targets data (~27GB) and update the folder structure (https://platform.opentargets.org/downloads). 

---

## 📂 Repository Structure

```text
OTRec/
├── 2-Temporal-Eval.ipynb       # Main evaluation notebook: Temporal split (2022 vs 2025)
├── 1-Train-DL-Retriever.ipynb  # Training loop for the Deep Learning Recommender
├── 0-OT-PreProcess_Recc.ipynb  # Data preprocessing and feature engineering
├── dl_model_def.py             # Keras definition of the Two-Tower model architecture
├── utils.py                    # Helper functions for data loading and metrics
├── gradio_app/                 # Standalone interactive web demo
│   ├── app.py                  # Gradio application entry point
│   └── ...                     # App-specific assets
└── output/                     # Saved model weights, logs, and evaluation metrics


---

Here is a concise, professional, and helpful `README.md` for your GitHub repository. It addresses the specific data "gotchas," explains the folder structure, and provides clear instructions for users and editors.

### README.md content

```markdown
# OTRec: Prospective Prediction of Druggable Disease–Target Associations

**OTRec** (Open Targets Recommender) is a deep learning recommender system designed to prioritize druggable targets for repurposing and novel drug discovery. Unlike retrospective evidence aggregation, OTRec uses a **Two-Tower Neural Network** architecture to learn latent representations from disease and target text descriptions, ontologies, and biological annotations.

This repository contains the source code, analysis notebooks, and an interactive Gradio application for the paper:
> **OTRec: prospective prediction of druggable target–disease associations via deep learning** > *Dan Ofer and Michal Linial (Hebrew University of Jerusalem)* > [Link to paper / DOI once available]

---

## 📂 Repository Structure

```text
OTRec/
├── 2-Temporal-Eval.ipynb       # Main evaluation notebook: Temporal split (2022 vs 2025)
├── 1-Train-DL-Retriever.ipynb  # Training loop for the Deep Learning Recommender
├── 0-OT-PreProcess_Recc.ipynb  # Data preprocessing and feature engineering
├── dl_model_def.py             # Keras definition of the Two-Tower model architecture
├── utils.py                    # Helper functions for data loading and metrics
├── gradio_app/                 # Standalone interactive web demo
│   ├── app.py                  # Gradio application entry point
│   └── ...                     # App-specific assets
└── output/                     # Saved model weights, logs, and evaluation metrics

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
cd gradio_app
python app.py

```

*Note: The Gradio app requires the pre-trained model weights and pre-computed embeddings to be present in its directory or accessible path.*

---

## 📊 Key Results

* **Prospective Validation:** Predicting 2025 clinical trials from 2022 data.
* **OTRec:** ROC-AUC **0.865**
* **Open Targets Score (Baseline):** ROC-AUC 0.56 (Fails to predict future utility)


* **Target-Disjoint Generalization:** 5-fold CV on targets unseen during training.
* **OTRec:** ROC-AUC **0.960** (for ensemble model, 94.9 recorded for single split)



---

## ✉️ Contact

For questions regarding the code or data:
**Dan Ofer**

If you use us, please cite us! 
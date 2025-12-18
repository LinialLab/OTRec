
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
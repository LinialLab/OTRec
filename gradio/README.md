---
title: OTRec
app_file: app.py
sdk: gradio
sdk_version: 6.0.1
---
# Disease–Target Recommender (Open Targets)

This Space exposes a two-tower recommender model trained on Open Targets–derived
disease–target data. Given a **disease ID** (matching the `diseaseId` column from
the preprocessed data), it returns a ranked list of predicted **target IDs**.

The backend is a TensorFlow / Keras model with:
- A **query tower** for diseases (disease text + disease ID embedding)
- A **key tower** for targets (target text only)
- Cosine similarity between disease and target embeddings

All candidate target embeddings are currently precomputed at startup for fast inference. (can drop)

---

## Files and structure

Expected repo layout:

```text
.
├── app.py
├── requirements.txt
├── model.weights.h5
└── data/
    └── proc/
        ├── disease_df.parquet
        └── target_df.parquet
        └── df_learn.parquet


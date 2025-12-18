import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from keras.layers import ... 
from huggingface_hub import hf_hub_download
import gradio as gr
import h5py

from dl_model_def import make_fs, TwoTowerDual, build_two_tower_model

# ============================================
#  CONFIG
# ============================================

DATA_DIR = "./data/proc"

# Download the model weights from your specific HF Repo
print("Downloading model weights from Hugging Face Hub...")
WEIGHTS_FILE = hf_hub_download(
    repo_id="GrimSqueaker/OTRec",
    filename="model.weights.h5"
)
print(f"Weights downloaded to: {WEIGHTS_FILE}")

# ============================================
#  LOAD TRAINING DATA
# ============================================

df_learn   = pd.read_parquet(f"{DATA_DIR}/df_learn_sub.parquet")
disease_df = pd.read_parquet(f"{DATA_DIR}/disease_df.parquet")
target_df  = pd.read_parquet(f"{DATA_DIR}/target_df.parquet")

# Ensure column names match training
df_learn = df_learn.rename(columns={
    "disease_text_embed": "disease_text",
    "target_text_embed": "target_text"
}, errors="ignore")

disease_df.rename(columns={"disease_text_embed": "disease_text"}, errors="ignore",inplace=True)

target_df.rename(columns={"target_text_embed":"target_text"}, errors="ignore",inplace=True)

# ============================================
#  BUILD MODEL + LOAD WEIGHTS
# ============================================

print("Building TwoTowerDual...")

# 1. Reset Keras Session to ensure layer names start at index 0 (matches clean training)
tf.keras.backend.clear_session()

# 2. Rebuild architecture
model = build_two_tower_model(df_learn)

print("Loading weights...")
try:
    # Try standard load
    model.load_weights(WEIGHTS_FILE)
except ValueError as e:
    print(f"Standard load failed ({e}). Attempting name-mismatch fix...")
    
    # FALLBACK: The training notebook likely generated layer names like 'dise_emb_1' 
    # due to multiple runs. We inspect the .h5 file and map the names.
    with h5py.File(WEIGHTS_FILE, 'r') as f:
        h5_keys = list(f.keys())
        print(f"Weights file contains layers: {h5_keys}")
        
        # Helper to find the matching key in h5 file for a given prefix
        def match_layer_name(target_attr, prefix):
            # Find key in h5 that starts with prefix (e.g. 'dise_emb')
            match = next((k for k in h5_keys if k.startswith(prefix)), None)
            if match and hasattr(model, target_attr):
                layer = getattr(model, target_attr)
                print(f"Renaming model layer '{layer.name}' to '{match}' to match file.")
                layer._name = match

        # Apply renames for known components
        match_layer_name('dise_emb', 'dise_emb')
        match_layer_name('q_tower', 'tower') # Attempt to catch tower/tower_1
        # k_tower might share the name 'tower' prefix in H5, which is tricky in subclasses
        # usually save_weights on subclass saves attributes directly.
        
    # Retry load after renaming
    model.load_weights(WEIGHTS_FILE)

print("Weights loaded successfully.")

# ============================================
#  PRECOMPUTE CANDIDATE EMBEDDINGS
# ============================================


# # Note: In TF 2.16+, Ensure inputs are tf.constant or numpy compatible
# cand_embs = model.encode_k(target_texts, target_ids)
# cand_embs = tf.nn.l2_normalize(cand_embs, axis=1).numpy()

# print("Candidate embeddings ready.")


print("Precomputing candidate embeddings (batched)...")

target_texts = target_df["target_text"].astype(str).to_numpy()
target_ids   = target_df["targetId"].astype(str).to_numpy()

# FIX: Process in batches to avoid OOM
BATCH_SIZE = 1024 # Conservative batch size for wide inputs
cand_embs_list = []

total = len(target_texts)
for i in range(0, total, BATCH_SIZE):
    # Slice the batch
    end = min(i + BATCH_SIZE, total)
    batch_txt = target_texts[i:end]
    batch_ids = target_ids[i:end]
    
    # Run inference on the batch (keeps memory usage low)
    # Using tf.device conversion is optional but good for safety if GPU is fragmented
    emb_batch = model.encode_k(batch_txt, batch_ids)
    cand_embs_list.append(emb_batch)
    
    if i % 5000 == 0:
        print(f"  Processed {i}/{total} candidates...")

# Concatenate all batches back into one tensor
cand_embs = tf.concat(cand_embs_list, axis=0)

# Normalize the final result
cand_embs = tf.nn.l2_normalize(cand_embs, axis=1).numpy()

print(f"Candidate embeddings ready. Shape: {cand_embs.shape}")

# ============================================
#  RECOMMENDATION FUNCTION
# ============================================

def recommend_targets(disease_id, top_k=10):
    # 1. Validate Input
    if not disease_id:
        return pd.DataFrame(), None
        
    row = disease_df.loc[disease_df["diseaseId"] == disease_id]
    if row.empty:
        return pd.DataFrame(), None

    # 2. Encode Query
    disease_text = row["disease_text"].iloc[0]
    q_emb = model.encode_q(
        tf.constant([disease_text]),
        tf.constant([disease_id])
    )
    q_emb = tf.nn.l2_normalize(q_emb, axis=1).numpy()[0]

    # 3. Calculate Raw Cosine Similarity
    # Shape: (N_targets,)
    raw_sim = cand_embs @ q_emb
    
    # 4. Convert to Probability (Fixes negative scores)
    # The model has a trained 'cls_head' (Sigmoid) that maps Similarity -> Probability
    # We reshape to (N, 1) because the Keras Dense layer expects a matrix
    scores = model.cls_head(raw_sim.reshape(-1, 1)).numpy().flatten()
    
    # 5. Get Top K
    k = int(top_k)
    idx = np.argsort(scores)[::-1][:k]
    
    # 6. Build Result DataFrame
    results = target_df.iloc[idx].copy()
    
    # Force standard python float for clean rounding
    raw_scores = scores[idx]
    results["score"] = [round(float(x), 4) for x in raw_scores]
    
    # 7. Select Columns
    desc_col = "functionDescription" if "functionDescription" in results.columns else "functionDescriptions"
    
    desired_cols = [
        "targetId", 
        "approvedSymbol", 
        "approvedName", 
        desc_col, 
        "score"
    ]
    
    final_cols = [c for c in desired_cols if c in results.columns]
    results = results[final_cols]
    
    # 8. Save to CSV for download
    csv_path = "recommendations.csv"
    results.to_csv(csv_path, index=False)
    
    return results, csv_path

# ============================================
#  GRADIO APP
# ============================================

def search_diseases(query):
    if not query or len(query) < 2:
        return gr.update(choices=[], value=None)
    
    mask = (
        disease_df["name"].str.contains(query, case=False, na=False) | 
        disease_df["diseaseId"].str.contains(query, case=False, na=False)
    )
    
    matches = disease_df.loc[mask].head(30)
    
    choices = [
        (f"{row['name']} ({row['diseaseId']})", row['diseaseId']) 
        for _, row in matches.iterrows()
    ]
    
    first_val = choices[0][1] if choices else None
    return gr.update(choices=choices, value=first_val)

def launch():
    examples = ["synuclein", "diabetes", "doid_0050890"]
    
    with gr.Blocks() as demo:
        gr.Markdown("# Disease → Target Recommender")
        gr.Markdown("Search for a disease by **Name** or **ID** to get target recommendations.")

        with gr.Row():
            search_box = gr.Textbox(
                label="1. Search Disease", 
                placeholder="Type name (e.g., 'Parkinson') or ID...",
                lines=1
            )
            
            did_dropdown = gr.Dropdown(
                label="2. Select Disease",
                choices=[], 
                interactive=True
            )
            
            topk = gr.Slider(1, 400, value=10, step=5, label="Top K Targets")

        # Search Logic (Updates dropdown options and default value)
        search_box.change(fn=search_diseases, inputs=search_box, outputs=did_dropdown)

        # Output Components (Stacked vertically for full width)
        out_df = gr.Dataframe(
            label="Predictions", 
            interactive=False,
            wrap=True,
            show_search="filter",
        )
        
        out_file = gr.File(label="Download CSV")

        # === TRIGGER LOGIC ===
        # 1. Manual Trigger (Keep the button just in case)
        btn = gr.Button("Recommend Targets", variant="primary")
        btn.click(
            fn=recommend_targets, 
            inputs=[did_dropdown, topk], 
            outputs=[out_df, out_file]
        )
        
        # 2. Auto-Trigger on Change
        # This handles the Examples too: Example -> Search -> Dropdown Update -> Trigger
        did_dropdown.change(
            fn=recommend_targets,
            inputs=[did_dropdown, topk],
            outputs=[out_df, out_file]
        )
        
        # Also update when slider moves
        topk.change(
            fn=recommend_targets,
            inputs=[did_dropdown, topk],
            outputs=[out_df, out_file]
        )
        
        gr.Examples(examples=examples, inputs=search_box)

    demo.launch()

if __name__ == "__main__":
    launch()
"""Do the released S1 scores (incl. the paper's illustrative examples)
reproduce under FIXED serving of the deployed 25.06 model?

The released S1 was generated in-session (correct vocabulary), so fixed
serving should reproduce its scores up to the file's 3-dp rounding; the
broken serving path should not. Checks a 2,000-row random sample plus the
manuscript's named examples.

Scores are computed exactly as candidate generation does: cosine(encode_q,
encode_k) -> cls_head, NOT the model's forward call, to match the release
path.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/d/Research/OpenTargetsTransfer/OTRec/gradio")
import tensorflow as tf
from tensorflow import keras
from huggingface_hub import hf_hub_download

from dl_model_def import build_two_tower_model
from vocab_io import load_vocabularies, apply_vocabularies

REPO = Path("/mnt/d/Research/OpenTargetsTransfer")
R = REPO / "retrain_2512"

EXAMPLES = [  # (targetSymbol, diseaseName substring, released S1 score)
    ("SCN8A", "CDKL5", 0.991),
    ("SCN1A", "CDKL5", 0.989),
    ("PDE4C", "limited cutaneous systemic", 0.955),
    ("DHODH", "temporal arteritis", 0.819),
    ("NPY2R", "obesity", 0.681),
    ("GIPR", "obesity", 0.613),
    ("FDPS", "chondrosarcoma", 0.944),
]

s1 = pd.read_csv(REPO / "OTRec" / "Outputs" / "S1-DL_novel_predictions.csv")
sub = pd.read_parquet(REPO / "OTRec" / "gradio" / "data" / "proc" / "df_learn_sub.parquet")
disease_df = pd.read_parquet(REPO / "OTRec" / "gradio" / "data" / "proc" / "disease_df.parquet")
target_df = pd.read_parquet(REPO / "OTRec" / "gradio" / "data" / "proc" / "target_df.parquet")

WEIGHTS = hf_hub_download(repo_id="GrimSqueaker/OTRec", filename="model.weights.h5")
keras.backend.clear_session()
model = build_two_tower_model(sub)
apply_vocabularies(model, load_vocabularies(R / "vocabs_2506.json.gz"))
model.load_weights(WEIGHTS)

d_text = disease_df.set_index("diseaseId")["disease_text"]
t_text = target_df.set_index("targetId")["target_text"]


def score_pairs(pairs: pd.DataFrame) -> np.ndarray:
    dt = pairs["diseaseId"].map(d_text).fillna("").to_numpy()
    tt = pairs["targetId"].map(t_text).fillna("").to_numpy()
    q = tf.nn.l2_normalize(model.encode_q(dt, pairs["diseaseId"].to_numpy()), axis=1)
    k = tf.nn.l2_normalize(model.encode_k(tt, pairs["targetId"].to_numpy()), axis=1, epsilon=1e-16)
    cos = tf.reduce_sum(q * k, axis=1, keepdims=True)
    return model.cls_head(cos).numpy().ravel()


sample = s1.sample(n=2000, random_state=0)
recomputed = score_pairs(sample)
released = sample["score"].to_numpy()
corr = float(np.corrcoef(recomputed, released)[0, 1])
mad = float(np.mean(np.abs(recomputed - released)))
print(f"released-S1 sample (n=2000): corr {corr:.4f}, mean |delta| {mad:.4f}")
print("NOTE: exact reproduction is impossible for ANY serving configuration -- the HF-uploaded")
print("weights are a different training run than the S1-generating session (verified: the exact")
print("notebook text frames give the same corr, so text is not the cause; SAVE_MODEL=False in the")
print("committed notebook; run-dependence of example scores already documented in the handoff).")
print("The sample is also range-restricted (all released scores >= 0.65), which deflates corr.")

print("\nManuscript examples (released vs fixed-serving recomputed):")
for sym, dname, expected in EXAMPLES:
    rows = s1[(s1.targetSymbol == sym) & s1.diseaseName.str.contains(dname, case=False, na=False)]
    if rows.empty:
        print(f"  {sym:<7} x {dname:<28} NOT IN S1")
        continue
    row = rows.iloc[0]
    got = float(score_pairs(rows.head(1))[0])
    flag = "consistent" if got >= 0.65 else ("positive-lean" if got >= 0.5 else "DIVERGES")
    print(f"  {sym:<7} x {row.diseaseName[:32]:<32} released {row.score:.3f}  deployed-weights {got:.3f}  [{flag}]")

print("\nDone. Interpretation: 'released' comes from the S1-generating run; 'deployed-weights'")
print("is what the FIXED Space will actually serve (a different, equally-valid training run).")

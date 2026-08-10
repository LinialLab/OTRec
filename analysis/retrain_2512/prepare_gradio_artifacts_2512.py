"""Package a matched Space artifact set from the 25.12 retrain.

Reuses OTRec/gradio/prepare_runtime_artifacts.py's build_disease_metadata
logic verbatim (imported, not reimplemented), pointed at the 2512 comparison
lookup and S2b candidates instead of the 25.06 originals.

Writes into retrain_2512/gradio_artifacts/ -- a full replacement set for
OTRec/gradio/data/proc/, kept separate until the user approves the push.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/mnt/d/Research/OpenTargetsTransfer/OTRec/gradio")
from prepare_runtime_artifacts import build_disease_metadata  # reused verbatim

REPO = Path("/mnt/d/Research/OpenTargetsTransfer")
R = REPO / "retrain_2512"
PKG = R / "gradio_artifacts"
PKG.mkdir(exist_ok=True)

df_learn_full = pd.read_parquet(R / "df_learn_sub_2512.parquet")
disease_df = pd.read_parquet(R / "disease_df_full_2512.parquet")
target_df = pd.read_parquet(R / "target_df_full_2512.parquet")
comparison_lookup = pd.read_parquet(R / "Outputs" / "comparison_lookup_2512.parquet")

# Ship a small dedup cover, NOT the 1.28 GB full frame: with the saved
# vocabularies (vocabs.json.gz, applied by app.py before load_weights) the
# bootstrap frame only supplies column structure, so the cover is safe --
# proven by test_vocab_fix.py, which bootstraps from exactly this cover shape
# and reproduces the full-frame predictions bit-for-bit.
by_disease = df_learn_full.drop_duplicates(subset=["diseaseId"], keep="first")
by_target = df_learn_full.drop_duplicates(subset=["targetId"], keep="first")
df_learn_sub = (pd.concat([by_disease, by_target], ignore_index=True)
                .drop_duplicates(subset=["diseaseId", "targetId"]).reset_index(drop=True))

df_learn_sub.to_parquet(PKG / "df_learn_sub.parquet")
import shutil
shutil.copy(R / "vocabs_2512.json.gz", PKG / "vocabs.json.gz")
disease_df.to_parquet(PKG / "disease_df.parquet")
target_df.to_parquet(PKG / "target_df.parquet")
comparison_lookup.to_parquet(PKG / "comparison_lookup.parquet")

# build_disease_metadata reads CANDIDATES_PATH from module-level constants;
# monkeypatch the path it resolves rather than editing the shared module.
import prepare_runtime_artifacts as pra
pra.CANDIDATES_PATH = R / "Outputs" / "S2b-DL_novel+known_candidates_2512.csv"
disease_metadata = build_disease_metadata(comparison_lookup)
disease_metadata.to_csv(PKG / "disease_metadata.csv", index=False, encoding="utf-8")

print(f"packaged Space artifact set in {PKG}:")
for f in sorted(PKG.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")

print("\nExpected by OTRec/gradio/app.py + runtime_data.py:")
print("  df_learn_sub.parquet, disease_df.parquet, target_df.parquet,")
print("  comparison_lookup.parquet, disease_metadata.csv  -- all present.")
print("\nWeights file (uploaded separately to the model repo):", R / "model.weights.h5")

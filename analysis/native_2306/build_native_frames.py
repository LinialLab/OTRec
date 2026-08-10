"""Build all frames for the fully release-native temporal replication: 23.06 -> 25.12.

SEPARATE BRANCH from the 22.02 leak analysis. Everything here lives in
rebuttal_scratch/native_2306/ with native2306_-prefixed outputs. The headline
22.02 -> 25.06 experiment and its artefacts are untouched.

Construction (ports the exact recipes; nothing new invented):
- Text frames: build_disease_text / build_target_text from
  rebuttal_scratch/build_2202_features.py (port fidelity 1.0000 pinned by
  test_build_2202_features.py), fed the 23.06 raw tables (same pre-25.03
  camelCase era as 22.02).
- Label frames (train 23.06 and eval 25.12): make_target_data recipe from
  code/2-Temporal-Eval.ipynb cells 6-7 —
    assoc pairs, drop diseases whose therapeutic area is in
    {measurement, phenotype, biological process, cell proliferation disorder},
    restrict to targets with any known_drug phase (non-NA),
    label = 1 iff (targetId, diseaseId) in known_drug (phase non-NA),
    keep `score` (auxiliary target), clean diseaseId via split('/')[-1],
    then filter train frame to ids present in the (native) annotation frames.

Eval = 25.12, NOT 26.03: Release 26.03 abolished the known_drug dataset
(replaced by the new clinical-mining tables), so 26.03 labels would have
different semantics. 25.12 is the newest release with the paper's label
definition. Gap: 2.5 years.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/d/Research/OpenTargetsTransfer/rebuttal_scratch")
from build_2202_features import build_disease_text, build_target_text

DATA = Path("/mnt/d/Research/OpenTargetsTransfer/data/historical_ot")
OUT = Path("/mnt/d/Research/OpenTargetsTransfer/rebuttal_scratch/native_2306")

TRAIN_RAW = DATA / "23_06"        # pre-25.03 era: camelCase plural tables
EVAL_RAW = DATA / "25_12"         # post-25.03 era: snake_case singular tables

TABLES_2306 = {"diseases": "diseases", "targets": "targets", "diseaseToPhenotype": "diseaseToPhenotype"}
LABELS_TO_REMOVE = ["measurement", "phenotype", "biological process", "cell proliferation disorder"]


def build_label_frame(assoc_dir, kd_dir, disease_dir, tag):
    assoc = pd.read_parquet(assoc_dir)
    if "timeseries" in assoc.columns:
        print(f"[{tag}] dropping timeseries column (backfilled history)")
        assoc = assoc.drop(columns=["timeseries"])
    assert "score" in assoc.columns, f"[{tag}] no `score` column: {assoc.columns.tolist()}"
    assoc = assoc[["diseaseId", "targetId", "score"]]

    kd = pd.read_parquet(kd_dir, columns=["targetId", "diseaseId", "phase"])
    kd = kd.dropna(subset=["phase"])

    dis = pd.read_parquet(disease_dir, columns=["id", "name", "therapeuticAreas"])
    ta_map = dis.set_index("id")["name"].to_dict()
    exploded = dis[["id", "therapeuticAreas"]].explode("therapeuticAreas").dropna()
    exploded["ta_name"] = exploded["therapeuticAreas"].map(ta_map)
    ids_to_remove = exploded[exploded["ta_name"].isin(LABELS_TO_REMOVE)]["id"].unique()
    print(f"[{tag}] diseases dropped by TA filter: {len(ids_to_remove):,}")

    assoc = assoc[~assoc["diseaseId"].isin(ids_to_remove)]
    validated_targets = kd["targetId"].unique()
    frame = assoc[assoc["targetId"].isin(validated_targets)].copy()

    pairs = kd[["targetId", "diseaseId"]].drop_duplicates()
    pairs["label"] = 1
    frame = frame.merge(pairs, on=["targetId", "diseaseId"], how="left")
    frame["label"] = frame["label"].fillna(0).astype(int)
    frame["diseaseId"] = frame["diseaseId"].str.split("/").str[-1]
    print(f"[{tag}] frame {frame.shape}, positives {int(frame.label.sum()):,} "
          f"({frame.label.mean():.2%}), targets {frame.targetId.nunique():,}, "
          f"diseases {frame.diseaseId.nunique():,}")
    return frame


if __name__ == "__main__":
    print("=== native text frames (23.06) ===")
    disease_text = build_disease_text(TRAIN_RAW, TABLES_2306)
    target_text = build_target_text(TRAIN_RAW, TABLES_2306)
    disease_text.to_parquet(OUT / "native2306_disease_df.parquet")
    target_text.to_parquet(OUT / "native2306_target_df.parquet")
    print("disease_df:", disease_text.shape, "| target_df:", target_text.shape)

    print("\n=== train label frame (23.06) ===")
    train_frame = build_label_frame(
        TRAIN_RAW / "associationByOverallDirect", TRAIN_RAW / "knownDrugsAggregated",
        TRAIN_RAW / "diseases", "23.06-train")
    n0 = len(train_frame)
    train_frame = train_frame[train_frame.diseaseId.isin(set(disease_text.diseaseId))]
    train_frame = train_frame[train_frame.targetId.isin(set(target_text.targetId))]
    print(f"[23.06-train] after annotation-frame id matching: {n0:,} -> {len(train_frame):,}")
    train_frame.to_parquet(OUT / "native2306_train_frame.parquet")

    print("\n=== eval label frame (25.12) ===")
    eval_frame = build_label_frame(
        EVAL_RAW / "association_overall_direct", EVAL_RAW / "known_drug",
        EVAL_RAW / "disease", "25.12-eval")
    eval_frame.to_parquet(OUT / "native2306_eval_frame.parquet")

    cov = eval_frame.diseaseId.isin(set(disease_text.diseaseId)).mean()
    cov_t = eval_frame.targetId.isin(set(target_text.targetId)).mean()
    print(f"\n23.06 annotation coverage of 25.12 eval frame: disease {cov:.4f}, target {cov_t:.4f}")

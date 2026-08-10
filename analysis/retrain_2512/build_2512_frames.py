"""Build disease_df / target_df / df_learn (main label+text frame) natively from
any post-25.03 OT release (schema-compatible: disease/target/disease_phenotype,
snake_case singular tables).

Thin wrapper: re-exports the fidelity-tested text builders from
rebuttal_scratch/build_2202_features.py (their construction logic is
schema-era-agnostic -- only table NAMES differ between eras, not field names --
already verified 1.0000 exact match against the committed copy_proc frames),
and ports the label-frame recipe verified in
rebuttal_scratch/native_2306/build_native_frames.py (itself ported from
code/2-Temporal-Eval.ipynb / code/0-OT-PreProcess_Recc.ipynb make_target_data()).

Port fidelity for THIS release-pair (25.06 schema, used as the oracle) is
pinned by test_build_2512_frames.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/d/Research/OpenTargetsTransfer/rebuttal_scratch")
from build_2202_features import _exact_synonyms, build_disease_text, build_target_text  # noqa: E402  (re-exported)

REPO = Path("/mnt/d/Research/OpenTargetsTransfer")
OUT = REPO / "retrain_2512"

# Post-25.03 era: snake_case singular tables.
TABLES_2512 = {"diseases": "disease", "targets": "target", "diseaseToPhenotype": "disease_phenotype"}
LABELS_TO_REMOVE = ["measurement", "phenotype", "biological process", "cell proliferation disorder"]


def build_label_frame(assoc_dir, kd_dir, disease_raw, tag):
    """Ports make_target_data() from code/0-OT-PreProcess_Recc.ipynb cells 133/151-155
    (identical recipe already verified for the 23.06 branch in
    rebuttal_scratch/native_2306/build_native_frames.py -- reproduced inline here
    so retrain_2512/ has no cross-branch import into rebuttal_scratch/native_2306/).
    """
    assoc = pd.read_parquet(assoc_dir)
    if "timeseries" in assoc.columns:
        print(f"[{tag}] dropping timeseries column (backfilled history)")
        assoc = assoc.drop(columns=["timeseries"])
    assert "score" in assoc.columns, f"[{tag}] no `score` column: {assoc.columns.tolist()}"
    assoc = assoc[["diseaseId", "targetId", "score"]]

    kd = pd.read_parquet(kd_dir, columns=["targetId", "diseaseId", "phase"]).dropna(subset=["phase"])

    ta_map = disease_raw.set_index("id")["name"].to_dict()
    exploded = disease_raw[["id", "therapeuticAreas"]].explode("therapeuticAreas").dropna()
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
    print(f"[{tag}] frame {frame.shape}, positives {int(frame.label.sum()):,} "
          f"({frame.label.mean():.2%}), targets {frame.targetId.nunique():,}, "
          f"diseases {frame.diseaseId.nunique():,}")
    return frame


def build_df_learn(raw_dir, tables, tag):
    """Full pipeline: raw release dir -> (disease_text, target_text, df_learn)."""
    disease_text = build_disease_text(raw_dir, tables)
    target_text = build_target_text(raw_dir, tables)

    disease_raw = pd.read_parquet(raw_dir / tables["diseases"], columns=["id", "name", "therapeuticAreas"])
    label_frame = build_label_frame(
        raw_dir / "association_overall_direct", raw_dir / "known_drug", disease_raw, tag)

    df_learn = label_frame.merge(disease_text, on="diseaseId").merge(target_text, on="targetId")
    df_learn = df_learn.rename(columns={"disease_text_embed": "disease_text", "target_text_embed": "target_text"})
    print(f"[{tag}] df_learn {df_learn.shape}, positives {int(df_learn.label.sum()):,}")
    return disease_text, target_text, df_learn


def build_full_disease_df(raw_dir, tables):
    """disease_text + every column OTRec/gradio/app.py reads from disease_df
    (contract: diseaseId, disease_text, name, description, synonyms,
    ExactSynonyms -- the search box greps name/synonyms, the summary shows
    description). Display columns come straight from the raw table."""
    text = build_disease_text(raw_dir, tables)
    raw = pd.read_parquet(f"{raw_dir}/{tables['diseases']}",
                          columns=["id", "name", "description", "synonyms"])
    raw = raw.rename(columns={"id": "diseaseId"})
    raw["ExactSynonyms"] = raw["synonyms"].apply(_exact_synonyms).apply(" ".join).str.strip()
    return text.merge(raw, on="diseaseId", how="left").rename(columns={"disease_text_embed": "disease_text"})


def build_full_target_df(raw_dir, tables):
    """target_text + display columns the Space UI reads (approvedSymbol,
    approvedName, functionDescriptions, sym), filtered to the app's candidate
    universe: any tractability evidence OR a known-drug target -- the same
    filter the preprocessing notebook applies (cell ~129), which is how the
    deployed 25.06 target_df arrived at 17,065 of 78,726 targets."""
    text = build_target_text(raw_dir, tables)
    raw = pd.read_parquet(
        f"{raw_dir}/{tables['targets']}",
        columns=["id", "approvedSymbol", "approvedName", "functionDescriptions", "tractability"],
    ).rename(columns={"id": "targetId"})
    raw["functionDescriptions"] = raw["functionDescriptions"].map(
        lambda x: " ".join(x) if isinstance(x, (list, tuple)) else (x if isinstance(x, str) else "")
    ).fillna("")
    raw["sym"] = raw["approvedSymbol"].str.replace(r"\d+", "", regex=True)

    # Truthy-value entries only, matching the notebook's count_tractability
    # (it first drops entries whose "value" is falsy, then counts).
    has_tractability = raw["tractability"].map(
        lambda v: isinstance(v, (list, tuple, np.ndarray))
        and any(isinstance(d, dict) and d.get("value") for d in v)
    )
    known_drug_ids = set(
        pd.read_parquet(f"{raw_dir}/known_drug", columns=["targetId"])["targetId"].unique()
    )
    keep = has_tractability | raw["targetId"].isin(known_drug_ids)
    raw = raw.loc[keep].copy()
    # App contract also reads a display `tractability` (list of "modality id"
    # strings, truthy entries only -- same transform as the text builder).
    raw["tractability"] = raw["tractability"].apply(
        lambda v: [" ".join([d["modality"], d["id"]])
                   for d in (v if isinstance(v, (list, tuple, np.ndarray)) else [])
                   if isinstance(d, dict) and d.get("value") and "id" in d]
    )
    print(f"target_df candidate filter: {len(keep)} -> {len(raw)} "
          f"(tractability or known-drug)")
    return text.merge(raw, on="targetId", how="inner").rename(columns={"target_text_embed": "target_text"})


if __name__ == "__main__":
    RAW_2512 = REPO / "data" / "historical_ot" / "25_12"
    disease_text, target_text, df_learn = build_df_learn(RAW_2512, TABLES_2512, "25.12")
    disease_text.to_parquet(OUT / "disease_df_2512.parquet")
    target_text.to_parquet(OUT / "target_df_2512.parquet")
    df_learn.to_parquet(OUT / "df_learn_2512.parquet")
    print("saved disease_df_2512 / target_df_2512 / df_learn_2512 to", OUT)

    full_disease = build_full_disease_df(RAW_2512, TABLES_2512)
    full_target = build_full_target_df(RAW_2512, TABLES_2512)
    full_disease.to_parquet(OUT / "disease_df_full_2512.parquet")
    full_target.to_parquet(OUT / "target_df_full_2512.parquet")
    print("saved disease_df_full_2512 / target_df_full_2512 (for Space packaging)")

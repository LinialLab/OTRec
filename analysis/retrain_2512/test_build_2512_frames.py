"""Port-fidelity test for build_2512_frames.py — RED before GREEN.

Seam under test: the frames-construction transformation (raw OT tables in,
(disease_text, target_text, df_learn label frame) out).

Oracle: the 25.06 raw snapshot is on disk at data/opentargets/. The committed
frames code/copy_proc/{disease_df,target_df}.parquet and code/final_df.parquet
were produced from it by the notebook (code/0-OT-PreProcess_Recc.ipynb). If our
functions, fed the 25.06 raw tables, reproduce those committed artifacts, the
port is faithful. Independent ground truth -- not recomputed by the code under
test.

Run: python3 retrain_2512/test_build_2512_frames.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/mnt/d/Research/OpenTargetsTransfer/rebuttal_scratch")

REPO = Path("/mnt/d/Research/OpenTargetsTransfer")
NEWER = REPO / "data" / "opentargets"
TABLES_2506 = {"diseases": "disease", "targets": "target", "diseaseToPhenotype": "disease_phenotype"}


def test_disease_text_matches_committed_frame():
    from build_2512_frames import build_disease_text
    got = build_disease_text(NEWER, TABLES_2506)
    expected = pd.read_parquet(REPO / "code" / "copy_proc" / "disease_df.parquet")
    merged = got.merge(expected[["diseaseId", "disease_text_embed"]], on="diseaseId", suffixes=("_got", "_exp"))
    assert len(merged) > 30_000, f"too few overlapping ids: {len(merged)}"
    match = (merged["disease_text_embed_got"] == merged["disease_text_embed_exp"]).mean()
    print(f"disease_text exact match: {match:.4f} over {len(merged):,} ids")
    assert match == 1.0, f"disease_text port not faithful: {match:.4f}"


def test_target_text_matches_committed_frame():
    from build_2512_frames import build_target_text
    got = build_target_text(NEWER, TABLES_2506)
    expected = pd.read_parquet(REPO / "code" / "copy_proc" / "target_df.parquet")
    merged = got.merge(expected[["targetId", "target_text_embed"]], on="targetId", suffixes=("_got", "_exp"))
    assert len(merged) > 15_000, f"too few overlapping ids: {len(merged)}"
    match = (merged["target_text_embed_got"] == merged["target_text_embed_exp"]).mean()
    print(f"target_text exact match: {match:.4f} over {len(merged):,} ids")
    assert match == 1.0, f"target_text port not faithful: {match:.4f}"


def test_label_frame_matches_final_df():
    from build_2512_frames import build_label_frame
    disease_raw = pd.read_parquet(NEWER / "disease", columns=["id", "name", "therapeuticAreas"])
    got = build_label_frame(
        NEWER / "association_overall_direct", NEWER / "known_drug", disease_raw, tag="2506-oracle"
    )
    expected = pd.read_parquet(REPO / "code" / "final_df.parquet")
    assert got.shape[0] == expected.shape[0] == 663_351, f"row count mismatch: {got.shape[0]} vs 663351"
    assert int(got.label.sum()) == int(expected.label.sum()) == 67_532, \
        f"positive count mismatch: {int(got.label.sum())} vs 67532"
    merged = got.merge(expected[["diseaseId", "targetId", "label"]], on=["diseaseId", "targetId"],
                        suffixes=("_got", "_exp"))
    assert len(merged) == len(expected)
    match = (merged["label_got"] == merged["label_exp"]).mean()
    print(f"label frame: {got.shape[0]:,} rows, {int(got.label.sum()):,} positives, label match {match:.4f}")
    assert match == 1.0, f"label mismatch: {match:.4f}"


if __name__ == "__main__":
    test_disease_text_matches_committed_frame()
    test_target_text_matches_committed_frame()
    test_label_frame_matches_final_df()
    print("PASS: frames builder reproduces the notebook's committed 25.06 artifacts")

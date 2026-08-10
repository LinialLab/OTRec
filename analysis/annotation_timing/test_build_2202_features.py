"""Port-fidelity test for build_2202_features.py.

Seam under test: the text-construction transformation (raw OT tables in,
disease_text_embed / target_text_embed strings out).

Oracle: the newer OT snapshot is on disk at data/opentargets/. The committed
frames code/copy_proc/{disease_df,target_df}.parquet were produced from it by
the notebook (code/0-OT-PreProcess_Recc.ipynb). So if the ported functions,
fed the newer raw tables, reproduce those committed text columns, the port is
faithful to the notebook recipe. That is independent ground truth -- the
expected values were computed by the notebook, not by this test.

Run: python3 rebuttal_scratch/test_build_2202_features.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from build_2202_features import build_disease_text, build_target_text

REPO = Path("/mnt/d/Research/OpenTargetsTransfer")
NEWER = REPO / "data" / "opentargets"

# Post-25.03 era table names (snake_case singular).
NEWER_TABLES = {
    "diseases": "disease",
    "targets": "target",
    "diseaseToPhenotype": "disease_phenotype",
}


def test_disease_text_matches_committed_frame():
    got = build_disease_text(NEWER, NEWER_TABLES)
    expected = pd.read_parquet(REPO / "code" / "copy_proc" / "disease_df.parquet")

    merged = got.merge(
        expected[["diseaseId", "disease_text_embed"]],
        on="diseaseId",
        suffixes=("_got", "_expected"),
    )
    assert len(merged) > 30_000, f"too few overlapping disease ids to be meaningful: {len(merged)}"

    match = (merged["disease_text_embed_got"] == merged["disease_text_embed_expected"]).mean()
    print(f"disease_text exact match: {match:.4f} over {len(merged):,} ids")
    if match < 1.0:
        bad = merged[merged["disease_text_embed_got"] != merged["disease_text_embed_expected"]]
        print("  first mismatch:")
        print("    id      :", bad.iloc[0]["diseaseId"])
        print("    got     :", repr(bad.iloc[0]["disease_text_embed_got"][:300]))
        print("    expected:", repr(bad.iloc[0]["disease_text_embed_expected"][:300]))
    assert match == 1.0, f"disease_text port is not faithful: {match:.4f} exact match"


def test_target_text_matches_committed_frame():
    got = build_target_text(NEWER, NEWER_TABLES)
    expected = pd.read_parquet(REPO / "code" / "copy_proc" / "target_df.parquet")

    merged = got.merge(
        expected[["targetId", "target_text_embed"]],
        on="targetId",
        suffixes=("_got", "_expected"),
    )
    assert len(merged) > 15_000, f"too few overlapping target ids to be meaningful: {len(merged)}"

    match = (merged["target_text_embed_got"] == merged["target_text_embed_expected"]).mean()
    print(f"target_text exact match: {match:.4f} over {len(merged):,} ids")
    if match < 1.0:
        bad = merged[merged["target_text_embed_got"] != merged["target_text_embed_expected"]]
        print("  first mismatch:")
        print("    id      :", bad.iloc[0]["targetId"])
        print("    got     :", repr(bad.iloc[0]["target_text_embed_got"][:300]))
        print("    expected:", repr(bad.iloc[0]["target_text_embed_expected"][:300]))
    assert match == 1.0, f"target_text port is not faithful: {match:.4f} exact match"


if __name__ == "__main__":
    test_disease_text_matches_committed_frame()
    test_target_text_matches_committed_frame()
    print("PASS: port reproduces the notebook's committed text columns")

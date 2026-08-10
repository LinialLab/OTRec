"""Old-vs-new popularity-bias comparison, plus an optional hub-filtered variant.

Uses the notebook's own written-but-unused hub filter (HUB_THRESHOLD=200,
code/1-Train-DL-Retriever.ipynb cell ~1443) to produce a SECONDARY file. The
primary S1b/S2b stay raw + target_nomination_count -- no silent filtering of
the primary release.
"""
from pathlib import Path

import pandas as pd

REPO = Path("/mnt/d/Research/OpenTargetsTransfer")
OUT = REPO / "retrain_2512"

HUB_THRESHOLD = 200

old = pd.read_csv(REPO / "OTRec" / "Outputs" / "S1-DL_novel_predictions.csv")
new = pd.read_csv(OUT / "Outputs" / "S1b-DL_novel_predictions_2512.csv")


def bias_metrics(df, label):
    n = len(df)
    vc = df["targetSymbol"].value_counts()
    top10_share = vc.head(10).sum() / n
    tubulin_prefixes = ("TUBB", "TUBA")
    tubulin_share = df["targetSymbol"].str.startswith(tubulin_prefixes).sum() / n
    print(f"\n--- {label} ---")
    print(f"  rows {n:,}, diseases {df.diseaseId.nunique():,}, targets {df.targetId.nunique():,}")
    print(f"  top-10 targets = {top10_share:.1%} of rows")
    print(f"  tubulin-family share = {tubulin_share:.1%} of rows")
    print("  top 8 nominated targets:")
    for sym, cnt in vc.head(8).items():
        print(f"    {sym:<10} {cnt:>6}  ({cnt/df.diseaseId.nunique():.1%} of diseases)")
    return vc


print("=" * 70)
print("POPULARITY BIAS: old (25.06) vs new (25.12) candidate sets")
print("=" * 70)
vc_old = bias_metrics(old, "OLD S1 (Release 25.06)")
vc_new = bias_metrics(new, "NEW S1b (Release 25.12)")

# Training-positives hub check on the new label frame, using the same
# recipe/threshold the notebook computes but never applies.
df_learn = pd.read_parquet(OUT / "df_learn_2512.parquet")
target_meta = pd.read_parquet(
    REPO / "data" / "historical_ot" / "25_12" / "target", columns=["id", "approvedSymbol"]
).set_index("id")["approvedSymbol"]
pos_target_counts = df_learn.query("label>0")["targetId"].value_counts()
hub_ids = set(pos_target_counts[pos_target_counts >= HUB_THRESHOLD].index)
print(f"\nHub proteins at training-positives threshold >={HUB_THRESHOLD}: {len(hub_ids)}")
if hub_ids:
    print("  examples:", [target_meta.get(t, t) for t in list(hub_ids)[:8]])

hub_rows_in_new = new["targetId"].isin(hub_ids).sum() if "targetId" in new.columns else None
print(f"S1b rows nominating a hub protein: {hub_rows_in_new:,} / {len(new):,} "
      f"({hub_rows_in_new/len(new):.1%})" if hub_rows_in_new is not None else "")

# Secondary, curated variant of the NEW set only. Two independent filters:
#  - biotype: protein-coding candidates only. The top-3 nominated targets are
#    all pseudogenes (GUCY1B2 80.7% of diseases, CLCA3P 56.4%, GLRA4 56.0%)
#    nominated on text similarity; the training-positives hub filter misses
#    all three because their label counts are low.
#  - hubs: targets with >=200 training positives (the notebook's own
#    written-but-unused HUB_THRESHOLD filter).
biotype = pd.read_parquet(
    REPO / "data" / "historical_ot" / "25_12" / "target", columns=["id", "biotype"]
).set_index("id")["biotype"]
new_biotype = new["targetId"].map(biotype)
keep = (new_biotype == "protein_coding") & (~new["targetId"].isin(hub_ids))
filtered = new.loc[keep].copy()
filtered.to_csv(OUT / "Outputs" / "S1b_curated_2512.csv", index=False)
dropped_bio = int((~(new_biotype == "protein_coding")).sum())
dropped_hub = int((new["targetId"].isin(hub_ids) & (new_biotype == "protein_coding")).sum())
print(f"\nsaved secondary curated variant: {len(filtered):,} rows -> S1b_curated_2512.csv")
print(f"  dropped {dropped_bio:,} non-protein-coding rows (incl. the pseudogene trio) "
      f"and {dropped_hub:,} hub rows")
print("Primary release remains S1b (raw scores + target_nomination_count + ottree_score); "
      "the curated file is an optional secondary view.")

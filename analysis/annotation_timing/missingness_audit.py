"""Missingness audit + coverage-stratified comparison of the three temporal runs.

Purpose: separate two confounded effects in the 22.02-native run --
  (a) removal of post-2022 information from annotation text (the leak), and
  (b) loss of annotation coverage, because 22.02 simply lacks many entities and
      fields that the 25.06-era snapshot has.

The decisive analysis is the coverage-stratified comparison: on the subset of
test pairs where 22.02 DOES supply disease text, effect (b) is neutralised, so
the remaining gap is attributable to (a) plus 22.02's within-entity field
sparsity.

Runs compared (do not mix these up):
  otrec            -- newer-snapshot annotation features (paper's Table 2 setting)
  otrec_stripped   -- newer snapshot, clinical-precedence tractability tokens removed
  otrec_2202native -- annotation text rebuilt entirely from Release 22.02
All three are seed 42, identical protocol, identical 403,132-pair test set.
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path("/mnt/d/Research/OpenTargetsTransfer")
OUT = REPO / "rebuttal_scratch"
RAW_2202 = REPO / "data" / "historical_ot" / "22_02"
RAW_NEW = REPO / "data" / "opentargets"

TEXT_FIELDS_TARGET = ["go", "pathways", "targetClass", "functionDescriptions",
                      "tractability", "synonyms", "constraint", "subcellularLocations"]
TEXT_FIELDS_DISEASE = ["name", "description", "synonyms", "ancestors", "parents", "therapeuticAreas", "dbXRefs"]


def _nonempty_rate(s):
    """Fraction of rows with usable content (non-null, and non-empty if list-like)."""
    def ok(x):
        if x is None:
            return False
        if isinstance(x, (list, tuple)):
            return len(x) > 0
        try:
            import numpy as np
            if isinstance(x, np.ndarray):
                return x.size > 0
            if pd.isna(x):
                return False
        except (TypeError, ValueError):
            pass
        return True
    return s.apply(ok).mean()


def field_missingness_table():
    print("=" * 78)
    print("FIELD-LEVEL ANNOTATION COVERAGE: Release 22.02 vs newer snapshot")
    print("(fraction of rows with usable, non-empty content)")
    print("=" * 78)

    t_old = pd.read_parquet(RAW_2202 / "targets", columns=["id"] + TEXT_FIELDS_TARGET)
    t_new = pd.read_parquet(RAW_NEW / "target", columns=["id"] + TEXT_FIELDS_TARGET)
    rows = []
    for c in TEXT_FIELDS_TARGET:
        rows.append({"entity": "target", "field": c,
                     "cov_22_02": _nonempty_rate(t_old[c]), "cov_newer": _nonempty_rate(t_new[c])})

    d_old = pd.read_parquet(RAW_2202 / "diseases", columns=["id"] + TEXT_FIELDS_DISEASE)
    d_new = pd.read_parquet(RAW_NEW / "disease", columns=["id"] + TEXT_FIELDS_DISEASE)
    for c in TEXT_FIELDS_DISEASE:
        rows.append({"entity": "disease", "field": c,
                     "cov_22_02": _nonempty_rate(d_old[c]), "cov_newer": _nonempty_rate(d_new[c])})

    tbl = pd.DataFrame(rows)
    tbl["delta"] = tbl["cov_newer"] - tbl["cov_22_02"]
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(f"entity counts: targets  22.02={len(t_old):,}  newer={len(t_new):,}")
    print(f"               diseases 22.02={len(d_old):,}  newer={len(d_new):,}")
    tbl.to_csv(OUT / "missingness_field_coverage.csv", index=False)
    return tbl


def coverage_stratified_comparison():
    print()
    print("=" * 78)
    print("COVERAGE-STRATIFIED COMPARISON OF THE THREE TEMPORAL RUNS")
    print("=" * 78)

    base = pd.read_parquet(OUT / "temporal_preds_seed42.parquet")[
        ["diseaseId", "targetId", "label", "otrec", "score_past"]]
    stripped = pd.read_parquet(OUT / "temporal_preds_seed42_stripped.parquet")[
        ["diseaseId", "targetId", "otrec_stripped", "n_ind_2022", "bin"]]
    native = pd.read_parquet(OUT / "temporal_preds_seed42_2202native.parquet")[
        ["diseaseId", "targetId", "otrec_2202native"]]

    df = base.merge(stripped, on=["diseaseId", "targetId"]).merge(native, on=["diseaseId", "targetId"])
    assert len(df) == len(base) == 403_132, f"join changed row count: {len(df)}"

    d2202 = pd.read_parquet(OUT / "disease_df_2202.parquet")
    t2202 = pd.read_parquet(OUT / "target_df_2202.parquet")
    d_ok = set(d2202.loc[d2202.disease_text_embed.str.strip().ne(""), "diseaseId"])
    t_ok = set(t2202.loc[t2202.target_text_embed.str.strip().ne(""), "targetId"])
    df["has_2202_disease_text"] = df.diseaseId.isin(d_ok)
    df["has_2202_target_text"] = df.targetId.isin(t_ok)
    df["fully_covered"] = df.has_2202_disease_text & df.has_2202_target_text

    models = [("otrec", "newer-snapshot (paper)"),
              ("otrec_stripped", "clinical tokens stripped"),
              ("otrec_2202native", "22.02-native features")]

    def block(sub, title):
        n, pos = len(sub), int(sub.label.sum())
        print(f"\n--- {title}: n={n:,}  pos={pos:,} ({pos/n:.2%}) ---")
        if pos == 0 or pos == n:
            print("    (degenerate label set, skipped)")
            return
        for col, name in models:
            print(f"    {name:<28} ROC {roc_auc_score(sub.label, sub[col]):.4f}"
                  f"   PR {average_precision_score(sub.label, sub[col]):.4f}")

    block(df, "ALL test pairs")
    block(df[df.fully_covered], "COVERED subset (22.02 has both disease+target text)")
    block(df[~df.fully_covered], "UNCOVERED subset (22.02 missing disease and/or target text)")

    print()
    print("--- COVERED subset, by 2022 indication count ---")
    cov = df[df.fully_covered]
    for b in ["0", "1", ">=2"]:
        sub = cov[cov.bin == b]
        if len(sub) == 0 or sub.label.sum() == 0:
            continue
        line = f"  bin {b:>3}: n={len(sub):>6} pos={int(sub.label.sum()):>5}"
        for col, _ in models:
            line += f" | {col.replace('otrec','').lstrip('_') or 'base':>12} ROC {roc_auc_score(sub.label, sub[col]):.4f} PR {average_precision_score(sub.label, sub[col]):.4f}"
        print(line)

    print()
    print("--- coverage summary ---")
    print(f"  pairs with 22.02 disease text : {df.has_2202_disease_text.mean():.4f}")
    print(f"  pairs with 22.02 target  text : {df.has_2202_target_text.mean():.4f}")
    print(f"  fully covered pairs           : {df.fully_covered.mean():.4f}")
    print(f"  positives fully covered       : {df[df.label==1].fully_covered.mean():.4f}")

    df.to_parquet(OUT / "temporal_three_way_comparison.parquet", index=False)
    print(f"\nsaved {OUT/'temporal_three_way_comparison.parquet'}")
    return df


if __name__ == "__main__":
    field_missingness_table()
    coverage_stratified_comparison()

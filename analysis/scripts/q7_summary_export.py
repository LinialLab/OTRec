"""Emit a machine-readable summary of the Q7 stratification from the saved per-pair frame."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = Path(__file__).resolve().parent
sub = pd.read_parquet(OUT / "q7_stratified.parquet")
hist = pd.read_parquet(OUT.parent / "code" / "history_df.parquet")
dd = pd.read_parquet(OUT.parent / "code" / "copy_proc" / "disease_df.parquet",
                     columns=["diseaseId", "therapeuticAreas"])
MODELS = [("OTRec", "otrec"), ("OTTree", "ottree"), ("TargetMean", "TargetMean")]


def per_disease(s, col):
    hits = {1: [], 5: [], 10: []}
    mrrs = []
    for _, g in s.groupby("diseaseId", sort=False):
        y = g.label.to_numpy()
        if y.sum() == 0:      # slicing by target can leave a disease with no positive
            continue
        v = g[col].to_numpy(float)
        n = len(v)
        o = np.argsort(-v, kind="stable")
        srt, ys = v[o], y[o]
        worst = np.empty(n, int)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and srt[j + 1] == srt[i]:
                j += 1
            worst[i:j + 1] = j + 1
            i = j + 1
        best = worst[ys == 1].min()
        mrrs.append(1.0 / best)
        for k in hits:
            hits[k].append(1.0 if best <= k else 0.0)
    return {f"Hit@{k}": float(np.mean(v)) for k, v in hits.items()} | {"MRR": float(np.mean(mrrs))}


rows = []


def emit(s, axis, stratum):
    if s.empty or s.label.sum() == 0:
        return
    for name, col in MODELS:
        r = {"axis": axis, "stratum": stratum, "model": name,
             "n_diseases": s.diseaseId.nunique(), "n_pairs": len(s),
             "n_positives": int(s.label.sum())}
        r.update(per_disease(s, col))
        r["pair_ROC"] = roc_auc_score(s.label, s[col]) if s.label.nunique() > 1 else np.nan
        r["pair_PR"] = average_precision_score(s.label, s[col])
        rows.append(r)


ta = dd.set_index("diseaseId")["therapeuticAreas"].to_dict()
def areas_of(d):
    v = ta.get(d)
    return [str(x) for x in v] if isinstance(v, (list, np.ndarray)) else ([str(v)] if v is not None else [])

cnt = {}
for d in sub.diseaseId.unique():
    for a in areas_of(d):
        cnt[a] = cnt.get(a, 0) + 1
for a, _ in sorted(cnt.items(), key=lambda kv: -kv[1])[:10]:
    ids = [d for d in sub.diseaseId.unique() if a in areas_of(d)]
    emit(sub[sub.diseaseId.isin(ids)], "therapeutic_area", a)

kt = hist[hist.label == 1].groupby("diseaseId")["targetId"].nunique()
sub["n_known_2022"] = sub.diseaseId.map(kt).fillna(0).astype(int)
for lo, hi, lab in [(0, 0, "0 known (orphan)"), (1, 2, "1-2 known"),
                    (3, 10, "3-10 known"), (11, 10**9, ">10 known")]:
    emit(sub[(sub.n_known_2022 >= lo) & (sub.n_known_2022 <= hi)], "known_targets_2022", lab)

qs = sub.drop_duplicates("targetId").ann_vol.quantile([.25, .5, .75]).to_list()
for (lo, hi), lab in zip([(-1, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 10**9)],
                         [f"Q1 (<={qs[0]:.0f})", f"Q2 ({qs[0]:.0f}-{qs[1]:.0f}]",
                          f"Q3 ({qs[1]:.0f}-{qs[2]:.0f}]", f"Q4 (>{qs[2]:.0f})"]):
    emit(sub[(sub.ann_vol > lo) & (sub.ann_vol <= hi)], "target_annotation_volume", lab)

df = pd.DataFrame(rows)
df.to_csv(OUT / "q7_summary.csv", index=False)
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\nwrote", OUT / "q7_summary.csv", df.shape)

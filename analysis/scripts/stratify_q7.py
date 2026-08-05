"""whGz Q7 stratification of the temporal seed-42 predictions.

Three axes, all re-analysis of saved predictions (no training):
  (a) therapeutic area (disease-level, multi-label -> disease counted in each area)
  (b) number of known clinical targets per disease in the 2022 frame
  (c) target annotation volume (GO + pathway + synonym counts) quartiles

Per-disease metrics use the same conservative worst-rank tie handling as the main T3
table. Pair-level ROC/PR are reported alongside so both framings are available.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]                 # workspace root (parent of OTRec)
OUT = ROOT / "OTRec" / "analysis" / "results"

preds = pd.read_parquet(OUT / "temporal_preds_seed42.parquet")
hist = pd.read_parquet(ROOT / "code" / "history_df.parquet")
dd = pd.read_parquet(ROOT / "code" / "copy_proc" / "disease_df.parquet",
                     columns=["diseaseId", "name", "therapeuticAreas"])
td = pd.read_parquet(ROOT / "code" / "copy_proc" / "target_df.parquet",
                     columns=["targetId", "approvedSymbol", "count_go", "count_pathways",
                              "count_synonyms", "count_functionDescriptions"])

prior = float(hist.label.mean())
tm = hist.groupby("targetId")["label"].mean()
preds["TargetMean"] = preds.targetId.map(tm).fillna(prior).astype("float32")

MODELS = [("OTRec", "otrec"), ("OTTree", "ottree"), ("TargetMean", "TargetMean")]


def per_disease(sub, col):
    """Hit@k / MRR macro-averaged over diseases; ties get the WORST rank in their block."""
    hits = {1: [], 5: [], 10: []}
    mrrs = []
    for _, g in sub.groupby("diseaseId", sort=False):
        s = g[col].to_numpy(float)
        y = g.label.to_numpy()
        n = len(s)
        o = np.argsort(-s, kind="stable")
        srt, ys = s[o], y[o]
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


def block(sub, title, extra=""):
    nd = sub.diseaseId.nunique()
    npos = int(sub.label.sum())
    print(f"\n--- {title} | {nd} diseases, {len(sub)} pairs, {npos} positives {extra}")
    if npos == 0:
        print("    no positives - skipped")
        return
    rows = []
    for name, col in MODELS:
        r = per_disease(sub, col)
        r["model"] = name
        r["pair_ROC"] = roc_auc_score(sub.label, sub[col]) if sub.label.nunique() > 1 else np.nan
        r["pair_PR"] = average_precision_score(sub.label, sub[col])
        rows.append(r)
    df = pd.DataFrame(rows)[["model", "Hit@1", "Hit@5", "Hit@10", "MRR", "pair_ROC", "pair_PR"]]
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if npos < 20:
        print("    (!) fewer than 20 positives - unstable")


# restrict to diseases with >=1 temporal positive, as in the main T3 table
posd = preds.groupby("diseaseId").label.sum()
sub = preds[preds.diseaseId.isin(posd[posd > 0].index)].copy()
print(f"BASE: {sub.diseaseId.nunique()} diseases with >=1 temporal positive, {len(sub)} pairs, "
      f"{int(sub.label.sum())} positives")

# ---------------- (a) therapeutic area ----------------
print("\n" + "=" * 78)
print("(a) BY THERAPEUTIC AREA  (disease counted once per area it carries)")
print("=" * 78)
ta = dd.set_index("diseaseId")["therapeuticAreas"].to_dict()
name_map = dd.set_index("diseaseId")["name"].to_dict()


def areas_of(did):
    v = ta.get(did)
    if v is None:
        return []
    if isinstance(v, (list, np.ndarray)):
        return [str(x) for x in v]
    return [str(v)]


area_counts = {}
for did in sub.diseaseId.unique():
    for a in areas_of(did):
        area_counts[a] = area_counts.get(a, 0) + 1
top_areas = sorted(area_counts.items(), key=lambda kv: -kv[1])[:10]
EFO_LABEL = {
    "MONDO_0045024": "cancer or benign tumor", "EFO_0000319": "cardiovascular disease",
    "EFO_0000405": "digestive system disease", "EFO_0005741": "infectious disease",
    "EFO_0005803": "hematologic disease", "EFO_0000540": "immune system disease",
    "EFO_0001379": "endocrine system disease", "EFO_0009605": "pancreas disease",
    "EFO_0010282": "gastrointestinal disease", "OTAR_0000018": "genetic, familial or congenital",
    "OTAR_0000017": "reproductive system or breast disease", "EFO_0000618": "nervous system disease",
    "OTAR_0000006": "musculoskeletal or connective tissue disease",
    "OTAR_0000010": "respiratory or thoracic disease", "OTAR_0000009": "injury, poisoning or complication",
    "EFO_0010285": "integumentary system disease", "OTAR_0000014": "pregnancy or perinatal disease",
    "EFO_0001444": "measurement", "EFO_0003966": "nutritional or metabolic disease",
    "OTAR_0000020": "nutritional or metabolic disease", "EFO_0009690": "urinary system disease",
    "EFO_0000508": "genetic disorder", "MONDO_0024458": "disorder of visual system",
    "OTAR_0000012": "psychiatric disorder",
}
for a, n in top_areas:
    ids = [d for d in sub.diseaseId.unique() if a in areas_of(d)]
    block(sub[sub.diseaseId.isin(ids)], f"{EFO_LABEL.get(a, a)} [{a}]")

# ---------------- (b) known clinical targets per disease (2022) ----------------
print("\n" + "=" * 78)
print("(b) BY NUMBER OF KNOWN CLINICAL TARGETS PER DISEASE IN THE 2022 FRAME")
print("=" * 78)
kt = hist[hist.label == 1].groupby("diseaseId")["targetId"].nunique()
sub["n_known_2022"] = sub.diseaseId.map(kt).fillna(0).astype(int)
for lo, hi, lab in [(0, 0, "0 known targets (orphan in 2022)"), (1, 2, "1-2 known"),
                    (3, 10, "3-10 known"), (11, 10**9, ">10 known")]:
    block(sub[(sub.n_known_2022 >= lo) & (sub.n_known_2022 <= hi)], lab)

# ---------------- (c) target annotation volume ----------------
print("\n" + "=" * 78)
print("(c) BY TARGET ANNOTATION VOLUME  (count_go + count_pathways + count_synonyms)")
print("=" * 78)
td = td.copy()
td["ann_vol"] = (td.count_go.fillna(0) + td.count_pathways.fillna(0) + td.count_synonyms.fillna(0))
sub = sub.merge(td[["targetId", "ann_vol", "approvedSymbol"]], on="targetId", how="left")
tgt_vol = sub.drop_duplicates("targetId")[["targetId", "ann_vol"]]
qs = tgt_vol.ann_vol.quantile([0.25, 0.5, 0.75]).to_list()
print(f"  target annotation-volume quartile cuts: {[round(q,1) for q in qs]} "
      f"(over {len(tgt_vol)} distinct test targets)")
labels = [f"Q1 (<= {qs[0]:.0f})", f"Q2 ({qs[0]:.0f}-{qs[1]:.0f}]",
          f"Q3 ({qs[1]:.0f}-{qs[2]:.0f}]", f"Q4 (> {qs[2]:.0f})"]
bounds = [(-1, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 10**9)]
print("\n  Pair-level discrimination within each target-annotation quartile:")
rows = []
for (lo, hi), lab in zip(bounds, labels):
    d = sub[(sub.ann_vol > lo) & (sub.ann_vol <= hi)]
    if d.label.nunique() < 2:
        continue
    r = {"quartile": lab, "n_pairs": len(d), "n_targets": d.targetId.nunique(),
         "n_pos": int(d.label.sum()), "pos_rate": d.label.mean()}
    for name, col in MODELS:
        r[f"{name}_ROC"] = roc_auc_score(d.label, d[col])
        r[f"{name}_PR"] = average_precision_score(d.label, d[col])
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

sub.to_parquet(OUT / "q7_stratified.parquet", index=False)
print("\nsaved ->", OUT / "q7_stratified.parquet")

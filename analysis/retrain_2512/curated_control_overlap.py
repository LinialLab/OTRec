"""Control arm: within-family overlap of CURATED clinical disease sets.

If curated (clinical-trial) gene-disease sets also show excess within-family
overlap vs a popularity-matched null, then family-structured predictions are a
property of the label space, not an OTRec artifact. Same statistic and null
design as family_overlap_analysis.py, on the 1,514-gene sub-universe with >= 1
known clinical pair (sets are ragged, so shared-count is per-pair mean, not /K).

Run: python3 retrain_2512/curated_control_overlap.py
Output: Outputs/family_overlap_curated_control.csv + printed summary.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/mnt/d/Research/OpenTargetsTransfer/retrain_2512")
OUT = R / "Outputs"
MIN_FAM, DRAWS, SEED = 5, 10_000, 42

target_df = pd.read_parquet(R / "gradio_artifacts" / "target_df.parquet")
disease_df = pd.read_parquet(R / "gradio_artifacts" / "disease_df.parquet")
tid_row = {t: i for i, t in enumerate(target_df.targetId)}
dis_row = {d: i for i, d in enumerate(disease_df.diseaseId)}
pos = pd.read_parquet(R / "df_learn_2512.parquet").query("label == 1")
pos = pos[pos.targetId.isin(tid_row) & pos.diseaseId.isin(dis_row)]

known = {}  # gene row (universe indexing) -> np.array of known disease rows
for t, g in pos.groupby("targetId"):
    known[tid_row[t]] = np.array([dis_row[d] for d in set(g.diseaseId)])
sub = np.array(sorted(known))  # curated sub-universe
lists = [known[g] for g in sub]
pos_of = {g: i for i, g in enumerate(sub)}
sizes_arr = np.array([len(x) for x in lists])
print(f"curated sub-universe: {len(sub)} genes, set sizes median {np.median(sizes_arr):.0f} "
      f"(max {sizes_arr.max()})")

hgnc = pd.read_csv(R / "data_hgnc_complete_set.txt", sep="\t", low_memory=False,
                   usecols=["gene_group", "ensembl_gene_id"]).dropna()
hgnc = hgnc[hgnc.ensembl_gene_id.isin(tid_row)]
fam_long = hgnc.assign(g=hgnc.gene_group.str.split("|")).explode("g")
fam_long["g"] = fam_long.g.str.strip()
families = {}
for g, d in fam_long.groupby("g"):
    mem = [pos_of[tid_row[e]] for e in set(d.ensembl_gene_id) if tid_row[e] in pos_of]
    if len(mem) >= MIN_FAM:
        families[g] = np.array(sorted(mem))
print(f"HGNC groups with >= {MIN_FAM} curated members: {len(families)}")


def stat(idx):
    arr = np.sort(np.concatenate([lists[i] for i in idx]))
    runs = np.diff(np.flatnonzero(np.r_[True, np.diff(arr) != 0, True]))
    m = len(idx)
    return float((runs * (runs - 1)).sum() / (m * (m - 1)))


rng = np.random.default_rng(SEED)
null = {}
for m in sorted({len(v) for v in families.values()}):
    null[m] = np.array([stat(rng.choice(len(sub), m, replace=False)) for _ in range(DRAWS)])

rows = []
for g, idx in families.items():
    obs = stat(idx)
    nd = null[len(idx)]
    rows.append({"family": g, "size": len(idx), "obs_overlap": obs,
                 "null_mean": float(nd.mean()),
                 "excess": obs - float(nd.mean()),
                 "p": (np.count_nonzero(nd >= obs) + 1) / (DRAWS + 1)})
res = pd.DataFrame(rows)
p = res.p.to_numpy()
order = np.argsort(p)
res.loc[res.index[order], "q"] = np.minimum.accumulate(
    (p[order] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
res.sort_values("excess", ascending=False).to_csv(
    OUT / "family_overlap_curated_control.csv", index=False)

sig = res[res.q < 0.05]
print(f"significant excess curated overlap (BH q<0.05): {len(sig)}/{len(res)} "
      f"({len(sig)/len(res):.0%})")
print("top 8 by excess (mean shared curated diseases per gene pair):")
for _, r in res.nlargest(8, "excess").iterrows():
    print(f"  {r.family[:52]:52s} n={r['size']:<4d} obs {r.obs_overlap:6.2f} "
          f"null {r.null_mean:5.2f} excess {r.excess:+6.2f} q {r.q:.2g}")

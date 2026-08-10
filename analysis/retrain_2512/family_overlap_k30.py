"""K=30 primary run: family overlap, sibling transfer, RAB decomposition, pair null.

Same design as family_overlap_analysis.py (audited), single K=30 variant.
Outputs: Outputs/family_overlap_results_k30.csv, rab_excess_decomposition_k30.csv,
rab_pair_stats_k30.csv; summary printed.
Run: python3 retrain_2512/family_overlap_k30.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/mnt/d/Research/OpenTargetsTransfer/retrain_2512")
OUT = R / "Outputs"
K, MIN_FAM, DRAWS, SEED = 30, 10, 10_000, 42

mem = np.load(OUT / "gene_topk.npz")["topk_idx"][:, :K]
N = len(mem)
tgt = pd.read_parquet(R / "gradio_artifacts" / "target_df.parquet")
dis = pd.read_parquet(R / "gradio_artifacts" / "disease_df.parquet")
tid_row = {t: i for i, t in enumerate(tgt.targetId)}
dis_row = {d: i for i, d in enumerate(dis.diseaseId)}

pos = pd.read_parquet(R / "df_learn_2512.parquet").query("label == 1")
pos = pos[pos.targetId.isin(tid_row) & pos.diseaseId.isin(dis_row)]
known_by_gene = {}
for t, d in zip(pos.targetId, pos.diseaseId):
    known_by_gene.setdefault(tid_row[t], set()).add(dis_row[d])

h = pd.read_csv(R / "data_hgnc_complete_set.txt", sep="\t", low_memory=False,
                usecols=["gene_group", "ensembl_gene_id"]).dropna()
h = h[h.ensembl_gene_id.isin(tid_row)]
fam_long = h.assign(g=h.gene_group.str.split("|")).explode("g")
fam_long["g"] = fam_long.g.str.strip()
families = {g: np.array(sorted(tid_row[e] for e in set(d.ensembl_gene_id)))
            for g, d in fam_long.groupby("g") if d.ensembl_gene_id.nunique() >= MIN_FAM}

n_d = np.bincount(mem.ravel(), minlength=len(dis)).astype(float)
exact_null = float((n_d * (n_d - 1)).sum() / (N * (N - 1)))
print(f"K={K}: distinct diseases in any list {(n_d>0).sum():,}; "
      f"top-100 slot share {np.sort(n_d)[::-1][:100].sum()/(N*K):.0%}; "
      f"exact null mean {exact_null:.2f}/{K}")

def stat(idx):
    arr = np.sort(mem[idx].ravel())
    runs = np.diff(np.flatnonzero(np.r_[True, np.diff(arr) != 0, True]))
    m = len(idx)
    return float((runs * (runs - 1)).sum() / (m * (m - 1)))

rng = np.random.default_rng(SEED)
null = {}
for m in sorted({len(v) for v in families.values()}):
    st = np.empty(DRAWS)
    done = 0
    while done < DRAWS:
        chunk = min(500, DRAWS - done)
        u = rng.random((chunk, N), dtype=np.float32)
        idxs = np.argpartition(u, m, axis=1)[:, :m]
        for j in range(chunk):
            st[done + j] = stat(idxs[j])
        done += chunk
    null[m] = st

rows = []
for g, idx in families.items():
    obs = stat(idx)
    nd = null[len(idx)]
    rows.append({"family": g, "size": len(idx), "obs_overlap": obs,
                 "null_mean": float(nd.mean()), "excess": obs - exact_null,
                 "p": (np.count_nonzero(nd >= obs) + 1) / (DRAWS + 1)})
res = pd.DataFrame(rows)
p = res.p.to_numpy(); order = np.argsort(p)
res.loc[res.index[order], "q"] = np.minimum.accumulate(
    (p[order] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
res.to_csv(OUT / "family_overlap_results_k30.csv", index=False)
sig = res[res.q < 0.05]
rab = res[res.family.str.startswith("RAB, member")].iloc[0]
print(f"significant: {len(sig)}/{len(res)} ({len(sig)/len(res):.0%}); "
      f"RAB obs {rab.obs_overlap:.2f} excess {rab.excess:+.2f} p {rab.p:.2g} "
      f"q {rab.q:.2g} pct {(res.excess < rab.excess).mean()*100:.0f}")

# sibling transfer at K
in_top = {g: set(mem[g]) for g in range(N)}
tot_obs = tot_exp = tot_var = tot_n = 0
for g, idx in families.items():
    for gg in idx:
        for d in known_by_gene.get(gg, ()):
            for s in idx:
                if s == gg or d in known_by_gene.get(s, ()):
                    continue
                tot_n += 1
                tot_obs += d in in_top[s]
                pd_ = n_d[d] / N
                tot_exp += pd_; tot_var += pd_ * (1 - pd_)
print(f"sibling transfer: {tot_n:,} triples, observed {tot_obs:,} vs expected "
      f"{tot_exp:,.0f} (lift {tot_obs/tot_exp:.2f}, z {(tot_obs-tot_exp)/np.sqrt(tot_var):.0f})")

# RAB decomposition at K (64-gene HGNC group)
hr = h[h.gene_group.str.contains("RAB, member RAS oncogene GTPases", regex=False)]
rrows = np.array(sorted(tid_row[e] for e in set(hr.ensembl_gene_id)))
m = len(rrows)
c_d = np.bincount(mem[rrows].ravel(), minlength=len(dis)).astype(float)
contrib = c_d*(c_d-1)/(m*(m-1)) - n_d*(n_d-1)/(N*(N-1))
dec = pd.DataFrame({"disease": dis.name.fillna(dis.diseaseId), "diseaseId": dis.diseaseId,
                    "rab_genes_topk": c_d.astype(int), "all_genes_topk_pct": (n_d/N).round(4),
                    "enrichment": ((c_d/m)/np.maximum(n_d/N, 1e-12)).round(2),
                    "excess_contribution": contrib.round(4)})
dec = dec[dec.rab_genes_topk > 0].sort_values("excess_contribution", ascending=False)
dec.to_csv(OUT / "rab_excess_decomposition_k30.csv", index=False)
t10 = dec.head(10)
print(f"decomposition: obs {(c_d*(c_d-1)).sum()/(m*(m-1)):.2f} excess {contrib.sum():.2f}; "
      f"top-10 share {t10.excess_contribution.sum()/contrib.sum():.0%}")
print(t10[["disease","rab_genes_topk","all_genes_topk_pct","enrichment"]].to_string(index=False))

# pair-level null + ladder
S = np.sort(mem, axis=1)
def shared(a, b):
    return len(np.intersect1d(S[a], S[b], assume_unique=True))
pairs = rng.integers(0, N, (200_000, 2)); pairs = pairs[pairs[:,0] != pairs[:,1]]
nullp = np.array([shared(a, b) for a, b in pairs])
p95 = np.percentile(nullp, 95)
print(f"pair null: mean {nullp.mean():.1f}/{K}, 95th pct {p95:.0f}")
obs_pairs = np.array([shared(a, b) for i, a in enumerate(rrows) for b in rrows[i+1:]])
syms = tgt.approvedSymbol.to_numpy()
iu0, iu1 = zip(*[(a, b) for i, a in enumerate(rrows) for b in rrows[i+1:]])
pd.DataFrame({"gene_a": syms[list(iu0)], "gene_b": syms[list(iu1)],
              "shared_topk": obs_pairs}).sort_values("shared_topk", ascending=False) \
    .to_csv(OUT / "rab_pair_stats_k30.csv", index=False)
print(f"RAB pairs > 95th pct: {(obs_pairs > p95).mean():.0%} of {len(obs_pairs):,} "
      f"(median {np.median(obs_pairs):.0f}, range {obs_pairs.min()}-{obs_pairs.max()})")
for fam in ["Immunoglobulin lambda locus", "Interleukin receptors", "Tubulin beta family"]:
    fr = families[[k for k in families if k.startswith(fam)][0]]
    fo = np.array([shared(a, b) for i, a in enumerate(fr) for b in fr[i+1:]])
    print(f"  ladder {fam[:30]:30s} pairs>95th: {(fo > p95).mean():.0%} median {np.median(fo):.0f}")

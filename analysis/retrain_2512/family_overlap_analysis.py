"""Gene-family overlap of OTRec-predicted diseases, at scale.

For every HGNC gene group with >= MIN_FAM members in the 17,073-target
candidate universe: do family members share their top-K predicted diseases
more than random gene sets do? Random sets drawn from the real per-gene
top-K lists condition EXACTLY on disease popularity (each disease's overall
frequency across lists is untouched), so excess overlap is popularity-
adjusted by construction.

Statistic per gene set S (|S|=m): mean pairwise shared-count
    stat(S) = sum_d C(c_d, 2) / C(m, 2),   c_d = #genes in S with d in top-K
which equals mean_{i<j in S} |topK_i ∩ topK_j| (in diseases out of K).
Null: 10,000 random m-subsets of the universe; p = (r+1)/(n+1), BH FDR.
Exact null mean (all-pairs average) = sum_d C(n_d,2)/C(N,2).

Variants: K=50 raw (primary), K=50 novel-only (known clinical pairs masked
before taking top-K), K=20 / K=100 (sensitivity).

Sibling-transfer test (non-circular check): for each known clinical pair
(g,d) with g in a family, how often is d in the raw top-50 of siblings s
(excluding s with (s,d) itself known), vs the popularity baseline n_d/N?

Inputs: Outputs/gene_topk.npz (precompute_gene_topk.py), gradio_artifacts
frames, df_learn_2512.parquet, data_hgnc_complete_set.txt.
Outputs: Outputs/family_overlap_results.csv, family_overlap_volcano.png,
         family_overlap_sibling_transfer.csv; summary printed.
Run: python3 retrain_2512/family_overlap_analysis.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("/mnt/d/Research/OpenTargetsTransfer/retrain_2512")
OUT = R / "Outputs"
MIN_FAM, DRAWS, SEED = 10, 10_000, 42

# ---------------------------------------------------------------- inputs ----
tk = np.load(OUT / "gene_topk.npz")
top200, N = tk["topk_idx"], tk["topk_idx"].shape[0]
target_df = pd.read_parquet(R / "gradio_artifacts" / "target_df.parquet")
disease_df = pd.read_parquet(R / "gradio_artifacts" / "disease_df.parquet")
n_dis = len(disease_df)
tid_row = {t: i for i, t in enumerate(target_df.targetId)}
dis_row = {d: i for i, d in enumerate(disease_df.diseaseId)}

pos = pd.read_parquet(R / "df_learn_2512.parquet").query("label == 1")
pos = pos[pos.targetId.isin(tid_row) & pos.diseaseId.isin(dis_row)]
known_by_gene = {}  # gene row -> set of known disease rows
for t, d in zip(pos.targetId, pos.diseaseId):
    known_by_gene.setdefault(tid_row[t], set()).add(dis_row[d])
print(f"universe {N} genes x {n_dis} diseases; known clinical pairs mapped: "
      f"{sum(len(v) for v in known_by_gene.values()):,} on {len(known_by_gene)} genes")

hgnc = pd.read_csv(R / "data_hgnc_complete_set.txt", sep="\t", low_memory=False,
                   usecols=["gene_group", "ensembl_gene_id"]).dropna()
hgnc = hgnc[hgnc.ensembl_gene_id.isin(tid_row)]
fam_long = hgnc.assign(g=hgnc.gene_group.str.split("|")).explode("g")
fam_long["g"] = fam_long.g.str.strip()
families = {g: np.array(sorted(tid_row[e] for e in set(d.ensembl_gene_id)))
            for g, d in fam_long.groupby("g") if d.ensembl_gene_id.nunique() >= MIN_FAM}
print(f"HGNC groups with >= {MIN_FAM} members in universe: {len(families)}")

# ------------------------------------------------------------ memberships ---
def novel_top50():
    m = top200[:, :50].copy()
    short = []
    for g, kn in known_by_gene.items():
        keep = [d for d in top200[g] if d not in kn]
        if len(keep) < 50:
            short.append(g)
        else:
            m[g] = keep[:50]
    if short:  # heavily-trialed genes: rank their FULL score row, then mask
        emb = np.load(R / "gradio_artifacts" / "embeddings.npz", allow_pickle=False)
        cos = emb["candidate_embs"][short] @ emb["disease_embs"].T
        for j, g in enumerate(short):
            full = np.argsort(-cos[j])
            m[g] = [d for d in full if d not in known_by_gene[g]][:50]
        print(f"  novel masking: {len(short)} heavily-trialed genes re-ranked from full rows")
    return m

VARIANTS = {"K50": top200[:, :50], "K50_novel": novel_top50(),
            "K20": top200[:, :20], "K100": top200[:, :100]}

def stat(mem, idx):
    """Mean pairwise shared-count for gene rows idx under membership mem."""
    arr = np.sort(mem[idx].ravel())
    runs = np.diff(np.flatnonzero(np.r_[True, np.diff(arr) != 0, True]))
    m = len(idx)
    return float((runs * (runs - 1)).sum() / (m * (m - 1)))  # /2 top+bottom cancel

# ------------------------------------------------- nulls, shared across variants
rng = np.random.default_rng(SEED)
sizes = sorted({len(v) for v in families.values()})
null = {v: {} for v in VARIANTS}  # variant -> size -> (draws,) stats
print(f"{len(sizes)} distinct family sizes {sizes[0]}..{sizes[-1]}; "
      f"{DRAWS} draws each (shared subsets across variants)")
for m in sizes:
    stats = {v: np.empty(DRAWS) for v in VARIANTS}
    done = 0
    while done < DRAWS:
        chunk = min(500, DRAWS - done)
        u = rng.random((chunk, N), dtype=np.float32)
        idxs = np.argpartition(u, m, axis=1)[:, :m]
        for v, mem in VARIANTS.items():
            for j in range(chunk):
                stats[v][done + j] = stat(mem, idxs[j])
        done += chunk
    for v in VARIANTS:
        null[v][m] = stats[v]
    print(f"  size {m} done", flush=True)

# ------------------------------------------------------------- per family ---
rows = []
for v, mem in VARIANTS.items():
    K = mem.shape[1]
    n_d = np.bincount(mem.ravel(), minlength=n_dis).astype(np.float64)
    exact_null = float((n_d * (n_d - 1)).sum() / (N * (N - 1)))
    for g, idx in families.items():
        obs = stat(mem, idx)
        nd = null[v][len(idx)]
        p = (np.count_nonzero(nd >= obs) + 1) / (DRAWS + 1)
        rows.append({"variant": v, "family": g, "size": len(idx), "K": K,
                     "obs_overlap": obs, "null_mean": float(nd.mean()),
                     "exact_null_mean": exact_null,
                     "excess": obs - exact_null, "p": p})
res = pd.DataFrame(rows)
for v in VARIANTS:  # BH within variant
    sub = res.variant == v
    p = res.loc[sub, "p"].to_numpy()
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    res.loc[res.index[sub][order], "q"] = q
res.to_csv(OUT / "family_overlap_results.csv", index=False)

for v in VARIANTS:
    sub = res[res.variant == v]
    sig = sub[sub.q < 0.05]
    rab = sub[sub.family.str.startswith("RAB, member")].iloc[0]
    pct = (sub.excess < rab.excess).mean() * 100
    print(f"\n[{v}] exact null mean {sub.exact_null_mean.iloc[0]:.2f}/{sub.K.iloc[0]} | "
          f"significant (BH q<0.05): {len(sig)}/{len(sub)} ({len(sig)/len(sub):.0%})")
    print(f"  RAB GTPases: obs {rab.obs_overlap:.2f}, excess {rab.excess:+.2f}, "
          f"p {rab.p:.4g}, q {rab.q:.4g}, excess percentile {pct:.0f}")
    print("  top 8 by excess:")
    for _, r in sub.nlargest(8, "excess").iterrows():
        print(f"    {r.family[:52]:52s} n={r['size']:<4d} obs {r.obs_overlap:6.2f} "
              f"excess {r.excess:+6.2f} q {r.q:.2g}")

# ------------------------------------------------------- sibling transfer ---
mem50 = VARIANTS["K50"]
n_d50 = np.bincount(mem50.ravel(), minlength=n_dis)
in_top = {g: set(mem50[g]) for g in range(N)}
srows = []
for g, idx in families.items():
    fam_set = set(idx)
    obs = exp = var = n_tr = 0
    for gg in idx:
        for d in known_by_gene.get(gg, ()):
            for s in idx:
                if s == gg or d in known_by_gene.get(s, ()):
                    continue
                n_tr += 1
                obs += d in in_top[s]
                pd_ = n_d50[d] / N
                exp += pd_
                var += pd_ * (1 - pd_)
    if n_tr:
        z = (obs - exp) / np.sqrt(var) if var else np.nan
        srows.append({"family": g, "triples": n_tr, "observed": obs,
                      "expected": round(exp, 1), "lift": round(obs / exp, 2) if exp else np.nan,
                      "z": round(z, 1)})
sib = pd.DataFrame(srows).sort_values("z", ascending=False)
sib.to_csv(OUT / "family_overlap_sibling_transfer.csv", index=False)
tot_obs, tot_exp = sib.observed.sum(), sib.expected.sum()
print(f"\nSIBLING TRANSFER (K=50): {len(sib)} families with testable triples, "
      f"{sib.triples.sum():,} triples; pooled observed {tot_obs:,} vs expected "
      f"{tot_exp:,.0f} (lift {tot_obs/tot_exp:.2f})")
rab_s = sib[sib.family.str.startswith("RAB, member")]
print("  RAB:", rab_s.to_dict("records") if len(rab_s) else "no testable triples (no known pairs)")
print("  top 6 by z:")
for _, r in sib.head(6).iterrows():
    print(f"    {r.family[:52]:52s} triples {r.triples:<6d} lift {r.lift} z {r.z}")

print("\nsaved family_overlap_results.csv, family_overlap_sibling_transfer.csv "
      "(figure: plot_family_volcano.py)")

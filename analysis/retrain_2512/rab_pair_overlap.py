"""RAB pair-level overlap: how much each gene pair shares, vs random pairs.

Produces the two PI-facing figures and the per-pair statistics:
  rab_pair_overlap_heatmap.png   -- 64x64 genes, color = shared top-50 diseases
                                    MINUS the random-pair baseline (diverging)
  rab_overlap_disease_dumbbell.png -- top-10 excess-driving diseases:
                                    % of RAB genes vs % of all genes predicting
  rab_pair_stats.csv             -- per pair: shared count, excess, null percentile
Null: 200,000 random gene pairs from the 17,073 universe (empirical).
Run: python3 retrain_2512/rab_pair_overlap.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage

R = Path("/mnt/d/Research/OpenTargetsTransfer/retrain_2512")
OUT = R / "Outputs"
INK, MUTED, SEQ, ACCENT = "#1f2430", "#5b6472", "#3d6de0", "#c23934"
rng = np.random.default_rng(42)

K = 30
top50 = np.load(OUT / "gene_topk.npz")["topk_idx"][:, :K]
N = len(top50)
tgt = pd.read_parquet(R / "gradio_artifacts" / "target_df.parquet")
tid_row = {t: i for i, t in enumerate(tgt.targetId)}
h = pd.read_csv(R / "data_hgnc_complete_set.txt", sep="\t", low_memory=False,
                usecols=["symbol", "gene_group", "ensembl_gene_id"]).dropna()
h = h[h.ensembl_gene_id.isin(tid_row)]

def members(group):
    sub = h[h.gene_group.str.contains(group, regex=False)]
    ids = sorted(set(sub.ensembl_gene_id))
    return np.array([tid_row[e] for e in ids]), \
        [sub[sub.ensembl_gene_id == e].symbol.iloc[0] for e in ids]

rows, syms = members("RAB, member RAS oncogene GTPases")
m = len(rows)

# sorted top-50 rows -> fast pairwise intersection counts
S = np.sort(top50, axis=1)
def shared(a, b):
    return len(np.intersect1d(S[a], S[b], assume_unique=True))

mat = np.zeros((m, m))
for i in range(m):
    for j in range(i + 1, m):
        mat[i, j] = mat[j, i] = shared(rows[i], rows[j])

# empirical random-pair null
pairs = rng.integers(0, N, (200_000, 2))
pairs = pairs[pairs[:, 0] != pairs[:, 1]]
null = np.array([shared(a, b) for a, b in pairs])
null_mean, p95 = null.mean(), np.percentile(null, 95)
print(f"random-pair null: mean {null_mean:.1f}, 95th pct {p95:.0f} ({len(null):,} pairs)")

iu = np.triu_indices(m, 1)
obs = mat[iu]
frac_sig = (obs > p95).mean()
print(f"RAB pairs above null 95th pct: {frac_sig:.0%} of {len(obs):,} pairs "
      f"(median shared {np.median(obs):.0f}, range {obs.min():.0f}-{obs.max():.0f})")
pd.DataFrame({"gene_a": np.array(syms)[iu[0]], "gene_b": np.array(syms)[iu[1]],
              "shared_topk": obs.astype(int), "excess_vs_null": (obs - null_mean).round(1),
              "null_percentile": [round((null < s).mean() * 100, 1) for s in obs]}) \
    .sort_values("shared_topk", ascending=False).to_csv(OUT / "rab_pair_stats_k30.csv", index=False)

for fam in ["Immunoglobulin lambda locus", "Interleukin receptors", "Tubulin beta family"]:
    fr, _ = members(fam)
    fo = np.array([shared(a, b) for i, a in enumerate(fr) for b in fr[i + 1:]])
    print(f"  context {fam[:32]:32s} n={len(fr):<3d} pairs>95th: {(fo > p95).mean():.0%} "
          f"median {np.median(fo):.0f}")

# ------------------------------------------------------------- pair heatmap -
from scipy.spatial.distance import squareform
d = mat.max() - mat
np.fill_diagonal(d, 0)
order = leaves_list(linkage(squareform(d), method="average"))
mo = mat[np.ix_(order, order)] - null_mean
np.fill_diagonal(mo, np.nan)
lab = np.array(syms)[order]
fig, ax = plt.subplots(figsize=(11.2, 10.4), dpi=200)
im = ax.imshow(mo, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=min(mo[~np.isnan(mo)].min(), -1), vmax=max(mo[~np.isnan(mo)].max(), 1)))
ax.set_xticks(range(m)); ax.set_yticks(range(m))
ax.set_xticklabels(lab, rotation=90, fontsize=6, color=INK)
ax.set_yticklabels(lab, fontsize=6, color=INK)
ax.set_title(f"Shared top-{K} predicted diseases per RAB gene pair, relative to the "
             f"random-pair baseline ({null_mean:.1f}/{K})", loc="left", fontsize=11, color=INK)
cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
cbar.set_label("shared diseases − baseline (red = above, blue = below)", color=MUTED)
cbar.ax.tick_params(colors=MUTED)
ax.tick_params(colors=MUTED)
fig.tight_layout()
fig.savefig(OUT / "rab_pair_overlap_heatmap.png", bbox_inches="tight")
print("saved rab_pair_overlap_heatmap.png")

# --------------------------------------------------------------- dumbbell ---
dec = pd.read_csv(OUT / "rab_excess_decomposition_k30.csv").head(10)[::-1]
fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=200)
y = np.arange(len(dec))
rabpct = dec.rab_genes_topk / m * 100
allpct = dec.all_genes_topk_pct * 100
ax.hlines(y, allpct, rabpct, color="#b8c0cc", lw=2, zorder=1)
ax.scatter(allpct, y, s=55, c="#8d97a8", zorder=2)
ax.scatter(rabpct, y, s=64, c=ACCENT, zorder=3)
ax.set_ylim(-0.6, len(dec) - 0.2)
ax.annotate("all 17,073 genes", (allpct.iloc[-1], y[-1]), xytext=(0, 13),
            textcoords="offset points", ha="center", color="#8d97a8", fontsize=9)
ax.annotate("RAB family (64 genes)", (rabpct.iloc[-1], y[-1]), xytext=(0, 13),
            textcoords="offset points", ha="center", color=ACCENT, fontsize=9, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels([d[:44] for d in dec.disease], fontsize=9, color=INK)
ax.set_xlabel(f"% of genes with the disease in their top-{K} predictions", color=INK)
ax.set_xlim(0, 104)
ax.set_title("The 10 diseases driving the RAB family's excess overlap\n"
             "(all oncology)", loc="left", color=INK, fontsize=11)
for s_ in ["top", "right"]:
    ax.spines[s_].set_visible(False)
ax.tick_params(colors=MUTED)
fig.tight_layout()
fig.savefig(OUT / "rab_overlap_disease_dumbbell.png", bbox_inches="tight")
print("saved rab_overlap_disease_dumbbell.png")

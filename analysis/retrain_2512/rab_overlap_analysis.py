"""RAB gene family case study: overlap structure of predicted diseases.

Reproducible analysis for the coauthor's question: how much do the RAB GTPases'
top predicted diseases overlap, which overlaps are expected (paralog
subfamilies sharing biology/annotation) vs unexpected (family-wide recurrence
of high-burden diseases, i.e. disease-side popularity), and how does it look
novel-only vs including known clinical pairs.

Scores reproduce the deployed Space exactly (packaged embeddings + deployed
cls_head calibration, saved by precompute_gene_topk.py). Companion scale-up
across all HGNC families: family_overlap_analysis.py.

Outputs (retrain_2512/Outputs/):
  rab_overlap_scores.csv        -- all RAB x disease scores >= 0.5 (long form)
  rab_overlap_recurrence.csv    -- per-disease: n RAB genes with high-conf pred
  rab_overlap_heatmap.png       -- genes x recurrent diseases
  rab_overlap_recurrence.png    -- histogram: disease recurrence across family
Run: python3 retrain_2512/rab_overlap_analysis.py   (needs Outputs/gene_topk.npz)
"""
import itertools
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage

R = Path("/mnt/d/Research/OpenTargetsTransfer/retrain_2512")
PKG = R / "gradio_artifacts"
OUT = R / "Outputs"
HI_CONF = 0.65  # the paper's release threshold

# ---------------------------------------------------------------- data ------
target_df = pd.read_parquet(PKG / "target_df.parquet")
disease_df = pd.read_parquet(PKG / "disease_df.parquet")
df_learn = pd.read_parquet(R / "df_learn_2512.parquet")

rab = target_df[target_df.approvedSymbol.str.match(r"^RAB\d+[A-Z]?$", na=False)] \
    .sort_values("approvedSymbol").reset_index(drop=True)
print(f"{len(rab)} RAB GTPases in the app candidate universe")

emb = np.load(PKG / "embeddings.npz", allow_pickle=False)
tk = np.load(OUT / "gene_topk.npz")
w, b = float(tk["w"]), float(tk["b"])
tid_to_row = {t: i for i, t in enumerate(target_df.targetId)}
rab_rows = [tid_to_row[t] for t in rab.targetId]
probs = 1.0 / (1.0 + np.exp(-(emb["candidate_embs"][rab_rows] @ emb["disease_embs"].T * w + b)))
print(f"score matrix {probs.shape}; cls_head w={w:+.2f} b={b:+.2f}")

known = set(zip(df_learn.query("label==1").targetId, df_learn.query("label==1").diseaseId))
dis_ids = disease_df.diseaseId.to_numpy()
dis_names = disease_df.name.fillna(disease_df.diseaseId).to_numpy()

# ---------------------------------------------------- long-form + recurrence
rows = []
for gi, (tid, sym) in enumerate(zip(rab.targetId, rab.approvedSymbol)):
    for di in np.where(probs[gi] >= 0.5)[0]:
        rows.append({"gene": sym, "targetId": tid, "diseaseId": dis_ids[di],
                     "disease": dis_names[di], "score": round(float(probs[gi, di]), 4),
                     "known_clinical": (tid, dis_ids[di]) in known})
long = pd.DataFrame(rows)
long.to_csv(OUT / "rab_overlap_scores.csv", index=False)

hi = long[long.score >= HI_CONF]
rec = (hi.groupby(["diseaseId", "disease"])
         .agg(n_genes=("gene", "nunique"), n_known=("known_clinical", "sum"))
         .reset_index().sort_values("n_genes", ascending=False))
rec.to_csv(OUT / "rab_overlap_recurrence.csv", index=False)
n_known_pairs = int(hi.known_clinical.sum())
print(f"high-confidence (>= {HI_CONF}) pairs: {len(hi):,} "
      f"({n_known_pairs} known clinical); diseases hit: {rec.shape[0]:,}")
print(f"diseases hit by >= half the family (>= {len(rab)//2} genes): "
      f"{(rec.n_genes >= len(rab)//2).sum()}")

# --------------------------------------------------------------- histogram --
INK, MUTED = "#1f2430", "#5b6472"
SEQ = "#3d6de0"
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
ax.hist(rec.n_genes, bins=np.arange(0.5, len(rab) + 1.5, 2), color=SEQ,
        edgecolor="white", linewidth=0.8)
ax.set_xlabel(f"RAB genes predicting the disease at score ≥ {HI_CONF}", color=INK)
ax.set_ylabel("Diseases", color=INK)
ax.set_title(f"Disease recurrence across the RAB family ({len(rab)} genes)",
             color=INK, loc="left")
med = rec.n_genes.median()
ax.axvline(med, color=MUTED, lw=1, ls="--")
ax.annotate(f"median {med:.0f}", (med, ax.get_ylim()[1] * 0.9),
            xytext=(6, 0), textcoords="offset points", color=MUTED, fontsize=9)
top = rec.iloc[0]
ax.annotate(f"max: {top.disease} ({top.n_genes} genes)",
            (top.n_genes, 2), xytext=(-8, 18), textcoords="offset points",
            ha="right", color=MUTED, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(colors=MUTED)
fig.tight_layout()
fig.savefig(OUT / "rab_overlap_recurrence.png", bbox_inches="tight")
print("saved rab_overlap_recurrence.png")

# ----------------------------------------------------------------- heatmap --
# Rows: top 25 recurrent diseases (readable horizontal labels, with each
# disease's genome-wide >=0.65 rate for scale); columns: the 62 genes.
cols = rec[rec.n_genes >= max(3, len(rab) // 4)].head(25)
col_idx = [np.where(dis_ids == d)[0][0] for d in cols.diseaseId]
glob = 1.0 / (1.0 + np.exp(-(emb["candidate_embs"] @ emb["disease_embs"][col_idx].T * w + b)))
glob_rate = (glob >= HI_CONF).mean(axis=0)  # share of ALL 17,073 genes >= 0.65

order = leaves_list(linkage(probs >= HI_CONF, method="average", metric="jaccard"))
mat = probs[:, col_idx].T[:, order]  # diseases x genes
gene_labels = rab.approvedSymbol.to_numpy()[order]

fig, ax = plt.subplots(figsize=(13.2, 8.2), dpi=200)
im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0.5, vmax=1.0)
ax.set_yticks(range(len(cols)))
ax.set_yticklabels([f"{textwrap.shorten(n, 44, placeholder='…')}   · {g:.0%} of all genes"
                    for n, g in zip(cols.disease, glob_rate)], fontsize=9, color=INK)
ax.set_xticks(range(len(gene_labels)))
ax.set_xticklabels(gene_labels, rotation=90, fontsize=6.5, color=INK)
ax.set_title(f"OTRec score ≥ {HI_CONF}: the {len(cols)} diseases most recurrent across the RAB family\n"
             f"(row label shows how much of the whole druggable genome also predicts it; "
             "no RAB–disease pair has clinical-trial evidence in Release 25.12)",
             loc="left", fontsize=10.5, color=INK)
cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
cbar.set_label("OTRec score (color floor 0.5)", color=MUTED)
cbar.ax.tick_params(colors=MUTED)
ax.tick_params(colors=MUTED)
fig.tight_layout()
fig.savefig(OUT / "rab_overlap_heatmap.png", bbox_inches="tight")
print("saved rab_overlap_heatmap.png")

# ------------------------------------------------------------- story stats --
jac = lambda a, b: len(a & b) / len(a | b) if a | b else 0.0
subfams = [("RAB3A", "RAB3B"), ("RAB27A", "RAB27B"), ("RAB39A", "RAB39B"), ("RAB5A", "RAB5B")]

print("\n--- overlap structure, threshold sets (score >= 0.65, Jaccard) ---")
sets = {g: set(long[(long.gene == g) & (long.score >= HI_CONF)].diseaseId) for g in rab.approvedSymbol}
for a, bb in subfams:
    print(f"  paralog Jaccard {a}-{bb}: {jac(sets[a], sets[bb]):.2f}")
print(f"  family-wide mean pairwise Jaccard: "
      f"{np.mean([jac(sets[a], sets[bb]) for a, bb in itertools.combinations(sets, 2)]):.2f}")

print("\n--- overlap structure, top-50 lists (shared diseases per pair /50) ---")
top50 = {sym: set(tk["topk_idx"][tid_to_row[tid], :50])
         for sym, tid in zip(rab.approvedSymbol, rab.targetId)}
shared = lambda a, b: len(top50[a] & top50[b])
for a, bb in subfams:
    print(f"  paralog shared {a}-{bb}: {shared(a, bb)}/50")
pair_means = [shared(a, bb) for a, bb in itertools.combinations(top50, 2)]
print(f"  family-wide mean pairwise shared: {np.mean(pair_means):.1f}/50")

"""Volcano of within-family predicted-disease overlap vs popularity-matched null.

Reads Outputs/family_overlap_results.csv (family_overlap_analysis.py).
Run: python3 retrain_2512/plot_family_volcano.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("/mnt/d/Research/OpenTargetsTransfer/retrain_2512/Outputs")
INK, MUTED, SEQ, ACCENT = "#1f2430", "#5b6472", "#3d6de0", "#c23934"

sub = pd.read_csv(OUT / "family_overlap_results_k30.csv")
logp = -np.log10(sub.p)
sig = sub.q < 0.05

fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=200)
ax.scatter(sub.excess[~sig], logp[~sig], s=np.sqrt(sub["size"][~sig]) * 3,
           c="#b8c0cc", alpha=0.7, lw=0)
ax.scatter(sub.excess[sig], logp[sig], s=np.sqrt(sub["size"][sig]) * 3,
           c=SEQ, alpha=0.75, lw=0)

rab = sub[sub.family.str.startswith("RAB, member")].iloc[0]
ax.scatter([rab.excess], [-np.log10(rab.p)], s=np.sqrt(rab["size"]) * 5, c=ACCENT, zorder=5)
ax.annotate("RAB GTPases", (rab.excess, -np.log10(rab.p)), xytext=(-10, -16),
            textcoords="offset points", ha="right", color=ACCENT, fontsize=9,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8))

# Staggered labels: top-3 excess, most negative, and the arbitrary-grouping control.
picks = [(r, off) for (_, r), off in zip(sub.nlargest(3, "excess").iterrows(),
                                         [(-6, -14), (-6, -30), (-6, -46)])]
neg = sub.nsmallest(1, "excess").iloc[0]
ctrl = sub[sub.family.str.contains("MicroRNA")].iloc[0]
picks += [(neg, (4, 10)), (ctrl, (10, 2))]
for r, (dx, dy) in picks:
    name = r.family if len(r.family) <= 34 else r.family[:32] + "…"
    if r.family == ctrl.family:
        name += " (arbitrary grouping)"
    ax.annotate(name, (r.excess, -np.log10(r.p)), xytext=(dx, dy),
                textcoords="offset points", color=MUTED, fontsize=7.5,
                ha="right" if dx < 0 else "left",
                arrowprops=dict(arrowstyle="-", color="#b8c0cc", lw=0.6))
ax.axvline(0, color=MUTED, lw=0.8, ls="--")
ax.set_xlabel("Excess within-family overlap (shared diseases per gene pair, top-30 lists)", color=INK)
ax.set_ylabel("−log10 empirical p", color=INK)
ax.set_title(f"Within-family sharing of predicted diseases vs popularity-matched null "
             f"({len(sub)} HGNC families)", loc="left", color=INK, fontsize=11)
ax.text(0.99, 0.02, "blue = BH q < 0.05; point size ∝ family size; p floor 1/10,001",
        transform=ax.transAxes, ha="right", color=MUTED, fontsize=7.5)
for s_ in ["top", "right"]:
    ax.spines[s_].set_visible(False)
ax.tick_params(colors=MUTED)
fig.tight_layout()
fig.savefig(OUT / "family_overlap_volcano.png", bbox_inches="tight")
print("saved family_overlap_volcano.png")

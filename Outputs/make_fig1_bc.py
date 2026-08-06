"""Regenerate Figure 1 panels B and C.

Table-driven: every number is read from a committed result artifact rather than
typed in, so the figure cannot drift from Tables 1 and 2 again.

Changes relative to the previous Fig1_BC.png. Titles, bar labels and styling are
otherwise identical to the original notebook.
  * OTRec CV ROC-AUC SD 0.007 -> 0.011 (per-fold value in CV_DL/oof_dl_summary.csv)
  * Disease Mean CV PR-AUC 0.482 -> 0.466, Open Targets Score 0.913/0.454 ->
    0.914/0.455 (both previously disagreed with Table 1)
  * Frozen BioClinical ModernBERT added to both panels, since it appears in both
    tables. Set INCLUDE_MODERNBERT = False to restore the original four/five bars.
  * Larger type (rebuttal promised figures at larger point size).

Usage:  python3 make_fig1_bc.py   (run from OTRec/Outputs)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["figure.dpi"] = 400
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.left"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#dddddd"
plt.rcParams["grid.linestyle"] = "-"
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["axes.axisbelow"] = True

C_ROC, C_PR = "#A6A6A6", "#005A9C"
# OTRec is highlighted in orange so it reads apart from the baselines at a glance.
# The light/dark relationship mirrors ROC/PR in the baseline palette; the black edge
# keeps the pair distinguishable in greyscale, where orange and grey are close.
C_ROC_HL, C_PR_HL = "#F5A24B", "#C75000"

# Set False to reproduce the original bar set (no ModernBERT).
INCLUDE_MODERNBERT = True


def load_panel_b():
    """Panel B from the canonical CV table, with the OTRec SD taken from per-fold data."""
    cv = pd.read_csv(HERE / "Table 1 - Performance in target-disjoint cross-validation.csv")
    cv.columns = [c.strip() for c in cv.columns]
    cv = cv.set_index(cv.columns[0])
    oof = pd.read_csv(HERE / "CV_DL" / "oof_dl_summary.csv").set_index("metric")

    rows = [
        ("OTRec", "OTRec (Deep Learning)"),
        ("OTTree\n(CatBoost)", "OTTree (CatBoost)"),
        ("Han et al.\n(XGB+Evidence)", "Han et al: XGB features + OT evidence"),
        ("Open\nTargets Score", "Open Targets Score"),
        ("Disease\nMean", "Disease Mean Baseline"),
    ]
    if INCLUDE_MODERNBERT:
        rows.insert(3, ("ModernBERT\n(frozen)", "Frozen BioClinical-ModernBERT + MLP"))

    labels, roc, roc_sd, pr, pr_sd = [], [], [], [], []
    for label, key in rows:
        r = cv.loc[key]
        labels.append(label)
        roc.append(float(r["ROC-AUC (Mean)"]))
        roc_sd.append(float(r["ROC-AUC ( ± SD)"]))
        pr.append(float(r["PR-AUC"]))
        pr_sd.append(float(r["PR-AUC ( ± SD)"]))

    # Authoritative per-fold SDs for OTRec (the table shipped 0.007 for ROC).
    roc[0] = float(oof.loc["auc", "mean"])
    roc_sd[0] = float(oof.loc["auc", "std"])
    pr[0] = float(oof.loc["pr_auc", "mean"])
    pr_sd[0] = float(oof.loc["pr_auc", "std"])
    return labels, roc, roc_sd, pr, pr_sd


def load_panel_c():
    """Panel C from the 5-seed temporal summaries."""
    t = pd.read_csv(HERE / "temporal_repeats_5seed" / "temporal_summary.csv").set_index("Model")
    mb = pd.read_csv(HERE / "temporal_frozen_encoder_mlp" / "temporal_summary.csv").set_index("Model")
    mb_row = mb.loc["Frozen BioClinical-ModernBERT + MLP"]

    rows = [
        ("OTRec", t.loc["OTRec"]),
        ("OTTree\n(CatBoost)", t.loc["OTTree (CatBoost)"]),
        ("Target\nMean", t.loc["Target Mean Baseline"]),
        ("Open\nTargets Score", t.loc["Open Targets Score"]),
    ]
    if INCLUDE_MODERNBERT:
        rows.insert(2, ("ModernBERT\n(frozen)", mb_row))

    labels = [r[0] for r in rows]
    roc = [float(r[1]["ROC-AUC"]) for r in rows]
    roc_sd = [float(r[1]["ROC-AUC SD"]) for r in rows]
    pr = [float(r[1]["PR-AUC"]) for r in rows]
    pr_sd = [float(r[1]["PR-AUC SD"]) for r in rows]
    return labels, roc, roc_sd, pr, pr_sd


def plot_panel(ax, models, rocs, roc_errs, prs, pr_errs, title, hide_y=False):
    x = np.arange(len(models))
    width = 0.35
    hl = [m.startswith("OTRec") for m in models]
    edges = ["black" if h else "none" for h in hl]
    widths = [1.1 if h else 0 for h in hl]
    bars1 = ax.bar(x - width / 2, rocs, width, label="ROC-AUC",
                   color=[C_ROC_HL if h else C_ROC for h in hl],
                   edgecolor=edges, linewidth=widths)
    bars2 = ax.bar(x + width / 2, prs, width, label="PR-AUC",
                   color=[C_PR_HL if h else C_PR for h in hl],
                   edgecolor=edges, linewidth=widths)

    for bars, errs in ((bars1, roc_errs), (bars2, pr_errs)):
        for bar, err in zip(bars, errs):
            if err and err > 0:
                ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            yerr=err, fmt="none", ecolor="black", capsize=3, elinewidth=1)
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + (err or 0) + 0.02,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=10, color="#333333")

    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, linespacing=1.25)
    for tick, h in zip(ax.get_xticklabels(), hl):
        if h:
            tick.set_fontweight("bold")
            tick.set_color(C_PR_HL)
    ax.set_ylim(0, 1.15)
    if hide_y:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    else:
        ax.set_ylabel("Score", fontsize=12, color="#333333")
        ax.tick_params(axis="y", labelsize=11)


def main():
    lb, rb, rbs, pb, pbs = load_panel_b()
    lc, rc, rcs, pc, pcs = load_panel_c()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 5.8), constrained_layout=True,
        gridspec_kw={"width_ratios": [len(lb), len(lc)]},
    )
    plot_panel(ax1, lb, rb, rbs, pb, pbs, "B. Target-Disjoint Cross-Validation (5x5-fold)")
    plot_panel(ax2, lc, rc, rcs, pc, pcs,
               "C. Temporal Validation (2025)", hide_y=True)

    # Explicit handles: a BarContainer legend entry would take its colour from the
    # first bar, which is now OTRec's orange, and mislabel the whole series.
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=C_ROC, label="ROC-AUC"),
        Patch(facecolor=C_PR, label="PR-AUC"),
        Patch(facecolor=C_ROC_HL, edgecolor="black", linewidth=1.1, label="OTRec ROC-AUC"),
        Patch(facecolor=C_PR_HL, edgecolor="black", linewidth=1.1, label="OTRec PR-AUC"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08),
               ncol=4, frameon=False, fontsize=12)

    for out in (HERE / "Fig1_BC.png", HERE / "Fig1_BC.pdf"):
        plt.savefig(out, dpi=400, bbox_inches="tight")
        print("wrote", out)
    for name, r, s, p, q in (("B", rb, rbs, pb, pbs), ("C", rc, rcs, pc, pcs)):
        print(f"panel {name}: ROC={[round(v,3) for v in r]} SD={[round(v,3) for v in s]}")
        print(f"           PR ={[round(v,3) for v in p]} SD={[round(v,3) for v in q]}")


if __name__ == "__main__":
    main()

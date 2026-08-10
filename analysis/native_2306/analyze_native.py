"""Downstream analyses on the native 23.06 -> 25.12 split.

Produces the native-split analogues of the manuscript's temporal tables:
  - per-disease shortlist utility (Hit@1/5/10, MRR), conservative worst-rank
    tie handling, ported verbatim from rebuttal_scratch/t3_t4b_final.py
  - cold-start / indication-count strata
  - seed stability summary for OTRec

Baseline predictions (Target Mean, Disease Mean, OT Score) are recomputed here
from the native train frame using the repo's own baseline definitions.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = Path("/mnt/d/Research/OpenTargetsTransfer/rebuttal_scratch/native_2306")


def per_disease(sub, col):
    hits = {1: [], 5: [], 10: []}
    mrrs, tiefrac = [], []
    for _, g in sub.groupby("diseaseId", sort=False):
        s = g[col].fillna(-np.inf).to_numpy(dtype=float)
        y = g.label.to_numpy()
        n = len(s)
        order = np.argsort(-s, kind="stable")
        srt, ysrt = s[order], y[order]
        worst = np.empty(n, dtype=int)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and srt[j + 1] == srt[i]:
                j += 1
            worst[i:j + 1] = j + 1
            i = j + 1
        pr = worst[ysrt == 1]
        best = pr.min()
        mrrs.append(1.0 / best)
        for k in hits:
            hits[k].append(1.0 if best <= k else 0.0)
        tiefrac.append((s == srt[0]).sum() / n)
    return dict(n_dis=sub.diseaseId.nunique(),
                **{f"Hit@{k}": float(np.mean(v)) for k, v in hits.items()},
                MRR=float(np.mean(mrrs)), top_tie_frac=float(np.mean(tiefrac)))


if __name__ == "__main__":
    train = pd.read_parquet(OUT / "native2306_train_frame.parquet")
    seed_files = sorted(OUT.glob("native2306_preds_s*.parquet"))
    assert seed_files, "no native OTRec predictions found"
    print(f"seeds available: {[f.stem.split('_s')[-1] for f in seed_files]}")

    # ---- seed stability -------------------------------------------------
    rows = []
    for f in seed_files:
        d = pd.read_parquet(f)
        rows.append({"seed": f.stem.split("_s")[-1],
                     "roc": roc_auc_score(d.label, d.otrec_native),
                     "pr": average_precision_score(d.label, d.otrec_native)})
    stab = pd.DataFrame(rows)
    print("\n=== OTRec seed stability (native split) ===")
    print(stab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"  ROC mean {stab.roc.mean():.4f} SD {stab.roc.std(ddof=1):.4f}"
          f" | PR mean {stab.pr.mean():.4f} SD {stab.pr.std(ddof=1):.4f}")
    stab.to_csv(OUT / "native2306_seed_stability.csv", index=False)

    # ---- assemble prediction frame with baselines ------------------------
    preds = pd.read_parquet(seed_files[0])[["diseaseId", "targetId", "label", "score_past", "n_ind_2306", "bin"]]
    for f in seed_files:
        s = f.stem.split("_s")[-1]
        preds[f"otrec_s{s}"] = pd.read_parquet(f)["otrec_native"].to_numpy()
    preds["otrec_mean"] = preds[[c for c in preds.columns if c.startswith("otrec_s")]].mean(axis=1)

    prior = float(train.label.mean())
    preds["target_mean"] = preds.targetId.map(train.groupby("targetId")["label"].mean()).fillna(prior)
    preds["disease_mean"] = preds.diseaseId.map(train.groupby("diseaseId")["label"].mean()).fillna(prior)

    ottree_f = OUT / "native2306_ottree_preds.parquet"
    if ottree_f.exists():
        preds["ottree"] = pd.read_parquet(ottree_f)["ottree"].to_numpy()

    MODELS = [c for c in ["otrec_s42", "otrec_mean", "ottree", "target_mean", "disease_mean", "score_past"]
              if c in preds.columns]

    # ---- pooled + strata -------------------------------------------------
    print("\n=== pooled and indication-count strata (seed 42 unless noted) ===")
    for b in ["all", "0", "1", ">=2"]:
        sub = preds if b == "all" else preds[preds.bin == b]
        if not len(sub) or not sub.label.sum():
            continue
        print(f"\n-- bin {b}: n={len(sub):,} pos={int(sub.label.sum()):,} ({sub.label.mean():.2%})")
        for m in MODELS:
            print(f"     {m:<14} ROC {roc_auc_score(sub.label, sub[m]):.4f}"
                  f"  PR {average_precision_score(sub.label, sub[m]):.4f}")

    # ---- per-disease shortlist -------------------------------------------
    pos_dis = preds.groupby("diseaseId")["label"].sum()
    sub = preds[preds.diseaseId.isin(pos_dis[pos_dis > 0].index)].copy()
    print(f"\n=== per-disease shortlist: {sub.diseaseId.nunique():,} diseases with >=1 later positive, "
          f"{len(sub):,} pairs, median {sub.groupby('diseaseId').size().median():.0f} candidates/disease ===")
    shortlist = []
    for m in MODELS:
        r = per_disease(sub, m)
        r["model"] = m
        shortlist.append(r)
        print(f"  {m:<14} Hit@1 {r['Hit@1']:.3f} Hit@5 {r['Hit@5']:.3f} "
              f"Hit@10 {r['Hit@10']:.3f} MRR {r['MRR']:.3f} tie {r['top_tie_frac']:.3f}")
    pd.DataFrame(shortlist).to_csv(OUT / "native2306_shortlist.csv", index=False)

    preds.to_parquet(OUT / "native2306_all_preds.parquet", index=False)
    print(f"\nsaved {OUT/'native2306_all_preds.parquet'}, native2306_shortlist.csv, native2306_seed_stability.csv")

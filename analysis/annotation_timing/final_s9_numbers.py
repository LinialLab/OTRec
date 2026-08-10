"""Aggregate all robustness runs into the final S9-ready numbers.

Emits:
  1. 22.02-native leak analysis, multi-seed: pooled/covered/uncovered ROC+PR
     mean +- SD over available seeds (42 + s43/s44/s45), alongside the paper
     config and token-stripped single-seed references.
  2. Native replication 23.06 -> 25.12, multi-seed: pooled ROC+PR mean +- SD,
     OT-Score temporal baseline, coverage stats.
  3. Per-subset annotation-free baselines (released CSV only, not for the
     paper table): OT-Score (score_past), Target Mean, Disease Mean.

Run after all GPU chains finish.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = Path("/mnt/d/Research/OpenTargetsTransfer/rebuttal_scratch")
NAT = OUT / "native_2306"


def metrics(y, p):
    return roc_auc_score(y, p), average_precision_score(y, p)


def fmt(vals):
    a = np.asarray(vals, dtype=float)
    if len(a) == 1:
        return f"{a[0]:.4f}"
    return f"{a.mean():.4f}±{a.std(ddof=1):.4f}"


# ---------------------------------------------------------------- part 1 ---
print("=" * 78)
print("1) 22.02-NATIVE LEAK ANALYSIS (multi-seed), stratified by 22.02 coverage")
print("=" * 78)

base = pd.read_parquet(OUT / "temporal_preds_seed42.parquet")[
    ["diseaseId", "targetId", "label", "otrec", "score_past"]]
stripped = pd.read_parquet(OUT / "temporal_preds_seed42_stripped.parquet")[
    ["diseaseId", "targetId", "otrec_stripped"]]
df = base.merge(stripped, on=["diseaseId", "targetId"])

d2202 = pd.read_parquet(OUT / "disease_df_2202.parquet")
t2202 = pd.read_parquet(OUT / "target_df_2202.parquet")
d_ok = set(d2202.loc[d2202.disease_text_embed.str.strip().ne(""), "diseaseId"])
t_ok = set(t2202.loc[t2202.target_text_embed.str.strip().ne(""), "targetId"])
df["covered"] = df.diseaseId.isin(d_ok) & df.targetId.isin(t_ok)

native_seeds = {}
for f in sorted(OUT.glob("temporal_preds_seed42_2202native*.parquet")):
    seed = f.stem.split("_s")[-1] if "_s" in f.stem else "42"
    native_seeds[seed] = pd.read_parquet(f)[["diseaseId", "targetId", "otrec_2202native"]]
print(f"22.02-native seeds found: {sorted(native_seeds)}")

subsets = {"pooled": df.index, "covered": df.index[df.covered], "uncovered": df.index[~df.covered]}
rows = []
for name, idx in subsets.items():
    sub = df.loc[idx]
    row = {"subset": name, "n": len(sub), "pos": int(sub.label.sum())}
    row["paper_roc"], row["paper_pr"] = metrics(sub.label, sub.otrec)
    row["stripped_roc"], row["stripped_pr"] = metrics(sub.label, sub.otrec_stripped)
    nat_roc, nat_pr = [], []
    for seed, preds in native_seeds.items():
        m = sub[["diseaseId", "targetId", "label"]].merge(preds, on=["diseaseId", "targetId"])
        assert len(m) == len(sub)
        r, p = metrics(m.label, m.otrec_2202native)
        nat_roc.append(r); nat_pr.append(p)
    row["native_roc"], row["native_pr"] = fmt(nat_roc), fmt(nat_pr)
    row["native_roc_vals"] = [round(v, 4) for v in nat_roc]
    rows.append(row)
res1 = pd.DataFrame(rows)
print(res1.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# covered-subset paired deltas per seed (the leak bound)
sub = df.loc[subsets["covered"]]
deltas = []
for seed, preds in native_seeds.items():
    m = sub[["diseaseId", "targetId", "label"]].merge(preds, on=["diseaseId", "targetId"])
    r, _ = metrics(m.label, m.otrec_2202native)
    deltas.append(res1.loc[res1.subset == "covered", "paper_roc"].iloc[0] - r)
print(f"\ncovered-subset ROC delta (paper - native) per seed: "
      f"{[round(d,4) for d in deltas]}  -> {fmt(deltas)}")

# ---------------------------------------------------------------- part 2 ---
print()
print("=" * 78)
print("2) NATIVE REPLICATION 23.06 -> 25.12 (fully release-native)")
print("=" * 78)

nat_files = sorted(NAT.glob("native2306_preds_s*.parquet"))
print(f"native replication seeds found: {[f.stem.split('_s')[-1] for f in nat_files]}")
if nat_files:
    roc_l, pr_l = [], []
    for f in nat_files:
        d = pd.read_parquet(f)
        r, p = metrics(d.label, d.otrec_native)
        roc_l.append(r); pr_l.append(p)
    d0 = pd.read_parquet(nat_files[0])
    roc_b, pr_b = metrics(d0.label, d0.score_past)
    print(f"test pairs {len(d0):,}, positives {int(d0.label.sum()):,} ({d0.label.mean():.2%})")
    print(f"OTRec native  ROC {fmt(roc_l)}  PR {fmt(pr_l)}")
    print(f"OT-Score temporal baseline  ROC {roc_b:.4f}  PR {pr_b:.4f}")

# ---------------------------------------------------------------- part 3 ---
print()
print("=" * 78)
print("3) Per-subset annotation-free baselines (released CSV only)")
print("=" * 78)

hist = pd.read_parquet("/mnt/d/Research/OpenTargetsTransfer/code/history_df.parquet")
tmean = hist.groupby("targetId")["label"].mean()
dmean = hist.groupby("diseaseId")["label"].mean()
prior = float(hist.label.mean())
df["target_mean"] = df.targetId.map(tmean).fillna(prior)
df["disease_mean"] = df.diseaseId.map(dmean).fillna(prior)

brows = []
for name, idx in subsets.items():
    sub = df.loc[idx]
    for col, label in [("score_past", "OT-Score temporal"), ("target_mean", "Target Mean"),
                       ("disease_mean", "Disease Mean")]:
        r, p = metrics(sub.label, sub[col])
        brows.append({"subset": name, "baseline": label, "roc": round(r, 4), "pr": round(p, 4)})
bdf = pd.DataFrame(brows)
print(bdf.to_string(index=False))

res1.drop(columns=["native_roc_vals"]).to_csv(OUT / "s9_table_numbers.csv", index=False)
bdf.to_csv(OUT / "s9_baselines_per_subset.csv", index=False)
print(f"\nsaved {OUT/'s9_table_numbers.csv'} and {OUT/'s9_baselines_per_subset.csv'}")

import pandas as pd, numpy as np

import os

# per-pair prediction parquets are large and not committed;
# set OTREC_SCRATCH to the directory holding them (default: ./rebuttal_scratch)
SCRATCH = os.environ.get("OTREC_SCRATCH", "rebuttal_scratch")

hist = pd.read_parquet("code/history_df.parquet")
fut  = pd.read_parquet("code/final_df.parquet")
print("history rows", len(hist), "positives", int(hist.label.sum()), "diseases", hist.diseaseId.nunique(), "targets", hist.targetId.nunique())
print("future  rows", len(fut),  "positives", int(fut.label.sum()),  "diseases", fut.diseaseId.nunique(),  "targets", fut.targetId.nunique())

join_keys=["diseaseId","targetId","label"]
test = (fut.merge(hist[join_keys], on=join_keys, how="left", indicator=True)
          .query('_merge=="left_only"').drop(columns="_merge").reset_index(drop=True))
print("\n=== T6 TEMPORAL TEST SET ===")
n=len(test); p=int(test.label.sum())
print(f"pairs={n}  positives={p}  pos_rate={p/n:.6f} ({100*p/n:.3f}%)")
print("diseases", test.diseaseId.nunique(), "targets", test.targetId.nunique())
print("CV frame (final_df) base rate:", fut.label.mean())

# T4a: indications per target in processed 2022 frame
ind = hist[hist.label==1].groupby("targetId")["diseaseId"].nunique()
tt = test.targetId.unique()
cnt = pd.Series(0, index=tt, dtype=int)
cnt.update(ind.reindex(tt).dropna().astype(int))
def binof(c): return "0" if c==0 else ("1" if c==1 else ">=2")
test["_n_ind_2022"] = test.targetId.map(cnt).fillna(0).astype(int)
test["_bin"] = test._n_ind_2022.map(binof)
print("\n=== T4a BINS (indications = distinct diseases with label==1 in processed 2022 frame) ===")
g = test.groupby("_bin").agg(n_pairs=("label","size"), n_positives=("label","sum"), n_targets=("targetId","nunique"))
g["pos_rate"]=g.n_positives/g.n_pairs
print(g.loc[["0","1",">=2"]].to_string())
print("\ntargets in test set total:", len(tt))
print("bin target counts:", {b:int((cnt.map(binof)==b).sum()) for b in ["0","1",">=2"]})
# also: do the 2022 frame's targets restrict test?
print("\ntest targets not present at all in 2022 frame:", len(set(tt)-set(hist.targetId.unique())))
test[["diseaseId","targetId","label","_n_ind_2022","_bin"]].to_parquet(SCRATCH+"/temporal_test_bins.parquet")

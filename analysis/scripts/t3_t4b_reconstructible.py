"""T3/T4b for the models whose temporal scores are exactly reconstructible without training."""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

import os

# per-pair prediction parquets are large and not committed;
# set OTREC_SCRATCH to the directory holding them (default: ./rebuttal_scratch)
SCRATCH = os.environ.get("OTREC_SCRATCH", "rebuttal_scratch")


hist = pd.read_parquet("code/history_df.parquet")
fut  = pd.read_parquet("code/final_df.parquet")
jk=["diseaseId","targetId","label"]
test=(fut.merge(hist[jk],on=jk,how="left",indicator=True).query('_merge=="left_only"')
        .drop(columns="_merge").reset_index(drop=True))
# historical OT score (score_past), exactly as run_temporal_repeated.add_historical_score
test = test.merge(hist[["diseaseId","targetId","score"]].rename(columns={"score":"score_past"}),
                  on=["diseaseId","targetId"], how="left")
prior=float(hist.label.mean())
tm = hist.groupby("targetId")["label"].mean(); dm = hist.groupby("diseaseId")["label"].mean()
test["TargetMean"]  = test.targetId.map(tm).fillna(prior).astype("float32")
test["DiseaseMean"] = test.diseaseId.map(dm).fillna(prior).astype("float32")
test["OTScore"]     = test["score_past"].astype("float32")   # NaN where pair absent in 2022

ind = hist[hist.label==1].groupby("targetId")["diseaseId"].nunique()
test["n_ind_2022"] = test.targetId.map(ind).fillna(0).astype(int)
test["bin"] = np.where(test.n_ind_2022==0,"0",np.where(test.n_ind_2022==1,"1",">=2"))

MODELS=["TargetMean","DiseaseMean","OTScore"]
print("=== sanity: pooled temporal metrics (should match Table 2) ===")
for m in MODELS:
    s=test[m].fillna(0.0) if m=="OTScore" else test[m]
    print(f"{m:12s} ROC {roc_auc_score(test.label,s):.4f}  PR {average_precision_score(test.label,s):.4f}")

print("\n=== T4b (reconstructible baselines only) per indication-count stratum ===")
rows=[]
for b in ["0","1",">=2"]:
    d=test[test.bin==b]
    r={"bin":b,"n_pairs":len(d),"n_targets":d.targetId.nunique(),"n_pos":int(d.label.sum()),
       "pos_rate":d.label.mean()}
    for m in MODELS:
        s=d[m].fillna(0.0)
        if d.label.nunique()<2 or s.nunique()<2:
            r[m+"_ROC"]=np.nan; r[m+"_PR"]=average_precision_score(d.label,s) if d.label.nunique()>1 else np.nan
        else:
            r[m+"_ROC"]=roc_auc_score(d.label,s); r[m+"_PR"]=average_precision_score(d.label,s)
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
# also merged <=1 stratum
d=test[test.bin.isin(["0","1"])]
print("\n<=1 indication combined: pairs %d targets %d pos %d rate %.4f"%(len(d),d.targetId.nunique(),d.label.sum(),d.label.mean()))
for m in MODELS:
    s=d[m].fillna(0.0)
    print(f"  {m:12s} ROC {roc_auc_score(d.label,s):.4f} PR {average_precision_score(d.label,s):.4f}  (unique score values: {s.nunique()})")

# ---------- T3 per-disease, conservative (worst-rank) tie handling ----------
pos_dis = test.groupby("diseaseId")["label"].sum()
keep = pos_dis[pos_dis>0].index
sub = test[test.diseaseId.isin(keep)].copy()
print(f"\n=== T3 per-disease: {sub.diseaseId.nunique()} diseases with >=1 temporal positive; {len(sub)} pairs ===")
print("candidates/disease: median %.0f mean %.1f min %d max %d"%(
    sub.groupby('diseaseId').size().median(), sub.groupby('diseaseId').size().mean(),
    sub.groupby('diseaseId').size().min(), sub.groupby('diseaseId').size().max()))

def per_disease(sub, col):
    hits={1:[],5:[],10:[]}; mrrs=[]; tiefrac=[]
    for did,g in sub.groupby("diseaseId",sort=False):
        s=g[col].fillna(-np.inf).to_numpy(dtype=float); y=g.label.to_numpy()
        n=len(s)
        # worst rank within tie block: rank = number of items with score >= s  (1-based, pessimistic)
        order=np.argsort(-s,kind="stable")
        srt=s[order]; ysrt=y[order]
        # worst rank for each item = last index of its tie block (1-based)
        worst=np.empty(n,dtype=int); i=0
        while i<n:
            j=i
            while j+1<n and srt[j+1]==srt[i]: j+=1
            worst[i:j+1]=j+1; i=j+1
        pr=worst[ysrt==1]
        best=pr.min()
        mrrs.append(1.0/best)
        for k in hits: hits[k].append(1.0 if best<=k else 0.0)
        top=srt[0]; tiefrac.append((s==top).sum()/n)
    return dict(n_dis=sub.diseaseId.nunique(),
                **{f"Hit@{k}":float(np.mean(v)) for k,v in hits.items()},
                MRR=float(np.mean(mrrs)), top_tie_frac=float(np.mean(tiefrac)))

res=[]
for m in MODELS:
    r=per_disease(sub,m); r["model"]=m; res.append(r)
print(pd.DataFrame(res)[["model","n_dis","Hit@1","Hit@5","Hit@10","MRR","top_tie_frac"]].to_string(index=False))

# orphan split: diseases with zero positives in the 2022 history frame
hist_pos_dis = set(hist[hist.label==1].diseaseId.unique())
sub["orphan_2022"] = ~sub.diseaseId.isin(hist_pos_dis)
for flag,name in [(True,"ORPHAN (no 2022 clinical target)"),(False,"NON-ORPHAN")]:
    s2=sub[sub.orphan_2022==flag]
    if s2.empty: continue
    print(f"\n--- {name}: {s2.diseaseId.nunique()} diseases, {len(s2)} pairs, {int(s2.label.sum())} positives ---")
    rr=[]
    for m in MODELS:
        r=per_disease(s2,m); r["model"]=m; rr.append(r)
    print(pd.DataFrame(rr)[["model","n_dis","Hit@1","Hit@5","Hit@10","MRR","top_tie_frac"]].to_string(index=False))
sub.to_parquet(SCRATCH+"/temporal_test_with_baselines.parquet")

import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

import os

# per-pair prediction parquets are large and not committed;
# set OTREC_SCRATCH to the directory holding them (default: ./rebuttal_scratch)
SCRATCH = os.environ.get("OTREC_SCRATCH", "rebuttal_scratch")


preds = pd.read_parquet(SCRATCH+"/temporal_preds_seed42.parquet")  # diseaseId,targetId,label,score_past,otrec,ottree
hist = pd.read_parquet("code/history_df.parquet")
n2v  = pd.read_parquet(SCRATCH+"/node2vec_temporal_preds.parquet")  # diseaseId,targetId,label,pred
bert = pd.read_parquet("OTRec/Outputs/frozen_encoder_mlp_bioclinical_modernbert_base/test_predictions.parquet")  # diseaseId,targetId,label,raw_cosine_score -- NOTE: this is NOT the temporal set (see caveat)

prior = float(hist.label.mean())
tm = hist.groupby("targetId")["label"].mean(); dm = hist.groupby("diseaseId")["label"].mean()
preds["TargetMean"]  = preds.targetId.map(tm).fillna(prior).astype("float32")
preds["DiseaseMean"] = preds.diseaseId.map(dm).fillna(prior).astype("float32")
preds["OTScore"]     = preds["score_past"].astype("float32")

ind = hist[hist.label==1].groupby("targetId")["diseaseId"].nunique()
preds["n_ind_2022"] = preds.targetId.map(ind).fillna(0).astype(int)
preds["bin"] = np.where(preds.n_ind_2022==0,"0",np.where(preds.n_ind_2022==1,"1",">=2"))

MODELS=["otrec","ottree","TargetMean","DiseaseMean","OTScore"]
NICE={"otrec":"OTRec","ottree":"OTTree","TargetMean":"TargetMean","DiseaseMean":"DiseaseMean","OTScore":"OTScore"}

print("=== sanity: pooled (this run) ===")
for m in MODELS:
    s=preds[m]
    print(f"{NICE[m]:12s} ROC {roc_auc_score(preds.label,s):.4f}  PR {average_precision_score(preds.label,s):.4f}")

print("\n=== T4b: OTRec/OTTree + baselines per indication-count stratum (seed42 reproduction) ===")
rows=[]
for b in ["0","1",">=2"]:
    d=preds[preds.bin==b]
    r={"bin":b,"n_pairs":len(d),"n_targets":d.targetId.nunique(),"n_pos":int(d.label.sum()),"pos_rate":d.label.mean()}
    for m in MODELS:
        s=d[m]
        if d.label.nunique()<2:
            r[NICE[m]+"_ROC"]=np.nan; r[NICE[m]+"_PR"]=np.nan
        else:
            r[NICE[m]+"_ROC"]=roc_auc_score(d.label,s); r[NICE[m]+"_PR"]=average_precision_score(d.label,s)
    rows.append(r)
tb=pd.DataFrame(rows)
print(tb.to_string(index=False))

d=preds[preds.bin.isin(["0","1"])]
print("\n<=1 indication combined: pairs %d targets %d pos %d rate %.4f"%(len(d),d.targetId.nunique(),d.label.sum(),d.label.mean()))
for m in MODELS:
    s=d[m]
    print(f"  {NICE[m]:12s} ROC {roc_auc_score(d.label,s):.4f} PR {average_precision_score(d.label,s):.4f}")

d2=preds[preds.bin==">=2"]
print("\n>=2 indication (for contrast): pairs %d pos %d rate %.4f"%(len(d2),d2.label.sum(),d2.label.mean()))
for m in MODELS:
    s=d2[m]
    print(f"  {NICE[m]:12s} ROC {roc_auc_score(d2.label,s):.4f} PR {average_precision_score(d2.label,s):.4f}")

# ---------- T3 per-disease, conservative worst-rank tie handling ----------
def per_disease(sub, col):
    hits={1:[],5:[],10:[]}; mrrs=[]; tiefrac=[]
    for did,g in sub.groupby("diseaseId",sort=False):
        s=g[col].fillna(-np.inf).to_numpy(dtype=float); y=g.label.to_numpy()
        n=len(s)
        order=np.argsort(-s,kind="stable")
        srt=s[order]; ysrt=y[order]
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

pos_dis = preds.groupby("diseaseId")["label"].sum()
keep = pos_dis[pos_dis>0].index
sub = preds[preds.diseaseId.isin(keep)].copy()
print(f"\n=== T3 per-disease (this run): {sub.diseaseId.nunique()} diseases with >=1 temporal positive; {len(sub)} pairs ===")
print("candidates/disease: median %.0f mean %.1f min %d max %d"%(
    sub.groupby('diseaseId').size().median(), sub.groupby('diseaseId').size().mean(),
    sub.groupby('diseaseId').size().min(), sub.groupby('diseaseId').size().max()))

res=[]
for m in MODELS:
    r=per_disease(sub,m); r["model"]=NICE[m]; res.append(r)
print(pd.DataFrame(res)[["model","n_dis","Hit@1","Hit@5","Hit@10","MRR","top_tie_frac"]].to_string(index=False))

hist_pos_dis = set(hist[hist.label==1].diseaseId.unique())
sub["orphan_2022"] = ~sub.diseaseId.isin(hist_pos_dis)
for flag,name in [(True,"ORPHAN (no 2022 clinical target)"),(False,"NON-ORPHAN")]:
    s2=sub[sub.orphan_2022==flag]
    print(f"\n--- {name}: {s2.diseaseId.nunique()} diseases, {len(s2)} pairs, {int(s2.label.sum())} positives ---")
    rr=[]
    for m in MODELS:
        r=per_disease(s2,m); r["model"]=NICE[m]; rr.append(r)
    print(pd.DataFrame(rr)[["model","n_dis","Hit@1","Hit@5","Hit@10","MRR","top_tie_frac"]].to_string(index=False))

# --- ModernBERT & Node2Vec provenance check for T3/T4b eligibility ---
print("\n=== ModernBERT test_predictions.parquet vs TRUE temporal test set ===")
k=lambda d: set(zip(d.diseaseId,d.targetId))
ov = len(k(bert) & k(preds))
print(f"overlap: {ov} / {len(bert)} bert rows; {ov/len(preds)*100:.1f}% of true temporal set covered -> NOT the temporal set, exclude from T3/T4b")

print("\n=== Node2Vec (real temporal run) sanity ===")
print("rows", len(n2v), "pos", int(n2v.label.sum()))
print("ROC", roc_auc_score(n2v.label, n2v.pred), "PR", average_precision_score(n2v.label, n2v.pred))

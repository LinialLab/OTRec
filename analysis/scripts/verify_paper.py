import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

print("="*70); print("A) Table 1 SD reconciliation (paper caption: 'mean +/- SD over 25 folds')")
r = pd.read_csv("OTRec/Outputs/CV_DL/oof_dl_results.csv")
print("  n folds:", len(r))
print("  ROC-AUC: mean %.4f | SD over 25 folds (ddof=1) %.4f | ddof=0 %.4f"%(r.auc.mean(), r.auc.std(ddof=1), r.auc.std(ddof=0)))
rep = r.groupby("repeat").auc.mean()
print("  SD across 5 REPEAT-MEANS (ddof=1): %.4f  <- paper Table 1 says 0.007"%rep.std(ddof=1))
print("  PR-AUC:  mean %.4f | SD over 25 folds %.4f | SD across repeat-means %.4f  <- paper says 0.017"%(
      r.pr_auc.mean(), r.pr_auc.std(ddof=1), r.groupby('repeat').pr_auc.mean().std(ddof=1)))

print("\n"+"="*70); print("B) Table 3 shortlist utility: reproduce from OOF preds, both tie policies")
dl = pd.read_parquet("OTRec/Outputs/CV_DL/oof_dl_preds.parquet", columns=["diseaseId","targetId","label","pred"])
cb = pd.read_parquet("OTRec/Outputs/CV_tree/CB_5_cv.parquet", columns=["diseaseId","targetId","label","pred"])
m = dl.merge(cb, on=["diseaseId","targetId"], suffixes=("_dl","_cb"))
assert (m.label_dl==m.label_cb).all()
m = m.rename(columns={"label_dl":"label"})
posd = m.groupby("diseaseId").label.sum()
keep = posd[posd>0].index
sub = m[m.diseaseId.isin(keep)]
print("  diseases with >=1 held-out positive:", sub.diseaseId.nunique(), " (paper says 2,329)")

def shortlist(sub, col, tie="worst"):
    H={1:[],5:[],10:[],25:[]}; mrr=[]; p5=[]; r5=[]
    for d,g in sub.groupby("diseaseId",sort=False):
        s=g[col].to_numpy(float); y=g.label.to_numpy(); n=len(s)
        o=np.argsort(-s,kind="stable"); srt=s[o]; ys=y[o]
        if tie=="worst":
            rk=np.empty(n,int); i=0
            while i<n:
                j=i
                while j+1<n and srt[j+1]==srt[i]: j+=1
                rk[i:j+1]=j+1; i=j+1
        else:
            rk=np.arange(1,n+1)
        pr=rk[ys==1]; best=pr.min(); mrr.append(1/best)
        for k in H: H[k].append(1.0 if best<=k else 0.0)
        top5=ys[:min(5,n)]
        p5.append(top5.sum()/min(5,n)); r5.append(top5.sum()/max(1,y.sum()))
    return {f"Hit@{k}":np.mean(v) for k,v in H.items()} | {"MRR":np.mean(mrr),"P@5":np.mean(p5),"R@5":np.mean(r5)}

for tie in ["worst","optimistic"]:
    print(f"  --- tie policy: {tie} ---")
    for name,col in [("OTRec","pred_dl"),("OTTree","pred_cb")]:
        z=shortlist(sub,col,tie)
        print(f"    {name:7s} Hit@1 {z['Hit@1']:.3f} Hit@5 {z['Hit@5']:.3f} P@5 {z['P@5']:.3f} R@5 {z['R@5']:.3f} Hit@10 {z['Hit@10']:.3f} Hit@25 {z['Hit@25']:.3f} MRR {z['MRR']:.3f}")
print("  paper:  OTRec  Hit@1 0.682 Hit@5 0.806 P@5 0.645 R@5 0.321 Hit@10 0.848 Hit@25 0.897 MRR 0.741")
print("  paper:  OTTree Hit@1 0.487 Hit@5 0.732 P@5 0.537 R@5 0.272 Hit@10 0.848 Hit@25 0.951 MRR 0.600")

print("\n"+"="*70); print("C) 0.65-threshold in-distribution precision/recall (paper: 0.92 / 0.62)")
y=dl.label.to_numpy(); p=dl.pred.to_numpy()
for t in [0.6,0.65,0.7]:
    yp=(p>=t)
    print(f"  thr {t}: precision {precision_score(y,yp):.4f} recall {recall_score(y,yp):.4f} n_pred_pos {yp.sum()}")

print("\n"+"="*70); print("D) positive-bearing disease counts")
fut=pd.read_parquet("code/final_df.parquet"); hist=pd.read_parquet("code/history_df.parquet")
print("  CV frame (25.06): diseases total %d | positive-bearing %d   (paper: >12,000 total, ~6,000 pos-bearing)"%(
      fut.diseaseId.nunique(), fut[fut.label==1].diseaseId.nunique()))
print("  2022 frame:       diseases total %d | positive-bearing %d"%(
      hist.diseaseId.nunique(), hist[hist.label==1].diseaseId.nunique()))
print("  CV frame targets %d | 2022 frame targets %d  (paper: training restricted to ~1,500 targets)"%(
      fut.targetId.nunique(), hist.targetId.nunique()))

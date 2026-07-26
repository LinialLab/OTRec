import pandas as pd, numpy as np, os
print("="*70); print("E) SD provenance for every Table-1 row")
cb=pd.read_csv("OTRec/Outputs/CV_tree/CB_5_cv_foldMetrics.csv")
print("  OTTree fold metrics cols:", list(cb.columns)[:8], "n=",len(cb))
for c in cb.columns:
    if "auc" in c.lower(): print(f"    {c}: mean {cb[c].mean():.4f} sd25 {cb[c].std(ddof=1):.4f}")
print("  paper OTTree: 0.947 +/- 0.005 | 0.772 +/- 0.023")
for f in ["OTRec/Outputs/CV_frozen_encoder_mlp/frozen_encoder_all25_fold_metrics.csv",
          "OTRec/Outputs/CV_frozen_encoder_mlp/frozen_encoder_fold_metrics.csv"]:
    d=pd.read_csv(f); print(f"  {os.path.basename(f)} n={len(d)}")
    for c in d.columns:
        if c in ("auc","roc_auc","pr_auc"): print(f"    {c}: mean {d[c].mean():.4f} sd {d[c].std(ddof=1):.4f}")
print("  paper ModernBERT: 0.912 +/- 0.008 | 0.666 +/- 0.028")
alt="code/Outputs/CV_DL/oof_dl_results.csv"
if os.path.exists(alt):
    a=pd.read_csv(alt); print(f"  ALT copy {alt}: n={len(a)} auc mean {a.auc.mean():.4f} sd {a.auc.std(ddof=1):.4f} | pr mean {a.pr_auc.mean():.4f} sd {a.pr_auc.std(ddof=1):.4f}")
else: print("  no alternate CV_DL results copy")

print("\n"+"="*70); print("F) '~6,000 positive-bearing diseases' claim")
fut=pd.read_parquet("code/final_df.parquet")
print("  final_df diseases with >=1 positive:", fut[fut.label==1].diseaseId.nunique())
try:
    kd=pd.read_parquet("data/opentargets/known_drug", columns=["diseaseId","targetId","phase"])
    print("  OTP 25.x known_drug: distinct diseases %d | distinct targets %d"%(kd.diseaseId.nunique(), kd.targetId.nunique()))
    print("  known_drug phase>=1: distinct diseases %d"%kd[kd.phase>=1].diseaseId.nunique())
except Exception as e: print("  known_drug read err:", e)

print("\n"+"="*70); print("G) Discovery-example scores claimed in the paper")
s2=pd.read_csv("OTRec/Outputs/S2-DL_novel+known_candidates.csv")
s1=pd.read_csv("OTRec/Outputs/S1-DL_novel_predictions.csv")
checks=[("EFO_0000326",["POLA1","POLA2"],"CNS cancer: POLA1 0.961 / POLA2 0.966"),
        (None,["SCN8A","SCN1A"],"CDKL5: SCN8A 0.991 / SCN1A 0.989"),
        (None,["PDE4C"],"lim cut systemic sclerosis: 0.924"),
        (None,["DHODH"],"giant cell arteritis: 0.829"),
        (None,["IL6R"],"ulcerative colitis")]
for did,syms,note in checks:
    print(f"  -- {note}")
    for sym in syms:
        d=s2[s2.targetSymbol==sym]
        if did: d=d[d.diseaseId==did]
        if len(d)==0: print(f"     {sym}: no rows"); continue
        top=d.nlargest(3,"score")[["diseaseId","diseaseName","score","source"]]
        for _,r in top.iterrows():
            print(f"     {sym}: {r.score:.3f}  {r.diseaseName[:52]}  [{r.source}]")
for kw,sym in [("CDKL5","SCN8A"),("systemic sclerosis","PDE4C"),("arteritis","DHODH"),("ulcerative colitis","IL6R")]:
    d=s2[(s2.diseaseName.str.contains(kw,case=False,na=False))&(s2.targetSymbol==sym)]
    if len(d): print(f"  MATCH {kw} x {sym}: "+", ".join(f"{r.score:.3f} ({r.diseaseName[:40]}, {r.source})" for _,r in d.nlargest(2,'score').iterrows()))
    else: print(f"  MATCH {kw} x {sym}: NOT FOUND in S2")

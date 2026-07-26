import pandas as pd, numpy as np

import os

# per-pair prediction parquets are large and not committed;
# set OTREC_SCRATCH to the directory holding them (default: ./rebuttal_scratch)
SCRATCH = os.environ.get("OTREC_SCRATCH", "rebuttal_scratch")

s1=pd.read_csv("OTRec/Outputs/S1-DL_novel_predictions.csv")
s2=pd.read_csv("OTRec/Outputs/S2-DL_novel+known_candidates.csv")
print("=== IL6R / ulcerative colitis ===")
print("  S2 rows w/ diseaseName containing 'colitis':", s2[s2.diseaseName.str.contains("colitis",case=False,na=False)].diseaseName.nunique(), "diseases")
uc=s2[s2.diseaseName.str.lower().str.strip()=="ulcerative colitis"]
print("  S2 rows for exactly 'ulcerative colitis':", len(uc), "| targets:", sorted(uc.targetSymbol.unique())[:15])
print("  IL6R rows anywhere in S2:", len(s2[s2.targetSymbol=="IL6R"]), "| in S1:", len(s1[s1.targetSymbol=="IL6R"]))
fut=pd.read_parquet("code/final_df.parquet")
dd=pd.read_parquet("code/copy_proc/disease_df.parquet",columns=["diseaseId","name"])
td=pd.read_parquet("code/copy_proc/target_df.parquet",columns=["targetId","approvedSymbol"])
ucid=dd[dd.name.str.lower().str.strip()=="ulcerative colitis"].diseaseId.tolist()
il6r=td[td.approvedSymbol=="IL6R"].targetId.tolist()
print("  ulcerative colitis diseaseId(s):",ucid," IL6R targetId:",il6r)
row=fut[(fut.diseaseId.isin(ucid))&(fut.targetId.isin(il6r))]
print("  IL6R x UC in 25.06 frame:", row[["diseaseId","targetId","score","label"]].to_dict("records"))

print("\n=== exact scores for the other named examples ===")
for sym,kw,claim in [("SCN8A","CDKL5","0.991"),("SCN1A","CDKL5","0.989"),
                     ("PDE4C","limited cutaneous systemic sclerosis","0.924"),
                     ("DHODH","temporal arteritis","0.829")]:
    d=s2[(s2.targetSymbol==sym)&(s2.diseaseName.str.contains(kw,case=False,na=False))]
    for _,r in d.iterrows():
        print(f"  {sym} x {r.diseaseName[:48]:50s} score {r.score:.3f} [{r.source}]  paper claims {claim}")
    if len(d)==0: print(f"  {sym} x {kw}: NOT FOUND")

print("\n=== POLA1/POLA2 x CNS cancer in the TEMPORAL model (paper: 0.961 / 0.966, OT known-drug ~0.06 in 2022) ===")
tp=pd.read_parquet(SCRATCH+"/temporal_preds_seed42.parquet")
pol=td[td.approvedSymbol.isin(["POLA1","POLA2"])]
m=tp[(tp.diseaseId=="EFO_0000326")&(tp.targetId.isin(pol.targetId))].merge(pol,on="targetId")
print(m[["approvedSymbol","label","score_past","otrec","ottree"]].to_string(index=False) if len(m) else "  not in temporal test set")
# where do they rank within that disease?
dis=tp[tp.diseaseId=="EFO_0000326"].copy()
if len(dis):
    dis["rank"]=dis.otrec.rank(ascending=False,method="max")
    mm=dis.merge(pol,on="targetId")
    print("  CNS cancer temporal candidates:",len(dis),"| POLA rank:",mm[["approvedSymbol","otrec","rank","label"]].to_dict("records"))

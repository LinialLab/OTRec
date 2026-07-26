import pandas as pd, numpy as np
print("=== H) druggable genome list (paper: ~4,600 protein-coding Finan genes) ===")
fg=pd.read_csv("data/finan_proc_druggable_genome_list.csv")
print("  rows:",len(fg),"cols:",list(fg.columns)[:6])
for c in fg.columns[:4]:
    print(f"   {c}: {fg[c].nunique()} unique")

print("\n=== I) OTP drug index claims (paper: N=18,081 entities; 10,956 phase>=1; SM 14,848 (82%); AB 963; prot/pept 830; oligo 159; ADC 119; gene ther 117; cell ther 52) ===")
try:
    dm=pd.read_parquet("data/opentargets/drug_molecule")
    print("  drug_molecule rows:",len(dm),"cols:",[c for c in dm.columns][:12])
    for c in ["drugType","maximumClinicalTrialPhase"]:
        if c in dm.columns:
            print(f"   {c} value counts:\n{dm[c].value_counts().head(10).to_string()}")
    if "maximumClinicalTrialPhase" in dm.columns:
        print("   phase>=1:",(dm.maximumClinicalTrialPhase>=1).sum())
except Exception as e: print("  ERR",e)

print("\n=== J) OTP target biotypes (paper: 20,130 protein-coding; lncRNA 34,882; miRNA 1,879) ===")
try:
    tg=pd.read_parquet("data/opentargets/target", columns=["id","biotype"])
    print("  target table rows:",len(tg))
    print(tg.biotype.value_counts().head(8).to_string())
except Exception as e: print("  ERR",e)

print("\n=== K) inference scope arithmetic (paper: ~4,600 genes x ~19,000 diseases = ~87M pairs) ===")
print("  4600*19000 = %.1fM"%(4600*19000/1e6))
dd=pd.read_parquet("code/copy_proc/disease_df.parquet",columns=["diseaseId"])
print("  diseases in processed disease frame:",len(dd))

print("\n=== L) InterFeat rerank artefacts ===")
import glob,os
for f in glob.glob("OTRec/Outputs/InterFeat_reranked_candidates/*.csv"):
    d=pd.read_csv(f); print(f"  {os.path.basename(f)}: {len(d)} rows, cols {list(d.columns)[:7]}")

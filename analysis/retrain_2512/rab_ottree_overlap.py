"""Does OTTree show the same popularity structure and RAB overlap as OTRec?

Trains one OTTree (CatBoost, comparison-lookup recipe) on the full 25.12 frame,
scores the 64 HGNC RAB genes plus 500 random genes against ALL 46,960 diseases,
and recomputes the overlap statistics under OTTree's own top-30 lists:
hub concentration, random-pair baseline, RAB pair overlap, list composition.

Caveats (exploratory): single fit on all labeled pairs (RAB genes appear only
as negatives); diseases outside the training frame are scored via text features
with diseaseId treated as an unseen category.

Run: python3 retrain_2512/rab_ottree_overlap.py
Outputs: Outputs/rab_ottree_top30.csv + printed summary.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

R = Path("/mnt/d/Research/OpenTargetsTransfer/retrain_2512")
OUT = R / "Outputs"
K, N_SAMPLE, SEED = 30, 500, 42
rng = np.random.default_rng(SEED)

df_learn = pd.read_parquet(R / "df_learn_2512.parquet")
tgt = pd.read_parquet(R / "gradio_artifacts" / "target_df.parquet")
dis = pd.read_parquet(R / "gradio_artifacts" / "disease_df.parquet")
feats = ["disease_text", "target_text", "diseaseId"]

m = CatBoostClassifier(depth=8, eval_metric="AUC", random_seed=SEED, verbose=200)
m.fit(Pool(df_learn[feats], df_learn["label"],
           text_features=["disease_text", "target_text"], cat_features=["diseaseId"]))
print("trained on", len(df_learn), "pairs", flush=True)

tid_row = {t: i for i, t in enumerate(tgt.targetId)}
h = pd.read_csv(R / "data_hgnc_complete_set.txt", sep="\t", low_memory=False,
                usecols=["symbol", "gene_group", "ensembl_gene_id"]).dropna()
rabh = h[h.gene_group.str.contains("RAB, member RAS oncogene GTPases", regex=False)
         & h.ensembl_gene_id.isin(tid_row)]
rab_rows = np.array(sorted(tid_row[e] for e in set(rabh.ensembl_gene_id)))
sample_rows = rng.choice(len(tgt), N_SAMPLE, replace=False)
score_rows = np.unique(np.r_[rab_rows, sample_rows])

dis_text = dis["disease_text"].astype(str).to_numpy()
dis_ids = dis["diseaseId"].astype(str).to_numpy()
n_dis = len(dis)

top_idx = {}
for j, g in enumerate(score_rows):
    frame = pd.DataFrame({"disease_text": dis_text,
                          "target_text": str(tgt.target_text.iloc[g]),
                          "diseaseId": dis_ids})
    pr = m.predict_proba(Pool(frame[feats], text_features=["disease_text", "target_text"],
                              cat_features=["diseaseId"]))[:, 1]
    top_idx[g] = np.argpartition(-pr, K)[:K]
    if (j + 1) % 50 == 0:
        print(f"  scored {j + 1}/{len(score_rows)} genes", flush=True)

mem = np.array([np.sort(top_idx[g]) for g in score_rows])
n_d = np.bincount(mem.ravel(), minlength=n_dis)
order = np.argsort(-n_d)
print(f"\nOTTree top-{K} structure ({len(score_rows)} genes):")
print(f"  distinct diseases used: {(n_d > 0).sum():,}")
print(f"  top-100 diseases fill {n_d[order[:100]].sum() / (len(score_rows) * K):.0%} of slots")
print("  top-8 hub diseases:")
for i in order[:8]:
    print(f"    {dis.name.iloc[i][:50]:50s} {n_d[i] / len(score_rows):.0%} of genes")

pos = {g: i for i, g in enumerate(score_rows)}
S = mem
def shared(a, b):
    return len(np.intersect1d(S[pos[a]], S[pos[b]], assume_unique=True))

samp = [g for g in sample_rows if g not in set(rab_rows)]
rand_pairs = [(samp[i], samp[j]) for i in range(len(samp)) for j in range(i + 1, len(samp))]
rnd = np.array([shared(a, b) for a, b in rand_pairs])
p95 = np.percentile(rnd, 95)
print(f"\nrandom-pair baseline (OTTree): mean {rnd.mean():.1f}/{K}, 95th pct {p95:.0f} "
      f"({len(rnd):,} pairs)")

rp = np.array([shared(a, b) for i, a in enumerate(rab_rows) for b in rab_rows[i + 1:]])
print(f"RAB pairs (OTTree): mean {rp.mean():.1f}, median {np.median(rp):.0f}, "
      f"range {rp.min()}-{rp.max()}, above 95th pct: {(rp > p95).mean():.0%}")

c_d = np.bincount(np.array([np.sort(top_idx[g]) for g in rab_rows]).ravel(), minlength=n_dis)
rec = np.argsort(-c_d)[:12]
sym = {tid_row[e]: s for e, s in zip(tgt.targetId, tgt.approvedSymbol)}
print("\nmost recurrent diseases in RAB OTTree top-30:")
for i in rec:
    print(f"  {dis.name.iloc[i][:50]:50s} {c_d[i]}/{len(rab_rows)}  (sample-wide {n_d[i]/len(score_rows):.0%})")

rows = []
for g in rab_rows:
    for i in top_idx[g]:
        rows.append({"gene": sym.get(g, tgt.approvedSymbol.iloc[g]),
                     "disease": dis.name.iloc[i], "diseaseId": dis_ids[i]})
pd.DataFrame(rows).to_csv(OUT / "rab_ottree_top30.csv", index=False)
print("\nsaved rab_ottree_top30.csv")

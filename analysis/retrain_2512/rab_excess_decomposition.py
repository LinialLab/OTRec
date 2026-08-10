"""Decompose the RAB family's excess overlap (+8.65/50) disease by disease.

Per-disease contribution to mean pairwise shared-count, for the 64-member HGNC
RAB group (the family-analysis definition):
  contribution(d) = c_d(c_d-1)/(m(m-1)) - n_d(n_d-1)/(N(N-1))
where c_d = RAB genes with d in their top-50, n_d = all genes with d in top-50.
Contributions sum exactly to obs - exact_null (33.25 - 24.60 = 8.65).

Run: python3 retrain_2512/rab_excess_decomposition.py
Output: Outputs/rab_excess_decomposition.csv (sorted by contribution).
"""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/mnt/d/Research/OpenTargetsTransfer/retrain_2512")
top50 = np.load(R / "Outputs" / "gene_topk.npz")["topk_idx"][:, :50]
N = len(top50)
tgt = pd.read_parquet(R / "gradio_artifacts" / "target_df.parquet")
dis = pd.read_parquet(R / "gradio_artifacts" / "disease_df.parquet")
tid_row = {t: i for i, t in enumerate(tgt.targetId)}

h = pd.read_csv(R / "data_hgnc_complete_set.txt", sep="\t", low_memory=False,
                usecols=["gene_group", "ensembl_gene_id"]).dropna()
h = h[h.gene_group.str.contains("RAB, member RAS oncogene GTPases")
      & h.ensembl_gene_id.isin(tid_row)]
rows = np.array([tid_row[e] for e in sorted(set(h.ensembl_gene_id))])
m = len(rows)

n_d = np.bincount(top50.ravel(), minlength=len(dis)).astype(float)
c_d = np.bincount(top50[rows].ravel(), minlength=len(dis)).astype(float)
contrib = c_d * (c_d - 1) / (m * (m - 1)) - n_d * (n_d - 1) / (N * (N - 1))
print(f"{m} RAB-group genes; obs {(c_d*(c_d-1)).sum()/(m*(m-1)):.2f} "
      f"null {(n_d*(n_d-1)).sum()/(N*(N-1)):.2f} excess {contrib.sum():.2f}")

out = pd.DataFrame({
    "disease": dis.name.fillna(dis.diseaseId), "diseaseId": dis.diseaseId,
    "rab_genes_top50": c_d.astype(int), "all_genes_top50_pct": (n_d / N).round(4),
    "enrichment": ((c_d / m) / np.maximum(n_d / N, 1e-12)).round(2),
    "excess_contribution": contrib.round(4),
})
out = out[(out.rab_genes_top50 > 0) | (out.excess_contribution != 0)] \
    .sort_values("excess_contribution", ascending=False)
out.to_csv(R / "Outputs" / "rab_excess_decomposition.csv", index=False)
top10 = out.head(10)
onc = sum(any(k in d.lower() for k in ["cancer", "carcinoma", "leukemia", "neoplasm",
                                       "tumor", "lymphoma", "granuloma", "melanoma",
                                       "myelodysplastic"])
          for d in out.head(50).disease)
print(f"top-10 share of excess {top10.excess_contribution.sum()/contrib.sum():.0%}; "
      f"oncology terms in top-50 contributors: {onc}/50")

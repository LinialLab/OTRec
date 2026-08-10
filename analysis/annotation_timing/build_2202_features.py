"""Build disease_df / target_df text features natively from OT Release 22.02.

Ports the exact text-construction recipe from code/0-OT-PreProcess_Recc.ipynb
(disease_text_embed cells ~476-564, target_text_embed cells ~578-659), but reads
every input table from data/historical_ot/22_02/ instead of the newer OTP
snapshot used by copy_proc/{disease_df,target_df}.parquet. This eliminates any
post-2022 information from annotation text by construction (not just the
clinical-precedence tractability tokens the strip ablation targets).

Port fidelity is verified by test_build_2202_features.py, which runs these same
functions against the newer snapshot and checks they reproduce the notebook's
committed copy_proc frames exactly.

No repo file is modified. Outputs go to rebuttal_scratch/.
"""
import numpy as np
import pandas as pd

RAW_2202 = "/mnt/d/Research/OpenTargetsTransfer/data/historical_ot/22_02"
OUT = "/mnt/d/Research/OpenTargetsTransfer/rebuttal_scratch"

# Pre-25.03 era table names (camelCase plural). The newer snapshot uses
# snake_case singular -- see test_build_2202_features.py.
TABLES_2202 = {
    "diseases": "diseases",
    "targets": "targets",
    "diseaseToPhenotype": "diseaseToPhenotype",
}

TRACTABILITY_CLINICAL_RE = "Approved Drug|Advanced Clinical|Phase 1 Clinical|Clinical Precedence"


def _exact_synonyms(d):
    if not isinstance(d, dict):
        return []
    v = d.get("hasExactSynonym")
    if v is None:
        return []
    return list(dict.fromkeys(v))[0:8]


def build_disease_text(raw_dir, tables):
    """Raw OT tables -> DataFrame[diseaseId, disease_text_embed]."""
    disease_df = pd.read_parquet(f"{raw_dir}/{tables['diseases']}").filter(
        ["id", "name", "description", "dbXRefs", "synonyms", "ancestors", "parents", "therapeuticAreas"],
        axis=1,
    ).dropna(axis=1, how="all")

    disease_df["ExactSynonyms"] = disease_df["synonyms"].apply(_exact_synonyms)
    disease_df["ExactSynonyms"] = disease_df["ExactSynonyms"].apply(" ".join).str.strip()

    for c in disease_df.select_dtypes("O").columns:
        disease_df[c] = disease_df[c].apply(
            lambda x: " ".join(x.split()[:2_500]) if isinstance(x, str) else x[:1_000] if isinstance(x, list) else x
        )

    disease_df["disease_text_embed"] = disease_df[
        ["name", "ExactSynonyms", "description", "dbXRefs", "therapeuticAreas", "parents"]
    ].astype(str).apply(" ".join, axis=1)

    df_pheno = (
        pd.read_parquet(f"{raw_dir}/{tables['diseaseToPhenotype']}", columns=["disease", "phenotype"])
        .rename(columns={"disease": "diseaseId"})
        .drop_duplicates()
    )
    counts = df_pheno["phenotype"].value_counts()
    rare = counts[counts <= 3].index
    df_pheno["phenotype"] = df_pheno["phenotype"].replace(rare, "Rare").fillna("Unknown_Phenotypes")
    df_pheno = df_pheno.groupby("diseaseId").agg(
        phenotypes=("phenotype", lambda x: " ".join(x))
    ).reset_index()

    disease_df = disease_df.merge(df_pheno, left_on="id", right_on="diseaseId", how="left")
    disease_df["phenotypes"] = disease_df["phenotypes"].fillna("Unknown_Phenotypes")
    disease_df["disease_text_embed"] = disease_df["disease_text_embed"] + " " + disease_df["phenotypes"]
    disease_df.drop(columns=["diseaseId", "phenotypes"], inplace=True, errors="ignore")
    disease_df.rename(columns={"id": "diseaseId"}, inplace=True)

    return disease_df[["diseaseId", "disease_text_embed"]].copy()


def build_target_text(raw_dir, tables):
    """Raw OT tables -> DataFrame[targetId, target_text_embed]."""
    target_df = pd.read_parquet(
        f"{raw_dir}/{tables['targets']}",
        columns=[
            "id", "approvedSymbol", "biotype", "genomicLocation", "alternativeGenes", "approvedName",
            "go", "hallmarks", "synonyms", "functionDescriptions", "subcellularLocations", "targetClass",
            "constraint", "tep", "proteinIds", "tractability", "safetyLiabilities", "pathways",
        ],
    ).dropna(how="all", axis=1)

    target_df["go"] = target_df["go"].map(
        lambda x: list(dict.fromkeys(d["id"] for d in x if isinstance(d, dict) and "id" in d))
        if isinstance(x, (list, tuple, np.ndarray)) else []
    )
    target_df["proteinIds"] = target_df["proteinIds"].map(
        lambda x: list(dict.fromkeys(d["id"] for d in x if isinstance(d, dict) and "id" in d))
        if isinstance(x, (list, tuple, np.ndarray)) else []
    )
    target_df["synonyms"] = target_df["synonyms"].str.lower().map(
        lambda x: list(dict.fromkeys(d["label"] for d in x if isinstance(d, dict) and "label" in d))
        if isinstance(x, (list, tuple, np.ndarray)) else []
    )
    target_df["synonyms"] = target_df["synonyms"].str.slice(0, 10)

    target_df["subcellularLocations"] = target_df["subcellularLocations"].map(
        lambda x: list(dict.fromkeys(d["location"] for d in x if isinstance(d, dict) and "location" in d))
        if isinstance(x, (list, tuple, np.ndarray)) else []
    )

    target_df["tractability"] = target_df["tractability"].apply(
        lambda v: list({
            frozenset(d.items()): d
            for d in (v if isinstance(v, (list, tuple, np.ndarray)) else [v])
            if isinstance(d, dict) and d.get("value")
        }.values())
    )
    target_df["tractability"] = target_df["tractability"].map(
        lambda x: list((" ".join([d["modality"], d["id"]]) for d in x if isinstance(d, dict) and "id" in d))
        if isinstance(x, (list, tuple, np.ndarray)) else []
    )

    for c in ["go", "pathways", "proteinIds", "targetClass", "safetyLiabilities", "tractability",
              "alternativeGenes", "hallmarks", "functionDescriptions", "tep"]:
        if c in target_df.columns:
            target_df[c] = target_df[c].apply(
                lambda x: " ".join(x.split()[:2_000]) if isinstance(x, str) else x[:250] if isinstance(x, list) else x
            )

    target_df["functionDescriptions"] = target_df["functionDescriptions"].map(
        lambda x: " ".join(x), na_action="ignore"
    ).fillna("").str.strip()

    target_df["synonyms"] = target_df["synonyms"].apply(" ".join).str.strip()
    target_df["sym"] = target_df["approvedSymbol"].str.replace(r"\d+", "", regex=True)

    # `constraint` is appended even though the shipped notebook line comments it out:
    # the committed copy_proc artefact the model actually consumed contains it.
    # test_build_2202_features.py pins this against the artefact.
    target_df["target_text_embed"] = target_df[
        ["sym", "approvedName", "synonyms", "functionDescriptions", "go", "tractability", "pathways",
         "targetClass", "constraint"]
    ].astype(str).apply(" ".join, axis=1)

    target_df.rename(columns={"id": "targetId"}, inplace=True)
    return target_df[["targetId", "target_text_embed"]].copy()


if __name__ == "__main__":
    disease_out = build_disease_text(RAW_2202, TABLES_2202)
    disease_out.to_parquet(f"{OUT}/disease_df_2202.parquet")
    print("disease_df_2202:", disease_out.shape)

    target_out = build_target_text(RAW_2202, TABLES_2202)
    n_clinical = target_out["target_text_embed"].str.contains(
        TRACTABILITY_CLINICAL_RE, case=False, regex=True
    ).sum()
    print(f"22.02-native target_text carrying clinical-precedence tokens: {n_clinical} "
          f"(legitimate -- reflects pre-2022 approval status, available at train time)")
    target_out.to_parquet(f"{OUT}/target_df_2202.parquet")
    print("target_df_2202:", target_out.shape)

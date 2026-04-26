from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data" / "proc"
PROJECT_ROOT = APP_ROOT.parent

COMPARISON_LOOKUP_PATH = DATA_DIR / "comparison_lookup.parquet"
DISEASE_METADATA_PATH = DATA_DIR / "disease_metadata.csv"

FALLBACK_DL_PATH = PROJECT_ROOT / "Outputs" / "CV_DL" / "oof_dl_preds.parquet"
FALLBACK_TREE_PATH = PROJECT_ROOT / "Outputs" / "CV_tree" / "CB_5_cv.parquet"
FALLBACK_CANDIDATES_PATH = PROJECT_ROOT / "Outputs" / "S2-DL_novel+known_candidates.csv"

COMPARISON_COLUMNS = [
    "diseaseId",
    "targetId",
    "ot_score",
    "known_label",
    "otrec_oof_pred",
    "ottree_pred",
]

DISEASE_META_COLUMNS = [
    "diseaseId",
    "diseaseName",
    "orphan",
    "known_clinical_targets",
    "comparison_row_count",
    "available_ot_score_count",
    "available_ottree_count",
]


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _fallback_comparison_available() -> bool:
    return FALLBACK_DL_PATH.exists() and FALLBACK_TREE_PATH.exists()


def _validate_comparison_frame(
    comparison_df: pd.DataFrame, source: Path
) -> pd.DataFrame:
    missing = [
        column for column in COMPARISON_COLUMNS if column not in comparison_df.columns
    ]
    if missing:
        raise ValueError(
            f"comparison lookup at {source} is missing columns: "
            f"{', '.join(sorted(missing))}"
        )
    return comparison_df[COMPARISON_COLUMNS].copy()


@lru_cache(maxsize=1)
def load_comparison_lookup() -> pd.DataFrame:
    if COMPARISON_LOOKUP_PATH.exists():
        comparison_df = pd.read_parquet(COMPARISON_LOOKUP_PATH)
        comparison_df = _validate_comparison_frame(
            comparison_df, COMPARISON_LOOKUP_PATH
        )
        if not comparison_df.empty or not _fallback_comparison_available():
            return comparison_df

    if not _fallback_comparison_available():
        return _empty_frame(COMPARISON_COLUMNS)

    otrec = pd.read_parquet(
        FALLBACK_DL_PATH,
        columns=["diseaseId", "targetId", "score", "label", "pred"],
    ).rename(
        columns={
            "score": "ot_score",
            "label": "known_label",
            "pred": "otrec_oof_pred",
        }
    )
    ottree = pd.read_parquet(
        FALLBACK_TREE_PATH,
        columns=["diseaseId", "targetId", "pred"],
    ).rename(columns={"pred": "ottree_pred"})

    comparison_df = otrec.merge(ottree, on=["diseaseId", "targetId"], how="left")
    return _validate_comparison_frame(comparison_df, FALLBACK_DL_PATH)


@lru_cache(maxsize=1)
def load_disease_metadata() -> pd.DataFrame:
    base_metadata = _empty_frame(DISEASE_META_COLUMNS)

    if DISEASE_METADATA_PATH.exists():
        base_metadata = pd.read_csv(DISEASE_METADATA_PATH)
    elif FALLBACK_CANDIDATES_PATH.exists():
        candidate_df = pd.read_csv(
            FALLBACK_CANDIDATES_PATH,
            usecols=[
                "diseaseId",
                "diseaseName",
                "disease_num_known_clinical_targets",
                "orphan",
            ],
        ).drop_duplicates()
        candidate_df = candidate_df.rename(
            columns={
                "disease_num_known_clinical_targets": "known_clinical_targets",
            }
        )
        base_metadata = candidate_df

    comparison_df = load_comparison_lookup()
    if comparison_df.empty:
        if base_metadata.empty:
            return _empty_frame(DISEASE_META_COLUMNS)
        for column in DISEASE_META_COLUMNS:
            if column not in base_metadata.columns:
                base_metadata[column] = pd.NA
        return base_metadata[DISEASE_META_COLUMNS].copy()

    derived_metadata = (
        comparison_df.groupby("diseaseId", as_index=False)
        .agg(
            known_clinical_targets=("known_label", "sum"),
            comparison_row_count=("targetId", "size"),
            available_ot_score_count=(
                "ot_score",
                lambda series: int(series.notna().sum()),
            ),
            available_ottree_count=(
                "ottree_pred",
                lambda series: int(series.notna().sum()),
            ),
        )
        .copy()
    )

    if base_metadata.empty:
        base_metadata = derived_metadata
    else:
        base_metadata = base_metadata.merge(
            derived_metadata, on="diseaseId", how="outer"
        )
        if "known_clinical_targets_x" in base_metadata.columns:
            base_metadata["known_clinical_targets"] = base_metadata[
                "known_clinical_targets_x"
            ].fillna(base_metadata.get("known_clinical_targets_y"))
            base_metadata = base_metadata.drop(
                columns=[
                    column
                    for column in [
                        "known_clinical_targets_x",
                        "known_clinical_targets_y",
                    ]
                    if column in base_metadata.columns
                ]
            )

    for column in DISEASE_META_COLUMNS:
        if column not in base_metadata.columns:
            base_metadata[column] = pd.NA

    return base_metadata[DISEASE_META_COLUMNS].copy()


def build_result_annotations(disease_id: str, target_ids: pd.Series) -> pd.DataFrame:
    comparison_df = load_comparison_lookup()
    if comparison_df.empty:
        return _empty_frame(COMPARISON_COLUMNS)

    target_ids = pd.Series(target_ids).astype(str)
    annotations = comparison_df[
        (comparison_df["diseaseId"] == disease_id)
        & (comparison_df["targetId"].isin(target_ids.tolist()))
    ].copy()
    return annotations


def get_disease_metadata_row(disease_id: str) -> dict[str, object]:
    disease_metadata = load_disease_metadata()
    if disease_metadata.empty:
        return {}

    matches = disease_metadata[disease_metadata["diseaseId"] == disease_id]
    if matches.empty:
        return {}

    return matches.iloc[0].to_dict()

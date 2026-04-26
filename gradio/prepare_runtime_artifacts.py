from __future__ import annotations

from pathlib import Path

import pandas as pd


APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data" / "proc"
PROJECT_ROOT = APP_ROOT.parent

CV_DL_PATH = PROJECT_ROOT / "Outputs" / "CV_DL" / "oof_dl_preds.parquet"
CV_TREE_PATH = PROJECT_ROOT / "Outputs" / "CV_tree" / "CB_5_cv.parquet"
CANDIDATES_PATH = PROJECT_ROOT / "Outputs" / "S2-DL_novel+known_candidates.csv"

COMPARISON_LOOKUP_PATH = DATA_DIR / "comparison_lookup.parquet"
DISEASE_METADATA_PATH = DATA_DIR / "disease_metadata.csv"


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required {label} input is missing: {path}. "
            "Run the OTRec evaluation notebooks first, or copy the packaged "
            "runtime artifacts into gradio/data/proc/."
        )


def build_comparison_lookup() -> pd.DataFrame:
    _require_file(CV_DL_PATH, "OTRec cross-validation")
    _require_file(CV_TREE_PATH, "OTTree cross-validation")

    otrec = pd.read_parquet(
        CV_DL_PATH,
        columns=["diseaseId", "targetId", "score", "label", "pred"],
    ).rename(
        columns={
            "score": "ot_score",
            "label": "known_label",
            "pred": "otrec_oof_pred",
        }
    )
    ottree = pd.read_parquet(
        CV_TREE_PATH,
        columns=["diseaseId", "targetId", "pred"],
    ).rename(columns={"pred": "ottree_pred"})
    return otrec.merge(ottree, on=["diseaseId", "targetId"], how="left")


def build_disease_metadata(comparison_df: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame(columns=["diseaseId", "diseaseName", "orphan"])
    if CANDIDATES_PATH.exists():
        base = pd.read_csv(
            CANDIDATES_PATH,
            usecols=[
                "diseaseId",
                "diseaseName",
                "disease_num_known_clinical_targets",
                "orphan",
            ],
        ).rename(
            columns={"disease_num_known_clinical_targets": "known_clinical_targets"}
        )
        base = base.drop_duplicates(subset=["diseaseId"])

    derived = comparison_df.groupby("diseaseId", as_index=False).agg(
        known_clinical_targets=("known_label", "sum"),
        comparison_row_count=("targetId", "size"),
        available_ot_score_count=("ot_score", lambda series: int(series.notna().sum())),
        available_ottree_count=(
            "ottree_pred",
            lambda series: int(series.notna().sum()),
        ),
    )

    if base.empty:
        return derived

    merged = base.merge(derived, on="diseaseId", how="outer", suffixes=("_base", ""))
    if "known_clinical_targets_base" in merged.columns:
        merged["known_clinical_targets"] = merged["known_clinical_targets_base"].fillna(
            merged["known_clinical_targets"]
        )
        merged = merged.drop(columns=["known_clinical_targets_base"])
    return merged


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    comparison_df = build_comparison_lookup()
    comparison_df.to_parquet(COMPARISON_LOOKUP_PATH, index=False)

    disease_metadata_df = build_disease_metadata(comparison_df)
    disease_metadata_df.to_csv(DISEASE_METADATA_PATH, index=False, encoding="utf-8")

    print(f"Wrote {COMPARISON_LOOKUP_PATH} with {len(comparison_df):,} rows")
    print(f"Wrote {DISEASE_METADATA_PATH} with {len(disease_metadata_df):,} rows")


if __name__ == "__main__":
    main()

"""Packaging column contract -- the columns OTRec/gradio/app.py actually reads.

Added after a live failure: the repackaged disease_df carried only the model's
columns, and the Space's disease SEARCH crashed with KeyError: 'synonyms'
(app.py:979) while every scoring audit passed. The contract below is the
enumerated set of frame columns referenced anywhere in app.py; the test also
executes the app's own search expressions against the packaged frames.

Run: python3 retrain_2512/test_app_column_contract.py
"""
from pathlib import Path

import pandas as pd

PKG = Path("/mnt/d/Research/OpenTargetsTransfer/retrain_2512/gradio_artifacts")

DISEASE_CONTRACT = {"diseaseId", "disease_text", "name", "description", "synonyms", "ExactSynonyms"}
TARGET_CONTRACT = {"targetId", "target_text", "approvedSymbol", "approvedName",
                   "functionDescriptions", "sym", "tractability"}


def test_disease_df_contract_and_search():
    d = pd.read_parquet(PKG / "disease_df.parquet")
    missing = DISEASE_CONTRACT - set(d.columns)
    assert not missing, f"packaged disease_df missing app columns: {missing}"
    # Execute the exact search expression from app.py search_diseases.
    query = "CDKL5"
    mask = (
        d["diseaseId"].astype(str).str.contains(query, case=False, na=False)
        | d["name"].astype(str).str.contains(query, case=False, na=False)
        | d["synonyms"].astype(str).str.contains(query, case=False, na=False)
    )
    assert mask.any(), "search expression found no CDKL5 rows"
    print(f"disease_df contract OK ({len(d):,} rows); CDKL5 search matches {int(mask.sum())} rows")


def test_target_df_contract_and_search():
    t = pd.read_parquet(PKG / "target_df.parquet")
    missing = TARGET_CONTRACT - set(t.columns)
    assert not missing, f"packaged target_df missing app columns: {missing}"
    query = "TNF"
    mask = (
        t["targetId"].astype(str).str.contains(query, case=False, na=False)
        | t["approvedSymbol"].astype(str).str.contains(query, case=False, na=False)
    )
    assert mask.any(), "search expression found no TNF rows"
    print(f"target_df contract OK ({len(t):,} rows); TNF search matches {int(mask.sum())} rows")


if __name__ == "__main__":
    test_disease_df_contract_and_search()
    test_target_df_contract_and_search()
    print("PASS: packaged frames satisfy the app column contract")

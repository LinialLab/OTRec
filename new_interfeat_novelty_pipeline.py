"""
Simple helper functions for running novelty filtering and PubMed co‑occurrence analysis on
prediction data frames.
Requires the InterFeat codebase: InterFeat: A Pipeline for Finding Interesting Scientific Features
https://arxiv.org/abs/2505.13534
https://github.com/LinialLab/InterFeat

These functions are intentionally lightweight and avoid importing heavy dependencies such as
``scispacy`` or any knowledge–graph components.  They provide a way to construct candidate
query terms directly from a pandas DataFrame and then invoke the existing PubMed search
pipeline provided in ``search_pubmed.py``.  The goal is to support quick experimentation
with new data formats without modifying the legacy functions that operate on disk files.
"""

from __future__ import annotations

from typing import Optional, Dict

import pandas as pd

from search_pubmed import cache_search
from tqdm import tqdm
import pandas as pd

def collect_pairwise_search_data(df_pairs, min_results_count=20):
    """Run PubMed search per (query, target) pair instead of full cross-product."""
    results = []
    progress_bar = tqdm(total=len(df_pairs), desc="Collecting Data (pairwise)")
    for _, row in df_pairs.iterrows():
        query, target = row["feature_name"], row["disease_name"]
        q_count = cache_search(query)
        t_count = cache_search(target)
        if q_count > min_results_count and t_count > min_results_count:
            combined = f"({query}) AND ({target})"
            qt_count = cache_search(combined)
            results.append({
                "Query": query,
                "Target": target,
                "Query Count": q_count,
                "Target Count": t_count,
                "Co-occurrence Count": qt_count,
            })
        progress_bar.update(1)
    progress_bar.close()
    return pd.DataFrame(results)


def prepare_candidate_df_from_predictions(
    df_preds: pd.DataFrame,
    feature_col: str = "approvedSymbol",
    pred_col: Optional[str] = None,
    threshold: Optional[float] = None,
    add_kg_columns: bool = True,
) -> pd.DataFrame:
    """Construct a candidate query DataFrame from a table of model predictions.

    Each unique value in ``feature_col`` becomes a candidate query.  Optionally filters rows
    based on a prediction column and threshold.

    Parameters
    ----------
    df_preds : pandas.DataFrame
        Original prediction data containing at least the column specified in ``feature_col``.
    feature_col : str, default ``"approvedSymbol"``
        Column name in ``df_preds`` representing the feature or gene to query.
    pred_col : str, optional
        Column name containing prediction scores.  If provided together with ``threshold``,
        rows will be filtered to those where ``df_preds[pred_col] >= threshold`` before
        extracting unique features.
    threshold : float, optional
        Minimum value in ``pred_col`` required for a row to be considered.  Ignored when
        ``pred_col`` is None.
    add_kg_columns : bool, default True
        When True, adds placeholder columns for ``KG_Hits`` and ``feature_level_min_kg_hits``
        with zeros.  These columns are expected by downstream filtering functions.  If you are
        computing real knowledge–graph connections separately you can disable this and merge
        your own results.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``feature_name``, ``cui_nomenclature`` and optionally
        knowledge–graph placeholder columns.  Ready to be passed directly to
        :func:`search_pubmed.run_search_pubmed`.

    Notes
    -----
    No attempt is made to resolve biomedical entities to UMLS CUIs; the ``feature_name``
    text is simply copied into ``cui_nomenclature``, and the ``cui`` column is left empty.
    """

    if feature_col not in df_preds.columns:
        raise ValueError(f"Column '{feature_col}' not found in prediction DataFrame.")

    # Apply filtering based on prediction score if specified
    if pred_col is not None and threshold is not None and pred_col in df_preds.columns:
        df_filtered = df_preds.loc[df_preds[pred_col] >= threshold].copy()
    else:
        df_filtered = df_preds.copy()

    # Extract unique candidate terms and drop blanks
    unique_values = (
        df_filtered[feature_col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )
    unique_terms = pd.DataFrame({"feature_name": unique_values})
    unique_terms = unique_terms.loc[unique_terms["feature_name"] != ""].reset_index(drop=True)

    # Duplicate the feature name into the query and add a blank CUI column
    unique_terms["cui_nomenclature"] = unique_terms["feature_name"]
    unique_terms["cui"] = ""

    if add_kg_columns:
        unique_terms["KG_Hits"] = 0
        unique_terms["feature_level_min_kg_hits"] = 0
        unique_terms['sim_score'] = 0.5
        unique_terms['feature_level_sum_kg_hits']=0

    return unique_terms


def run_pubmed_pipeline_on_predictions(
    df_preds: pd.DataFrame,
    feature_col: str = "approvedSymbol",
    disease_col: str = "disease_name",
    pred_col: Optional[str] = None,
    threshold: Optional[float] = None,
    config_overrides: Optional[Dict[str, str]] = None,
    save_outputs: bool = True,
):
    """Run the PubMed search pipeline on a prediction DataFrame.

    This function is a convenience wrapper around :func:`search_pubmed.run_search_pubmed`.  It
    first calls :func:`prepare_candidate_df_from_predictions` to build a minimal candidate
    query DataFrame and then constructs a configuration dictionary.  The search is executed
    using the list of unique diseases from ``disease_col`` as the targets.

    Parameters
    ----------
    df_preds : pandas.DataFrame
        A DataFrame containing your prediction results with at least columns for the feature
        names (e.g. genes) and the associated disease/target labels.  See ``feature_col`` and
        ``disease_col`` for details.
    feature_col : str, default ``"approvedSymbol"``
        Column in ``df_preds`` containing the gene or feature names.
    disease_col : str, default ``"disease_name"``
        Column in ``df_preds`` containing the disease or target names.  Unique values from this
        column will be used as the search targets.
    pred_col : str, optional
        Name of a numeric prediction column.  If provided along with ``threshold``, only rows
        with values greater than or equal to ``threshold`` will be considered when building the
        candidate queries.  If omitted, all rows are used.
    threshold : float, optional
        Threshold applied to ``pred_col`` to select high‑confidence predictions.  Ignored when
        ``pred_col`` is None.
    config_overrides : dict, optional
        Dictionary of configuration values to override defaults for the underlying
        ``run_search_pubmed`` call.  Useful keys include ``"ENTREZ_EMAIL"``,
        ``"ENTREZ_API_KEY"``, ``"DO_MINI_COMBINED_PATH_FILT"``, and file name prefixes.
    save_outputs : bool, default True
        Passed through to :func:`search_pubmed.run_search_pubmed` to control whether result
        CSV files are persisted.  Setting this to ``False`` will suppress file writes.

    Returns
    -------
    pandas.DataFrame
        The full result DataFrame returned by :func:`search_pubmed.run_search_pubmed`.

    Notes
    -----
    The underlying search can be slow for large numbers of features and targets because it
    performs a PubMed search for every query–target pair.  Consider sampling a subset of
    features or setting a higher prediction threshold when experimenting.
    """

    # Build candidate query DataFrame from predictions
    candidate_df = prepare_candidate_df_from_predictions(
        df_preds=df_preds,
        feature_col=feature_col,
        pred_col=pred_col,
        threshold=threshold,
        add_kg_columns=True,
    )

    # Determine the list of targets from the disease column
    if disease_col not in df_preds.columns:
        raise ValueError(f"Column '{disease_col}' not found in prediction DataFrame.")
    targets = (
        df_preds[disease_col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    # Construct default configuration for the search
    cfg = {
        "targets": targets,
        "QUERY_CANDIDATES_FILE": "",  # not used when passing df argument
        "DO_MINI_COMBINED_PATH_FILT": False,
        "full_results_filename": "candidates_search_results_new.csv",
        "filtered_results_filename": "review_interesting_candidates_results_new.csv",
    }
    if config_overrides:
        cfg.update(config_overrides)

    try:
        from search_pubmed import run_search_pubmed
    except ImportError as e:
        raise ImportError(
            "Unable to import run_search_pubmed. Ensure search_pubmed.py is available "
            "and on the Python path."
        ) from e

    # Execute the search pipeline.  Passing candidate_df triggers the df branch in run_search_pubmed.
    res_df = run_search_pubmed( cfg, df=candidate_df, SAVE_OUTPUTS=save_outputs,get_promising_results=False,concat_targets =False )
    return res_df



import re
import pandas as pd
from search_pubmed import run_search_pubmed
from new_interfeat_novelty_pipeline import prepare_candidate_df_from_predictions  # if same file, just call directly

def _sanitize_fn(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', str(s)).strip('_')

def run_pubmed_per_disease_then_concat(
    df_preds: pd.DataFrame,
    feature_col: str = "approvedSymbol",
    disease_col: str = "disease_name",
    pred_col: str | None = None,
    threshold: float | None = None,
    config_overrides: dict | None = None,
    save_outputs: bool = False,
) -> pd.DataFrame:
    """
    Run PubMed search per disease, taking all features associated with that disease,
    then concat results across diseases.
    """
    # unique diseases
    diseases = (
        df_preds[disease_col]
        .dropna().astype(str).str.strip().unique().tolist()
    )

    out = []
    for d in diseases:
        sub = df_preds.loc[df_preds[disease_col].astype(str).str.strip() == d].copy()
        # build per-disease candidates
        cand_df = prepare_candidate_df_from_predictions(
            df_preds=sub,
            feature_col=feature_col,
            pred_col=pred_col,
            threshold=threshold,
            add_kg_columns=True,
        )

        # config: single target, disable concatenation
        cfg = {
            "targets": [d],
            "QUERY_CANDIDATES_FILE": "",
            "DO_MINI_COMBINED_PATH_FILT": False,
            "CONCAT_TARGET_TERMS": False,
            "full_results_filename": f"candidates_search_results_{_sanitize_fn(d)}.csv",
            "filtered_results_filename": f"review_interesting_candidates_results_{_sanitize_fn(d)}.csv",
        }
        if config_overrides:
            cfg.update(config_overrides)

        res = run_search_pubmed(
            cfg,
            df=cand_df,
            SAVE_OUTPUTS=save_outputs,
            get_promising_results=False,
            concat_targets=False,   # force no target OR-merge
        )
        # enforce explicit target
        res["Target"] = d
        out.append(res)

    if not out:
        return pd.DataFrame()

    out = [c for c in out if c.shape[0]>0]
    combined = pd.concat(out, ignore_index=True)

    # light de-dup for safety
    combined.drop_duplicates(
        subset=["Query", "feature_name", "Target", "Co-occurrence Count"],
        inplace=True
    )
    return combined

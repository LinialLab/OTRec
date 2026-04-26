import html
import os
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote, urlparse

import gradio as gr
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from huggingface_hub import hf_hub_download
from tensorflow import keras

from dl_model_def import build_two_tower_model
from runtime_data import build_result_annotations

# ============================================
#  CONFIG
# ============================================

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data" / "proc"
MODEL_REPO_ID = os.environ.get("OTREC_MODEL_REPO_ID", "GrimSqueaker/OTRec")
MODEL_FILENAME = os.environ.get("OTREC_MODEL_FILENAME", "model.weights.h5")
MODEL_DOWNLOAD_ETAG_TIMEOUT = int(os.environ.get("OTREC_HF_ETAG_TIMEOUT", "30"))

FILTER_HIDE_PACKAGED = "Hide packaged known hits"
FILTER_ALL = "All targets"
FILTER_ONLY_PACKAGED = "Only packaged known hits"

DISPLAY_COLUMNS = [
    "Rank",
    "Gene",
    "Gene Name",
    "OTRec score",
    "Open Targets score",
    "OTTree score",
    "Tractability",
    "Packaged label",
    "Function",
    "Open Targets link",
]

DISEASE_DISPLAY_COLUMNS = [
    "Rank",
    "Disease",
    "Disease ID",
    "OTRec score",
    "Description",
    "Open Targets link",
]

SORT_OPTIONS = {
    "OTRec score": ("otrec_score", True),
    "Open Targets association score": ("ot_score", True),
    "OTTree score": ("ottree_pred", True),
    "Gene symbol": ("approvedSymbol", False),
}

RUNTIME_LOCK = threading.RLock()

# ============================================
#  LOAD TRAINING DATA
# ============================================

df_learn = pd.read_parquet(DATA_DIR / "df_learn_sub.parquet")
disease_df = pd.read_parquet(DATA_DIR / "disease_df.parquet")
target_df = pd.read_parquet(DATA_DIR / "target_df.parquet")

# Ensure column names match training
df_learn = df_learn.rename(
    columns={"disease_text_embed": "disease_text", "target_text_embed": "target_text"},
    errors="ignore",
)

disease_df.rename(
    columns={"disease_text_embed": "disease_text"}, errors="ignore", inplace=True
)

disease_df = disease_df.copy()
disease_df["diseaseId"] = disease_df["diseaseId"].astype(str)
disease_df["name"] = disease_df["name"].fillna("").astype(str)

target_df.rename(
    columns={"target_text_embed": "target_text"}, errors="ignore", inplace=True
)

BATCH_SIZE = 1024


def _load_weights_file() -> str:
    last_error = None
    for attempt in range(2):
        try:
            print("Downloading model weights from Hugging Face Hub...")
            weights_file = hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=MODEL_FILENAME,
                etag_timeout=MODEL_DOWNLOAD_ETAG_TIMEOUT,
            )
            print(f"Weights downloaded to: {weights_file}")
            return weights_file
        except Exception as error:
            last_error = error
            if attempt == 0:
                print(f"Model download failed ({error}); retrying once...")
                continue
            raise RuntimeError(
                "Unable to download OTRec model weights from Hugging Face. "
                f"Check network access and that {MODEL_REPO_ID}/{MODEL_FILENAME} "
                "is reachable, or set OTREC_MODEL_REPO_ID / OTREC_MODEL_FILENAME. "
                f"Last error: {error}"
            ) from error
    raise RuntimeError(f"Unable to download OTRec model weights: {last_error}")


def _load_model_with_weights():
    weights_file = _load_weights_file()
    print("Building TwoTowerDual...")
    keras.backend.clear_session()
    loaded_model = build_two_tower_model(df_learn)

    print("Loading weights...")
    try:
        loaded_model.load_weights(weights_file)
    except ValueError as error:
        print(f"Standard load failed ({error}). Attempting name-mismatch fix...")
        with h5py.File(weights_file, "r") as handle:
            h5_keys = list(handle.keys())
            print(f"Weights file contains layers: {h5_keys}")

            def match_layer_name(target_attr, prefix):
                match = next((key for key in h5_keys if key.startswith(prefix)), None)
                if match and hasattr(loaded_model, target_attr):
                    layer = getattr(loaded_model, target_attr)
                    print(
                        f"Renaming model layer '{layer.name}' to '{match}' to match file."
                    )
                    layer._name = match

            match_layer_name("dise_emb", "dise_emb")
            match_layer_name("q_tower", "tower")

        loaded_model.load_weights(weights_file)

    print("Weights loaded successfully.")
    return loaded_model


def _precompute_candidate_embeddings(loaded_model):
    print("Precomputing candidate embeddings (batched)...")
    target_texts = target_df["target_text"].astype(str).to_numpy()
    target_ids = target_df["targetId"].astype(str).to_numpy()
    cand_embs_list = []

    total = len(target_texts)
    for index in range(0, total, BATCH_SIZE):
        end = min(index + BATCH_SIZE, total)
        batch_txt = target_texts[index:end]
        batch_ids = target_ids[index:end]
        emb_batch = loaded_model.encode_k(batch_txt, batch_ids)
        cand_embs_list.append(emb_batch)
        if index % 5000 == 0:
            print(f"  Processed {index}/{total} candidates...")

    cand_embs = tf.concat(cand_embs_list, axis=0)
    cand_embs = tf.nn.l2_normalize(cand_embs, axis=1).numpy()
    print(f"Candidate embeddings ready. Shape: {cand_embs.shape}")
    return cand_embs


def _precompute_disease_embeddings(loaded_model):
    print("Precomputing disease embeddings (batched)...")
    disease_texts = disease_df["disease_text"].astype(str).to_numpy()
    disease_ids = disease_df["diseaseId"].astype(str).to_numpy()
    embs_list = []
    total = len(disease_texts)
    for index in range(0, total, BATCH_SIZE):
        end = min(index + BATCH_SIZE, total)
        batch_txt = disease_texts[index:end]
        batch_ids = disease_ids[index:end]
        emb_batch = loaded_model.encode_q(batch_txt, batch_ids)
        embs_list.append(emb_batch)
        if index % 5000 == 0:
            print(f"  Processed {index}/{total} diseases...")
    dis_embs = tf.concat(embs_list, axis=0)
    dis_embs = tf.nn.l2_normalize(dis_embs, axis=1).numpy()
    print(f"Disease embeddings ready. Shape: {dis_embs.shape}")
    return dis_embs


@lru_cache(maxsize=1)
def _get_runtime_cached():
    loaded_model = _load_model_with_weights()
    cand_embs = _precompute_candidate_embeddings(loaded_model)
    return loaded_model, cand_embs


def get_runtime():
    with RUNTIME_LOCK:
        return _get_runtime_cached()


@lru_cache(maxsize=1)
def _get_disease_runtime_cached():
    loaded_model, _ = get_runtime()
    dis_embs = _precompute_disease_embeddings(loaded_model)
    return loaded_model, dis_embs


def get_disease_runtime():
    with RUNTIME_LOCK:
        return _get_disease_runtime_cached()


target_df = target_df.copy()
target_df["targetId"] = target_df["targetId"].astype(str)
target_df["approvedSymbol"] = (
    target_df["approvedSymbol"].fillna(target_df.get("sym", "")).astype(str)
)
target_df["approvedName"] = target_df["approvedName"].fillna("").astype(str)

FUNCTION_COLUMN = (
    "functionDescriptions"
    if "functionDescriptions" in target_df.columns
    else "functionDescription"
)


def _empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=DISPLAY_COLUMNS)


def _is_safe_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _render_results_table(results: pd.DataFrame, columns: list[str]) -> str:
    header_cells = "".join(
        f'<th scope="col" style="text-align:left;padding:0.6rem;border-bottom:1px solid #d9d9d9;">{html.escape(column)}</th>'
        for column in columns
    )

    if results.empty:
        body_rows = (
            f'<tr><td colspan="{len(columns)}" style="padding:0.9rem;color:#666;">'
            "No rows to display for the current selection."
            "</td></tr>"
        )
    else:
        rendered_rows: list[str] = []
        display_rows = results.reindex(columns=columns, fill_value="")
        for _, row in display_rows.iterrows():
            cells: list[str] = []
            for column in columns:
                value = row.get(column, "")
                if pd.isna(value):
                    text = ""
                else:
                    text = " ".join(str(value).split())
                if column == "Open Targets link" and text and _is_safe_url(text):
                    cell_value = (
                        f'<a href="{html.escape(text, quote=True)}" target="_blank" '
                        'rel="noopener noreferrer">Open Targets</a>'
                    )
                else:
                    cell_value = html.escape(text) if text else "&mdash;"
                cells.append(
                    f'<td style="padding:0.55rem;border-bottom:1px solid #ececec;vertical-align:top;word-break:break-word;">{cell_value}</td>'
                )
            rendered_rows.append(f"<tr>{''.join(cells)}</tr>")
        body_rows = "".join(rendered_rows)

    return (
        '<div style="overflow-x:auto;">'
        '<table role="table" aria-label="OTRec ranked results" style="width:100%;border-collapse:collapse;font-size:0.95rem;table-layout:auto;">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{body_rows}</tbody>"
        "</table>"
        "</div>"
    )


def _truncate_text(value: object, limit: int = 220) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}..."


def _known_status(value: object) -> str:
    if pd.isna(value):
        return "unlabeled"
    try:
        numeric_value = float(str(value))
    except (TypeError, ValueError):
        return "unlabeled"
    return "packaged known hit" if int(numeric_value) == 1 else "packaged novel"


def _format_bool(value: object) -> str:
    if pd.isna(value):
        return "—"
    return "Yes" if bool(value) else "No"


def _format_tractability(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and np.isnan(value):
        return "—"
    try:
        items = list(value)
    except TypeError:
        items = [value]
    items = [str(x) for x in items if x is not None and str(x).strip()]
    if not items:
        return "—"
    if len(items) <= 2:
        return "; ".join(items)
    return f"{'; '.join(items[:2])} (+{len(items) - 2})"


def _safe_error_message(error: Exception, limit: int = 200) -> str:
    message = " ".join(str(error).split())
    if len(message) <= limit:
        return message
    return f"{message[: limit - 3].rstrip()}..."


def _write_csv(export_df: pd.DataFrame, prefix: str) -> str:
    file_descriptor, csv_path = tempfile.mkstemp(prefix=prefix, suffix=".csv")
    os.close(file_descriptor)
    export_df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def _export_target_results(
    results: pd.DataFrame,
    disease_row: pd.Series,
) -> str | None:
    if results.empty:
        return None

    export_df = pd.DataFrame(
        {
            "diseaseId": disease_row["diseaseId"],
            "diseaseName": disease_row["name"],
            "rank": results["rank"].astype(int),
            "targetId": results["targetId"],
            "approvedSymbol": results["approvedSymbol"],
            "approvedName": results["approvedName"],
            "otrec_score": results["otrec_score"],
            "rank_percentile": results["percentile"],
            "open_targets_score": results.get("ot_score"),
            "ottree_score": results.get("ottree_pred"),
            "packaged_label": results.get("known_label"),
            "packaged_label_status": results.get("known_label").map(_known_status),
            "tractability": results.get(
                "tractability", pd.Series(index=results.index)
            ).map(_format_tractability),
            "function": results[FUNCTION_COLUMN].map(
                lambda value: _truncate_text(value, 1000)
            ),
            "openTargetsUrl": results["targetId"].map(
                lambda target_id: f"https://platform.opentargets.org/target/{target_id}"
            ),
        }
    )
    return _write_csv(export_df, "otrec_targets_")


def _export_disease_results(
    results: pd.DataFrame,
    target_dict: dict[str, object],
) -> str | None:
    if results.empty:
        return None

    target_id = str(target_dict.get("targetId", ""))
    export_df = pd.DataFrame(
        {
            "targetId": target_id,
            "targetSymbol": target_dict.get("approvedSymbol", ""),
            "targetName": target_dict.get("approvedName", ""),
            "rank": results["rank"].astype(int),
            "diseaseId": results["diseaseId"],
            "diseaseName": results["name"],
            "otrec_score": results["otrec_score"],
            "description": results["description"].map(
                lambda value: _truncate_text(value, 1000)
            ),
            "openTargetsUrl": results["diseaseId"].map(
                lambda disease_id: (
                    f"https://platform.opentargets.org/disease/{disease_id}"
                )
            ),
        }
    )
    return _write_csv(export_df, "otrec_diseases_")


def _disease_ols_url(disease_id: str) -> str | None:
    ontology_map = {
        "EFO_": ("efo", "http://www.ebi.ac.uk/efo/"),
        "MONDO_": ("mondo", "http://purl.obolibrary.org/obo/"),
        "DOID_": ("doid", "http://purl.obolibrary.org/obo/"),
    }
    for prefix, (ontology, iri_base) in ontology_map.items():
        if disease_id.startswith(prefix):
            iri = quote(f"{iri_base}{disease_id}", safe="")
            return f"https://www.ebi.ac.uk/ols4/ontologies/{ontology}/classes?iri={iri}"
    return None


def _build_summary(
    disease_row: pd.Series,
    shown_count: int,
    candidate_count: int,
    total_ranked_count: int | None = None,
) -> str:
    description = _truncate_text(disease_row.get("description", ""), limit=320)
    disease_id = disease_row["diseaseId"]
    ot_url = f"https://platform.opentargets.org/disease/{disease_id}"
    ols_url = _disease_ols_url(disease_id)
    links = [f"[Open Targets]({ot_url})"]
    if ols_url:
        links.append(f"[OLS]({ols_url})")
    lines = [
        f"## {disease_row['name']}  \u00b7  `{disease_id}`",
        "  \u00b7  ".join(links),
    ]
    if description:
        lines.append("")
        lines.append(description)
    lines.append("")
    if total_ranked_count is not None and total_ranked_count != candidate_count:
        lines.append(
            "Showing "
            f"{shown_count} of {candidate_count} rows matching the current filters "
            f"({total_ranked_count} ranked targets total)."
        )
    else:
        lines.append(f"Showing {shown_count} of {candidate_count} candidate targets.")
    return "\n".join(lines)


def _build_note(
    fallback_sort: bool,
    sort_label: str,
    min_score: float,
    comparison_row_count: int,
    available_ot_score_count: int,
    available_ottree_count: int,
    candidate_count: int,
    total_ranked_count: int,
) -> str:
    parts = [
        "Packaged label semantics: `packaged known hit` means label `1` in the packaged comparison data; `packaged novel` means packaged label `0`; unlabeled rows remain rankable by OTRec.",
    ]
    if comparison_row_count > 0:
        parts.append(
            "Packaged comparison coverage for this disease: "
            f"{comparison_row_count} pairs, "
            f"{available_ot_score_count} Open Targets scores, and "
            f"{available_ottree_count} OTTree scores."
        )
    else:
        parts.append(
            "Packaged comparison coverage is unavailable for this disease, so packaged-label filters and external-score sorts may be incomplete."
        )
    if candidate_count == 0 and total_ranked_count > 0:
        parts.append("No rows matched the current filters.")
    if fallback_sort:
        parts.append(
            f"`{sort_label}` was not available for this disease; results are ordered by OTRec score."
        )
    if min_score > 0:
        parts.append(f"Filtered to OTRec score ≥ {min_score:.2f}.")
    parts.append(
        "Downloads export the full filtered ranking and the full unfiltered ranking, not only the visible top-K rows; both CSVs include `diseaseId` and `diseaseName` columns."
    )
    return "\n\n".join(parts)


def _prepare_display_frame(results: pd.DataFrame) -> pd.DataFrame:
    display_df = results.copy()
    display_df["Rank"] = display_df["rank"].astype(int)
    display_df["Gene"] = display_df["approvedSymbol"]
    display_df["Gene Name"] = display_df["approvedName"]
    display_df["OTRec score"] = display_df["otrec_score"].round(3)
    display_df["Open Targets score"] = display_df["ot_score"].round(3)
    display_df["OTTree score"] = display_df["ottree_pred"].round(3)
    if "tractability" in display_df.columns:
        display_df["Tractability"] = display_df["tractability"].map(
            _format_tractability
        )
    else:
        display_df["Tractability"] = "—"
    display_df["Packaged label"] = display_df["known_label"].map(_known_status)
    display_df["Function"] = display_df[FUNCTION_COLUMN].map(_truncate_text)
    display_df["Open Targets link"] = display_df["targetId"].map(
        lambda target_id: f"https://platform.opentargets.org/target/{target_id}"
    )
    return display_df[DISPLAY_COLUMNS].copy()


@lru_cache(maxsize=128)
def _score_and_enrich_results_cached(
    disease_id: str,
) -> tuple[dict[str, object] | None, pd.DataFrame]:
    disease_rows = disease_df.loc[disease_df["diseaseId"] == disease_id]
    if disease_rows.empty:
        return None, pd.DataFrame()

    disease_row = disease_rows.iloc[0].copy()
    model, cand_embs = get_runtime()
    q_emb = model.encode_q(
        tf.constant([disease_row["disease_text"]]),
        tf.constant([disease_id]),
    )
    q_emb = tf.nn.l2_normalize(q_emb, axis=1).numpy()[0]

    raw_sim = cand_embs @ q_emb
    scores = model.cls_head(raw_sim.reshape(-1, 1)).numpy().flatten()

    results = target_df.copy()
    results["diseaseId"] = disease_id
    results["targetId"] = results["targetId"].astype(str)
    results["otrec_score"] = scores.astype(float)
    results["rank"] = (
        pd.Series(scores, index=results.index)
        .rank(method="first", ascending=False)
        .astype(int)
    )
    results["percentile"] = (results["rank"] / len(results) * 100.0).astype(float)

    annotations = build_result_annotations(disease_id, results["targetId"])
    annotations = annotations.copy()
    annotations["diseaseId"] = annotations["diseaseId"].astype(str)
    annotations["targetId"] = annotations["targetId"].astype(str)
    results = results.merge(annotations, on=["diseaseId", "targetId"], how="left")
    for column in ["ot_score", "ottree_pred", "otrec_oof_pred", "known_label"]:
        if column not in results.columns:
            results[column] = np.nan
        results[column] = pd.to_numeric(results[column], errors="coerce")
    results["comparison_available"] = results["otrec_oof_pred"].notna()
    return disease_row.to_dict(), results


def _score_and_enrich_results(disease_id: str) -> tuple[pd.Series | None, pd.DataFrame]:
    disease_row_dict, results = _score_and_enrich_results_cached(disease_id)
    if disease_row_dict is None:
        return None, pd.DataFrame()
    return pd.Series(disease_row_dict), results.copy(deep=True)


def recommend_targets(
    disease_id: str,
    top_k: int = 25,
    filter_mode: str = FILTER_HIDE_PACKAGED,
    sort_label: str = "OTRec score",
    min_score: float = 0.0,
):
    if not disease_id:
        return (
            "## Select a disease to begin",
            "",
            _render_results_table(_empty_results(), DISPLAY_COLUMNS),
            None,
            None,
        )

    try:
        disease_row, results = _score_and_enrich_results(disease_id)
    except Exception as error:
        return (
            "## Unable to load OTRec runtime",
            "First use downloads model weights and computes embeddings, so the initial response can take around a minute.\n\n"
            f"Runtime error: {_safe_error_message(error)}",
            _render_results_table(_empty_results(), DISPLAY_COLUMNS),
            None,
            None,
        )
    if disease_row is None or results.empty:
        return (
            f"## Disease `{disease_id}` was not found",
            "Try a broader name, exact identifier, or one of the examples below.",
            _render_results_table(_empty_results(), DISPLAY_COLUMNS),
            None,
            None,
        )

    total_ranked_count = len(results)
    comparison_row_count = int(
        results.get("comparison_available", pd.Series(False)).sum()
    )
    available_ot_score_count = int(
        results.get("ot_score", pd.Series(dtype=float)).notna().sum()
    )
    available_ottree_count = int(
        results.get("ottree_pred", pd.Series(dtype=float)).notna().sum()
    )

    if min_score > 0:
        results = results[results["otrec_score"] >= float(min_score)].copy()

    if filter_mode == FILTER_HIDE_PACKAGED:
        results = results[
            (results["known_label"] != 1) | (results["known_label"].isna())
        ].copy()
    elif filter_mode == FILTER_ONLY_PACKAGED:
        results = results[results["known_label"] == 1].copy()

    sort_column, descending = SORT_OPTIONS[sort_label]
    fallback_sort = False
    if sort_column != "otrec_score" and results[sort_column].notna().sum() == 0:
        sort_column = "otrec_score"
        descending = True
        fallback_sort = True

    results = results.sort_values(
        by=[sort_column, "otrec_score", "approvedSymbol"],
        ascending=[not descending, False, True],
        na_position="last",
    ).copy()

    if results.empty:
        summary = _build_summary(
            disease_row,
            shown_count=0,
            candidate_count=0,
            total_ranked_count=total_ranked_count,
        )
        note = _build_note(
            fallback_sort,
            sort_label,
            min_score,
            comparison_row_count,
            available_ot_score_count,
            available_ottree_count,
            candidate_count=0,
            total_ranked_count=total_ranked_count,
        )
        return (
            summary,
            note,
            _render_results_table(_empty_results(), DISPLAY_COLUMNS),
            None,
            None,
        )

    limited_results = results.head(int(top_k)).copy()
    display_df = _prepare_display_frame(limited_results)
    summary = _build_summary(
        disease_row,
        shown_count=len(display_df),
        candidate_count=len(results),
        total_ranked_count=total_ranked_count,
    )
    note = _build_note(
        fallback_sort,
        sort_label,
        min_score,
        comparison_row_count,
        available_ot_score_count,
        available_ottree_count,
        candidate_count=len(results),
        total_ranked_count=total_ranked_count,
    )
    csv_path = _export_target_results(results, disease_row)

    # Full unfiltered druggable-genome ranking export.
    _, full_results = _score_and_enrich_results(disease_id)
    full_results = full_results.sort_values(
        by=["otrec_score", "approvedSymbol"], ascending=[False, True]
    )
    full_results["rank"] = np.arange(1, len(full_results) + 1)
    full_csv_path = _export_target_results(full_results, disease_row)
    return (
        summary,
        note,
        _render_results_table(display_df, DISPLAY_COLUMNS),
        csv_path,
        full_csv_path,
    )


def _resolve_disease_id(search_query: str, disease_id: str | None) -> tuple[str, str]:
    if disease_id:
        return str(disease_id), ""

    query = (search_query or "").strip()
    if len(query) < 2:
        return "", "Choose a disease from the dropdown before ranking."

    lowered_query = query.lower()
    exact_id_matches = disease_df[
        disease_df["diseaseId"].astype(str).str.lower().eq(lowered_query)
    ]
    if not exact_id_matches.empty:
        return str(exact_id_matches.iloc[0]["diseaseId"]), ""

    exact_name_matches = disease_df[
        disease_df["name"].astype(str).str.lower().eq(lowered_query)
    ]
    if len(exact_name_matches) == 1:
        return str(exact_name_matches.iloc[0]["diseaseId"]), ""
    if len(exact_name_matches) > 1:
        return "", "Multiple diseases matched exactly; choose one from the dropdown."

    return "", "No exact disease match was found; choose one from the dropdown."


def run_disease_query(
    search_query: str,
    disease_id: str,
    top_k: int = 25,
    filter_mode: str = FILTER_HIDE_PACKAGED,
    sort_label: str = "OTRec score",
    min_score: float = 0.0,
):
    resolved_disease_id, resolution_message = _resolve_disease_id(
        search_query, disease_id
    )
    if not resolved_disease_id:
        return (
            "## Select a disease to begin",
            resolution_message,
            _render_results_table(_empty_results(), DISPLAY_COLUMNS),
            None,
            None,
        )
    return recommend_targets(
        resolved_disease_id,
        top_k=top_k,
        filter_mode=filter_mode,
        sort_label=sort_label,
        min_score=min_score,
    )


# ============================================
#  REVERSE QUERY: target → ranked diseases
# ============================================


@lru_cache(maxsize=128)
def _score_diseases_for_target_cached(target_id: str):
    target_rows = target_df.loc[target_df["targetId"] == target_id]
    if target_rows.empty:
        return None, pd.DataFrame()
    target_row = target_rows.iloc[0]
    model, dis_embs = get_disease_runtime()
    k_emb = model.encode_k(
        tf.constant([target_row["target_text"]]),
        tf.constant([target_id]),
    )
    k_emb = tf.nn.l2_normalize(k_emb, axis=1).numpy()[0]
    raw_sim = dis_embs @ k_emb
    scores = model.cls_head(raw_sim.reshape(-1, 1)).numpy().flatten()
    out = disease_df[["diseaseId", "name", "description"]].copy()
    out["otrec_score"] = scores.astype(float)
    out = out.sort_values("otrec_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return target_row.to_dict(), out


def _prepare_disease_display(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    out["Rank"] = out["rank"].astype(int)
    out["Disease"] = out["name"]
    out["Disease ID"] = out["diseaseId"]
    out["OTRec score"] = out["otrec_score"].round(3)
    out["Description"] = out["description"].map(_truncate_text)
    out["Open Targets link"] = out["diseaseId"].map(
        lambda did: f"https://platform.opentargets.org/disease/{did}"
    )
    return out[DISEASE_DISPLAY_COLUMNS].copy()


def recommend_diseases(target_id: str, top_k: int = 25, min_score: float = 0.0):
    if not target_id:
        return (
            "## Select a target to begin",
            _render_results_table(_empty_disease_results(), DISEASE_DISPLAY_COLUMNS),
            None,
        )
    try:
        target_dict, results = _score_diseases_for_target_cached(target_id)
    except Exception as error:
        return (
            "## Unable to load OTRec runtime",
            _render_results_table(
                pd.DataFrame(
                    [
                        {
                            "Rank": "",
                            "Disease": f"Runtime error: {_safe_error_message(error)}",
                            "Disease ID": "",
                            "OTRec score": "",
                            "Description": "First use downloads model weights and computes embeddings.",
                            "Open Targets link": "",
                        }
                    ]
                ),
                DISEASE_DISPLAY_COLUMNS,
            ),
            None,
        )
    if target_dict is None or results.empty:
        return (
            f"## Target `{target_id}` was not found",
            _render_results_table(_empty_disease_results(), DISEASE_DISPLAY_COLUMNS),
            None,
        )
    if min_score > 0:
        results = results[results["otrec_score"] >= float(min_score)].copy()
    summary = (
        f"## {target_dict.get('approvedSymbol', target_id)}  \u00b7  "
        f"`{target_id}`\n"
        f"[Open Targets](https://platform.opentargets.org/target/{target_id})\n\n"
        f"{_truncate_text(target_dict.get('approvedName', ''), 200)}\n\n"
        f"Showing top {min(int(top_k), len(results))} of {len(results)} diseases."
    )
    display_df = _prepare_disease_display(results.head(int(top_k)))
    csv_path = _export_disease_results(results, target_dict)
    return summary, _render_results_table(display_df, DISEASE_DISPLAY_COLUMNS), csv_path


def _resolve_target_id(search_query: str, target_id: str | None) -> tuple[str, str]:
    if target_id:
        return str(target_id), ""

    query = (search_query or "").strip()
    if len(query) < 2:
        return "", "Choose a target from the dropdown before ranking."

    lowered = query.lower()
    exact_id_matches = target_df[
        target_df["targetId"].astype(str).str.lower().eq(lowered)
    ]
    if not exact_id_matches.empty:
        return str(exact_id_matches.iloc[0]["targetId"]), ""

    exact_symbol_matches = target_df[
        target_df["approvedSymbol"].astype(str).str.lower().eq(lowered)
    ]
    if len(exact_symbol_matches) == 1:
        return str(exact_symbol_matches.iloc[0]["targetId"]), ""
    if len(exact_symbol_matches) > 1:
        return "", "Multiple targets matched exactly; choose one from the dropdown."

    return "", "No exact target match was found; choose one from the dropdown."


def run_target_query(
    search_query: str, target_id: str, top_k: int = 25, min_score: float = 0.0
):
    resolved_target_id, resolution_message = _resolve_target_id(search_query, target_id)
    if not resolved_target_id:
        return (
            f"## {resolution_message}",
            _render_results_table(_empty_disease_results(), DISEASE_DISPLAY_COLUMNS),
            None,
        )
    return recommend_diseases(
        resolved_target_id,
        top_k=top_k,
        min_score=min_score,
    )


def _empty_disease_results() -> pd.DataFrame:
    return pd.DataFrame(columns=DISEASE_DISPLAY_COLUMNS)


def search_targets(query):
    if not query or len(query) < 2:
        return gr.update(choices=[], value=None)
    query = query.strip()
    lowered = query.lower()
    mask = (
        target_df["approvedSymbol"].str.contains(query, case=False, na=False)
        | target_df["targetId"].str.contains(query, case=False, na=False)
        | target_df["approvedName"]
        .astype(str)
        .str.contains(query, case=False, na=False)
    )
    matches = target_df.loc[mask].copy()
    if matches.empty:
        return gr.update(choices=[], value=None)
    matches["exact_sym"] = matches["approvedSymbol"].str.lower().eq(lowered)
    matches["sym_starts"] = (
        matches["approvedSymbol"].str.lower().str.startswith(lowered)
    )
    matches = matches.sort_values(
        by=["exact_sym", "sym_starts", "approvedSymbol"],
        ascending=[False, False, True],
    ).head(30)
    choices = [
        (
            f"{row['approvedSymbol']} — {row['approvedName']} ({row['targetId']})",
            row["targetId"],
        )
        for _, row in matches.iterrows()
    ]
    return gr.update(choices=choices, value=None)


# ============================================
#  GRADIO APP
# ============================================


def search_diseases(query):
    if not query or len(query) < 2:
        return gr.update(choices=[], value=None)

    query = query.strip()
    lowered_query = query.lower()
    mask = (
        disease_df["name"].str.contains(query, case=False, na=False)
        | disease_df["diseaseId"].str.contains(query, case=False, na=False)
        | disease_df["synonyms"].astype(str).str.contains(query, case=False, na=False)
        | disease_df["ExactSynonyms"]
        .astype(str)
        .str.contains(query, case=False, na=False)
        | disease_df["description"]
        .astype(str)
        .str.contains(query, case=False, na=False)
    )

    matches = disease_df.loc[mask].copy()
    if matches.empty:
        return gr.update(choices=[], value=None)

    matches["exact_id"] = matches["diseaseId"].str.lower().eq(lowered_query)
    matches["exact_name"] = matches["name"].astype(str).str.lower().eq(lowered_query)
    matches["name_starts"] = (
        matches["name"].astype(str).str.lower().str.startswith(lowered_query)
    )
    matches = matches.sort_values(
        by=["exact_id", "exact_name", "name_starts", "name", "diseaseId"],
        ascending=[False, False, False, True, True],
    ).head(30)

    choices = [
        (f"{row['name']} ({row['diseaseId']})", row["diseaseId"])
        for _, row in matches.iterrows()
    ]

    return gr.update(choices=choices, value=None)


def launch():
    examples = [
        ["spinal muscular atrophy"],
        ["ulcerative colitis"],
        ["systemic sclerosis"],
        ["DOID_0050890"],
    ]

    with gr.Blocks(title="OTRec") as demo:
        gr.Markdown(
            """
            # OTRec — disease$\\leftrightarrow$target prioritization

            Rank druggable-genome genes for a given disease (forward), or rank diseases for a given gene (reverse).
            Open Targets and OTTree comparison columns are shown only where packaged comparison data is available.
            The first query downloads model weights and precomputes embeddings, so the initial response can take around a minute.
            Research screening tool, not clinical evidence.
            """
        )

        with gr.Tabs():
            with gr.Tab("Disease \u2192 Targets"):
                with gr.Row():
                    search_box = gr.Textbox(
                        label="Search disease",
                        placeholder="Name, synonym, or disease ID (e.g. DOID_0050890)",
                        lines=1,
                        scale=3,
                    )
                    did_dropdown = gr.Dropdown(
                        label="Disease",
                        choices=[],
                        interactive=True,
                        scale=2,
                    )

                with gr.Row():
                    filter_mode = gr.Dropdown(
                        label="Packaged label filter",
                        choices=[
                            FILTER_HIDE_PACKAGED,
                            FILTER_ALL,
                            FILTER_ONLY_PACKAGED,
                        ],
                        value=FILTER_HIDE_PACKAGED,
                    )
                    sort_by = gr.Dropdown(
                        label="Sort by",
                        choices=list(SORT_OPTIONS.keys()),
                        value="OTRec score",
                    )
                    topk = gr.Slider(5, 200, value=25, step=5, label="Results to show")
                    min_score = gr.Slider(
                        0.0, 0.95, value=0.0, step=0.05, label="Min OTRec score"
                    )

                btn = gr.Button("Rank targets", variant="primary")

                disease_summary = gr.Markdown(value="## Select a disease to begin")
                coverage_note = gr.Markdown(
                    value=(
                        "Packaged label semantics and comparison coverage will appear here once you rank a disease."
                    )
                )

                gr.Markdown("Predicted targets")
                out_df = gr.HTML(
                    value=_render_results_table(_empty_results(), DISPLAY_COLUMNS)
                )
                with gr.Row():
                    out_file = gr.File(label="Download full filtered ranking (CSV)")
                    out_file_full = gr.File(
                        label="Download full unfiltered ranking (CSV)"
                    )

                gr.Examples(examples=examples, inputs=search_box)

                search_box.input(
                    fn=search_diseases, inputs=search_box, outputs=did_dropdown
                )
                search_box.submit(
                    fn=search_diseases, inputs=search_box, outputs=did_dropdown
                )

                inputs = [
                    search_box,
                    did_dropdown,
                    topk,
                    filter_mode,
                    sort_by,
                    min_score,
                ]
                outputs = [
                    disease_summary,
                    coverage_note,
                    out_df,
                    out_file,
                    out_file_full,
                ]

                btn.click(fn=run_disease_query, inputs=inputs, outputs=outputs)
                topk.change(fn=run_disease_query, inputs=inputs, outputs=outputs)
                filter_mode.change(fn=run_disease_query, inputs=inputs, outputs=outputs)
                sort_by.change(fn=run_disease_query, inputs=inputs, outputs=outputs)
                min_score.change(fn=run_disease_query, inputs=inputs, outputs=outputs)

            with gr.Tab("Target \u2192 Diseases"):
                gr.Markdown(
                    "Reverse query: rank diseases by the model's predicted "
                    "relevance to a given gene/target."
                )
                with gr.Row():
                    t_search_box = gr.Textbox(
                        label="Search target",
                        placeholder="Gene symbol or Ensembl ID (e.g. TNF, ENSG00000232810)",
                        lines=1,
                        scale=3,
                    )
                    tid_dropdown = gr.Dropdown(
                        label="Target", choices=[], interactive=True, scale=2
                    )
                with gr.Row():
                    t_topk = gr.Slider(
                        5, 200, value=25, step=5, label="Results to show"
                    )
                    t_min_score = gr.Slider(
                        0.0, 0.95, value=0.0, step=0.05, label="Min OTRec score"
                    )
                t_btn = gr.Button("Rank diseases", variant="primary")
                target_summary = gr.Markdown(value="## Select a target to begin")
                gr.Markdown("Predicted diseases")
                t_out_df = gr.HTML(
                    value=_render_results_table(
                        _empty_disease_results(), DISEASE_DISPLAY_COLUMNS
                    )
                )
                t_out_file = gr.File(label="Download full disease ranking (CSV)")
                t_search_box.input(
                    fn=search_targets, inputs=t_search_box, outputs=tid_dropdown
                )
                t_search_box.submit(
                    fn=search_targets, inputs=t_search_box, outputs=tid_dropdown
                )
                t_inputs = [t_search_box, tid_dropdown, t_topk, t_min_score]
                t_outputs = [target_summary, t_out_df, t_out_file]
                t_btn.click(fn=run_target_query, inputs=t_inputs, outputs=t_outputs)
                t_topk.change(fn=run_target_query, inputs=t_inputs, outputs=t_outputs)
                t_min_score.change(
                    fn=run_target_query, inputs=t_inputs, outputs=t_outputs
                )

    demo.queue(default_concurrency_limit=2).launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    launch()

"""Generate a NEW novel-candidate set from the 25.12-trained model.

Ports predict_all_global / format_predictions / explode_and_merge_positives
verbatim from code/1-Train-DL-Retriever.ipynb (cells ~4-5, lines 212-399 of the
nbconverted script) and the caller cell (~1456-1489): candidates = druggable
genome targets, k=800 retrieval per disease by cosine similarity then scored
by the trained cls_head, min_prob=0.65, top_n=200 per disease, novelty
anti-join against 25.12 known positives.

Additions beyond the original recipe (this is a NEW, separate release, not a
replacement):
  - ottree_score: a CatBoost model trained on the 25.12 label frame (same
    recipe as OTRec/baselines/run_temporal_repeated.py train_ottree_once)
    scores every S1b-nominated pair, giving a second opinion alongside OTRec.
  - target_nomination_count: per-target count of distinct nominating diseases
    in S1b -- the popularity-diagnostic column the paper's Section 4.4 already
    describes in prose but the original S1/S2 files never shipped.

Outputs (NEW files, old S1/S2 never touched):
  retrain_2512/Outputs/S1b-DL_novel_predictions_2512.csv
  retrain_2512/Outputs/S2b-DL_novel+known_candidates_2512.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, "/mnt/d/Research/OpenTargetsTransfer/OTRec/gradio")
from dl_model_def import build_two_tower_model

REPO = Path("/mnt/d/Research/OpenTargetsTransfer")
OUT = REPO / "retrain_2512"
(OUT / "Outputs").mkdir(exist_ok=True)

MIN_PROB = 0.65
TOP_N = 200
K_RETRIEVE = 800

df_learn = pd.read_parquet(OUT / "df_learn_2512.parquet")
disease_df = pd.read_parquet(OUT / "disease_df_2512.parquet").rename(columns={"disease_text_embed": "disease_text"})
target_df = pd.read_parquet(OUT / "target_df_2512.parquet").rename(columns={"target_text_embed": "target_text"})

model = build_two_tower_model(df_learn)
model.load_weights(str(OUT / "model.weights.h5"))
print(f"model loaded; df_learn {df_learn.shape}", flush=True)

druggable_genome_list = pd.read_csv(REPO / "data" / "finan_proc_druggable_genome_list.csv")["ensembl_gene_id"]
candidates_df = (
    target_df[["targetId", "target_text"]].drop_duplicates(subset=["targetId"]).reset_index(drop=True)
)
n0 = len(candidates_df)
candidates_df = candidates_df.loc[
    candidates_df["targetId"].isin(druggable_genome_list) | candidates_df["targetId"].isin(df_learn["targetId"])
].reset_index(drop=True)
print(f"candidates: {n0} -> {len(candidates_df)} (druggable genome or training target)", flush=True)

target_subset = disease_df.loc[disease_df["diseaseId"].isin(df_learn["diseaseId"])].copy()
print(f"scoring {len(target_subset)} diseases x {len(candidates_df)} candidates, k={K_RETRIEVE}", flush=True)


def predict_all_global(model, disease_df, candidates_df, batch_size=2048, k=100):
    cand_embs = model.encode_k(candidates_df["target_text"].to_numpy(), candidates_df["targetId"].to_numpy())
    cand_embs = tf.nn.l2_normalize(cand_embs, axis=1, epsilon=1e-16)
    disease_ds = tf.data.Dataset.from_tensor_slices({
        "disease_text": disease_df["disease_text"].to_numpy(),
        "diseaseId": disease_df["diseaseId"].to_numpy(),
    }).batch(batch_size)
    results = []
    for batch in disease_ds:
        q_embs = tf.nn.l2_normalize(model.encode_q(batch["disease_text"], batch["diseaseId"]), axis=1)
        sim_matrix = tf.matmul(q_embs, cand_embs, transpose_b=True)
        # Rank by the CALIBRATED probability, not by raw cosine similarity.
        # The sign of the learned cls_head kernel is arbitrary: the deployed
        # 25.06 model learned +9.85, this 25.12 model learned -5.52, i.e. it
        # encodes positives as NEGATIVE cosine. The notebook's original
        # top_k-on-cosine step silently assumes a positive kernel and retrieves
        # the worst candidates when the sign flips. cls_head is monotonic in
        # the cosine, so ranking on its output is sign-agnostic and identical
        # to the original behaviour whenever the kernel is positive.
        probs_matrix = tf.reshape(
            model.cls_head(tf.reshape(sim_matrix, (-1, 1))), tf.shape(sim_matrix)
        )
        top_k_probs, top_k_indices = tf.math.top_k(probs_matrix, k=k)
        results.append((batch["diseaseId"].numpy(), top_k_indices.numpy(), top_k_probs.numpy()))
    return results


def format_predictions(raw_results, candidates_df, positives_set, targets_map, top_n=5, min_prob=0.0):
    data = []
    candidate_ids = candidates_df["targetId"].to_numpy()
    for batch_dids, batch_indices, batch_probs in raw_results:
        for i, disease_id_bytes in enumerate(batch_dids):
            disease_id = disease_id_bytes.decode("utf-8")
            found_ids, found_syms, found_probs = [], [], []
            for rank_idx, cand_idx in enumerate(batch_indices[i]):
                target_id = candidate_ids[cand_idx]
                prob = round(float(batch_probs[i][rank_idx]), 3)
                if prob < min_prob:
                    continue
                if (disease_id, target_id) in positives_set:
                    continue
                found_ids.append(target_id)
                found_probs.append(prob)
                found_syms.append(targets_map.get(target_id, {}).get("approvedSymbol", target_id))
                if len(found_ids) >= top_n:
                    break
            if found_ids:
                data.append({"diseaseId": disease_id, "novel_target_ids": found_ids,
                             "novel_target_sym": found_syms, "probabilities": found_probs})
    return pd.DataFrame(data)


def explode_and_merge_positives(preds_df, positives_set, disease_df, targets_map):
    if "name" in preds_df.columns:
        preds_df = preds_df.rename(columns={"name": "diseaseName"})
    list_cols = ["novel_target_ids", "novel_target_sym", "probabilities"]
    if len(preds_df) > 0:
        long_preds = preds_df.explode(list_cols).rename(columns={
            "novel_target_ids": "targetId", "novel_target_sym": "targetSymbol", "probabilities": "score"})
    else:
        long_preds = pd.DataFrame(columns=["diseaseId", "diseaseName", "targetId", "targetSymbol", "score"])
    long_preds["label"] = -1
    long_preds["source"] = "model_prediction"
    cols_to_keep = ["diseaseId", "diseaseName", "targetId", "targetSymbol", "score", "label", "source"]
    long_preds = long_preds[[c for c in cols_to_keep if c in long_preds.columns]].copy()

    if len(positives_set) > 0:
        pos_df = pd.DataFrame(list(positives_set), columns=["diseaseId", "targetId"])
        dise_name_map = disease_df.set_index("diseaseId")["name"].to_dict()
        pos_df["diseaseName"] = pos_df["diseaseId"].map(dise_name_map)
        pos_df["targetSymbol"] = pos_df["targetId"].map(lambda t: targets_map.get(t, {}).get("approvedSymbol", t))
        pos_df["score"] = 1.0
        pos_df["label"] = 1
        pos_df["source"] = "known_positive"
        pos_df = pos_df.reindex(columns=cols_to_keep)
    else:
        pos_df = pd.DataFrame(columns=cols_to_keep)

    combined_df = pd.concat([long_preds, pos_df], axis=0, ignore_index=True)
    if not combined_df.empty:
        combined_df = combined_df.sort_values(by=["diseaseId", "targetId", "label"], ascending=[True, True, False])
        combined_df = combined_df.drop_duplicates(subset=["diseaseId", "targetId"], keep="first")
    return combined_df.sort_values(["diseaseId", "source", "score"], ascending=False).reset_index(drop=True)


# We need `name` on disease_df for the merge below (retain from raw table).
disease_names = pd.read_parquet(REPO / "data" / "historical_ot" / "25_12" / "disease", columns=["id", "name"])
disease_df = disease_df.merge(disease_names, left_on="diseaseId", right_on="id", how="left").drop(columns=["id"])

positives = set(zip(df_learn.query("label==1")["diseaseId"], df_learn.query("label==1")["targetId"]))
targets_dict = target_df.reset_index().set_index("targetId")[[]].to_dict(orient="index")
# approvedSymbol/approvedName aren't in the trimmed target_df (text-only); pull from raw target table.
target_meta = pd.read_parquet(
    REPO / "data" / "historical_ot" / "25_12" / "target", columns=["id", "approvedSymbol", "approvedName"]
).set_index("id")
targets_dict = target_meta.to_dict(orient="index")

raw_results = predict_all_global(model, target_subset, candidates_df, k=K_RETRIEVE)
final_preds_df = format_predictions(raw_results, candidates_df, positives, targets_dict,
                                     top_n=TOP_N, min_prob=MIN_PROB)
print(f"{final_preds_df.shape[0]} diseases with candidates above threshold", flush=True)
final_preds_df = final_preds_df.merge(disease_df[["diseaseId", "name"]], on="diseaseId", how="left")

all_candidates_long_df = explode_and_merge_positives(final_preds_df, positives, disease_df, targets_dict)

disease_pos_counts = df_learn.groupby("diseaseId")["label"].sum()
all_candidates_long_df["disease_num_known_clinical_targets"] = all_candidates_long_df["diseaseId"].map(disease_pos_counts)
all_candidates_long_df["orphan"] = all_candidates_long_df["diseaseId"].isin(
    set(disease_pos_counts[disease_pos_counts <= 0].index))

# --- addition 1: target nomination count (novel rows only, per the paper's own diagnostic) ---
novel_mask = all_candidates_long_df["label"] == -1
nom_counts = all_candidates_long_df.loc[novel_mask].groupby("targetId")["diseaseId"].transform("nunique")
all_candidates_long_df["target_nomination_count"] = 0
all_candidates_long_df.loc[novel_mask, "target_nomination_count"] = nom_counts

# --- addition 2: OTTree second opinion on nominated pairs ---
print("training 25.12 OTTree (CatBoost) for second-opinion scoring...", flush=True)
from catboost import CatBoostClassifier, Pool

feats = ["disease_text", "target_text", "diseaseId"]
df_learn_txt = df_learn.rename(columns={})  # already has disease_text/target_text
train_pool = Pool(df_learn_txt[feats], df_learn_txt["label"],
                   text_features=["disease_text", "target_text"], cat_features=["diseaseId"])
params = {"depth": 8, "eval_metric": "AUC", "random_seed": 42, "verbose": False}
if tf.config.list_physical_devices("GPU"):
    params["task_type"] = "GPU"
ottree = CatBoostClassifier(**params)
ottree.fit(train_pool)

score_pairs = all_candidates_long_df.loc[novel_mask, ["diseaseId", "targetId"]].merge(
    disease_df[["diseaseId", "disease_text"]], on="diseaseId", how="left"
).merge(target_df[["targetId", "target_text"]], on="targetId", how="left")
score_pool = Pool(score_pairs[feats], text_features=["disease_text", "target_text"], cat_features=["diseaseId"])
ottree_scores = ottree.predict_proba(score_pool)[:, 1]
all_candidates_long_df["ottree_score"] = np.nan
all_candidates_long_df.loc[novel_mask, "ottree_score"] = ottree_scores

print("source counts:", all_candidates_long_df["source"].value_counts().to_dict(), flush=True)

all_candidates_long_df.to_csv(OUT / "Outputs" / "S2b-DL_novel+known_candidates_2512.csv", index=False)
s1b_cols = ["diseaseId", "diseaseName", "targetId", "targetSymbol", "score",
            "disease_num_known_clinical_targets", "orphan", "target_nomination_count", "ottree_score"]
all_candidates_long_df.query("label == -1")[s1b_cols].to_csv(
    OUT / "Outputs" / "S1b-DL_novel_predictions_2512.csv", index=False)
print("saved S1b/S2b to", OUT / "Outputs", flush=True)

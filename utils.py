###utils

import os
import numpy as np
import pandas as pd

try:
    import shap
except Exception:
    shap = None

# from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

from sklearn.metrics import (
    log_loss,
)

try:
    from tensorflow.keras.layers import Dot, Activation
    from tensorflow.keras.losses import BinaryCrossentropy
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf
    # import tensorflow_recommenders as tfrs

    # 👉 everything below comes from *tf.keras*
    from tensorflow import keras
    from tensorflow.keras.utils import FeatureSpace
    from tensorflow.keras.layers import TextVectorization
    import numpy as np
    import pandas as pd
    import keras
    from keras.layers import TextVectorization, Embedding, Concatenate
    from keras.utils import FeatureSpace
    # import tensorflow_recommenders as tfrs          # 0.7.3+

except:
    print("No TF in env")
try:
    from keras_rs.layers import BruteForceRetrieval
    from keras_rs.metrics import PrecisionAtK, RecallAtK
    from keras.losses import BinaryFocalCrossentropy
except:
    print("No keras-rs in env")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def group_mean_baseline(test_df, full_df, group_col, label_col="label", fillna="prior"):
    """Broadcast a training-set group positive rate onto a test frame."""
    group_means = full_df.groupby(group_col)[label_col].mean()
    if fillna == "prior":
        prior = group_means.mean()
    else:
        prior = fillna
    return test_df[group_col].map(group_means).fillna(prior).to_numpy(dtype="float32")


def drop_unary_cols(df, dropna=False):
    """
    Drops columns with only one unique value. Handles Object cols (e.g. lists - unhashable) by efficiently by applying
    a per-column check, avoiding unnecessary full-DataFrame conversions.
    """

    def check_col(col):
        """Helper function to apply to each column (Series)."""
        try:
            # 1. Try the fast, native method first
            return col.nunique(dropna=dropna)
        except TypeError:
            # 2. Fallback: convert just this one column to str
            return col.astype(str).nunique(dropna=dropna)

    # df.apply() runs the 'check_col' function on every column
    # The result is a Series of the unique counts for each column
    unique_counts = df.apply(check_col)

    # Keep columns where the unique count is 2 or more
    return df.loc[:, unique_counts >= 2]


def evaluate_prob_series(y_true, y_hat, name, threshold=0.5):
    # Handle epsilon vs eps for log_loss
    ll_kwargs = {"labels": [0, 1]}
    param_names = log_loss.__code__.co_varnames
    if "eps" in param_names:
        ll_kwargs["eps"] = 1e-7
    elif "epsilon" in param_names:
        ll_kwargs["epsilon"] = 1e-7

    # Binary predictions from scores
    y_pred_bin = (y_hat >= threshold).astype("int32")

    print(
        f"{name:18s}  "
        f"AUC={roc_auc_score(y_true, y_hat):.4f}  "
        f"PR-AUC={average_precision_score(y_true, y_hat):.4f}  "
        f"BCE={log_loss(y_true, y_hat, **ll_kwargs):.4f}  "
        f"Acc={accuracy_score(y_true, y_pred_bin):.4f}  "
        f"P={precision_score(y_true, y_pred_bin, zero_division=0):.4f}  "
        f"R={recall_score(y_true, y_pred_bin):.4f}"
    )


def target_mean_baseline(test_df, full_df, fillna="prior"):
    """Return target-wise mean-label predictions from the training dataframe.

    usage:
    y_test = test_df["label"].to_numpy(dtype="float32")
    evaluate_prob_series(y_test,
                         target_mean_baseline(test_df, train_df),
                         "TARGET mean")
    """
    return group_mean_baseline(
        test_df=test_df,
        full_df=full_df,
        group_col="targetId",
        fillna=fillna,
    )


def disease_mean_baseline(test_df, full_df, fillna="prior"):
    """Return disease-wise mean-label predictions from the training dataframe."""
    return group_mean_baseline(
        test_df=test_df,
        full_df=full_df,
        group_col="diseaseId",
        fillna=fillna,
    )


def summarize_group_ranking_metrics(
    df,
    score_col,
    label_col="label",
    group_col="diseaseId",
    ks=(1, 5, 10),
    min_positives=1,
):
    """Summarize within-group ranking quality for shortlist-style evaluation."""
    rows = []

    for group_value, group_df in df.groupby(group_col, sort=False):
        positives = int(group_df[label_col].sum())
        if positives < min_positives:
            continue

        ranked = group_df.sort_values(score_col, ascending=False).reset_index(drop=True)
        labels = ranked[label_col].to_numpy(dtype=int)
        prevalence = float(labels.mean()) if len(labels) else 0.0
        hit_positions = np.flatnonzero(labels > 0)

        row = {
            group_col: group_value,
            "n_candidates": int(len(ranked)),
            "n_positives": positives,
            "prevalence": prevalence,
            "average_precision": average_precision_score(labels, ranked[score_col]),
            "mrr": 0.0
            if len(hit_positions) == 0
            else 1.0 / float(hit_positions[0] + 1),
        }

        for k in ks:
            topk = labels[:k]
            denom = max(1, min(k, len(labels)))
            positives_at_k = int(topk.sum())
            precision_at_k = positives_at_k / denom
            recall_at_k = positives_at_k / positives
            row[f"hit@{k}"] = float(positives_at_k > 0)
            row[f"precision@{k}"] = precision_at_k
            row[f"recall@{k}"] = recall_at_k
            row[f"enrichment@{k}"] = (
                np.nan if prevalence == 0 else precision_at_k / prevalence
            )

        rows.append(row)

    per_group = pd.DataFrame(rows)
    if per_group.empty:
        return per_group, pd.Series(dtype="float64")

    summary = {
        "n_groups": int(len(per_group)),
        "mean_average_precision": float(per_group["average_precision"].mean()),
        "median_average_precision": float(per_group["average_precision"].median()),
        "mean_mrr": float(per_group["mrr"].mean()),
    }
    for k in ks:
        summary[f"mean_hit@{k}"] = float(per_group[f"hit@{k}"].mean())
        summary[f"mean_precision@{k}"] = float(per_group[f"precision@{k}"].mean())
        summary[f"mean_recall@{k}"] = float(per_group[f"recall@{k}"].mean())
        summary[f"mean_enrichment@{k}"] = float(
            per_group[f"enrichment@{k}"].dropna().mean()
        )

    return per_group, pd.Series(summary, dtype="float64")

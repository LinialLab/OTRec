###utils

import os
import numpy as np
import pandas as pd
import shap
# from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
)

import pandas as pd
import re
import os
try:
    from tensorflow.keras.layers import Dot, Activation
    from tensorflow.keras.losses import BinaryCrossentropy
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf
    # import tensorflow_recommenders as tfrs

    # 👉 everything below comes from *tf.keras*
    from tensorflow import keras
    from tensorflow.keras.utils   import FeatureSpace
    from tensorflow.keras.layers  import TextVectorization
    import numpy as np, pandas as pd, tensorflow as tf, keras
    from keras.layers import TextVectorization, Embedding, Concatenate
    from keras.utils  import FeatureSpace
    # import tensorflow_recommenders as tfrs          # 0.7.3+
    import tensorflow as tf#, keras

except:
    print("No TF in env")
try:
    from keras_rs.layers import BruteForceRetrieval
    from keras_rs.metrics import PrecisionAtK, RecallAtK
    from keras.losses import BinaryFocalCrossentropy
except:
    print("No keras-rs in env")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model            import LogisticRegression
from sklearn.metrics                 import roc_auc_score


from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

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

def target_mean_baseline(test_df, full_df,fillna="prior"):
    """Return target-wise oracle mean predictions from full dataset (df_learn).

    usage:
    y_test = test_df["label"].to_numpy(dtype="float32")
    evaluate_prob_series(y_test,
                         test_df[["diseaseId"]].merge(train_df.groupby("diseaseId")["label"].mean(),on="diseaseId",how="left")["label"].fillna(0), # mean of disease in train, apply to test
                         "DISEASE mean")
    """
    target_means = full_df.groupby("targetId")["label"].mean()
    if fillna =="prior":
        prior = target_means.mean()
    else:
        prior = fillna
    return test_df["targetId"].map(target_means).fillna(prior).to_numpy(dtype="float32")





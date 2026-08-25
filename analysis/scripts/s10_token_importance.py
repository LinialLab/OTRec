"""Supplementary S10: token-level feature interpretability.

Fits a sparse logistic-regression surrogate on the temporal training frame
(count-vectorized disease_text + target_text, the same fields the towers
consume) and reports the strongest positive/negative token coefficients per
tower. A second fit strips the clinical-precedence tractability tokens
("Approved Drug", "Advanced Clinical", "Phase 1 Clinical", "Clinical
Precedence") to show how much of the target-tower signal rides on that
channel. Optionally (--ottree) retrains the matched OTTree baseline and
prints CatBoost's native importance over its three inputs.

Outputs:
  analysis/results/s10_token_importance.csv           (all coefficients, both fits)
  analysis/results/s10_ottree_native_importance.csv   (with --ottree)
  Outputs/S10_token_importance.{pdf,png}              (figure)
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]          # .../OTRec
RESULTS = ROOT / "analysis" / "results"
OUTPUTS = ROOT / "Outputs"
SEED = 42

LEAK_TOKENS = ["Approved Drug", "Advanced Clinical", "Phase 1 Clinical", "Clinical Precedence"]
LEAK_RE = re.compile("|".join(re.escape(t) for t in LEAK_TOKENS), flags=re.IGNORECASE)

# Tokens occurring in fewer than this many distinct diseases/targets are
# excluded from the top-token plots: near-categorical identity features
# (a rare disease's own name) inherit that entity's small-sample base rate.
MIN_ENTITY_DF = 5

# Paper accent palette (matches make_fig1_bc.py): blue = positive, orange = negative.
C_POS, C_NEG = "#0f62a0", "#e08214"


def load_frame() -> pd.DataFrame:
    code = ROOT.parent / "code"
    df = pd.read_parquet(code / "history_df.parquet")
    disease_df = pd.read_parquet(code / "copy_proc" / "disease_df.parquet")
    target_df = pd.read_parquet(code / "copy_proc" / "target_df.parquet")
    df = df.merge(disease_df[["diseaseId", "disease_text_embed"]], on="diseaseId", how="left")
    df = df.merge(target_df[["targetId", "target_text_embed"]], on="targetId", how="left")
    df = df.rename(columns={"disease_text_embed": "disease_text", "target_text_embed": "target_text"})
    df["disease_text"] = df["disease_text"].fillna("").astype(str)
    df["target_text"] = df["target_text"].fillna("").astype(str)
    return df


def fit_variant(train, test, strip_leak: bool):
    tgt_train, tgt_test = train["target_text"], test["target_text"]
    if strip_leak:
        tgt_train = tgt_train.str.replace(LEAK_RE, " ", regex=True)
        tgt_test = tgt_test.str.replace(LEAK_RE, " ", regex=True)
    # ngram_range (1,2) so multiword tractability buckets stay one feature.
    vec_d = CountVectorizer(min_df=25, ngram_range=(1, 2), strip_accents="unicode")
    vec_t = CountVectorizer(min_df=25, ngram_range=(1, 2), strip_accents="unicode")
    Xd, Xt = vec_d.fit_transform(train["disease_text"]), vec_t.fit_transform(tgt_train)
    X = hstack([Xd, Xt]).tocsr()
    Xtest = hstack([vec_d.transform(test["disease_text"]), vec_t.transform(tgt_test)]).tocsr()
    clf = LogisticRegression(C=0.5, max_iter=1500, solver="lbfgs")
    clf.fit(X, train["label"])
    pred = clf.predict_proba(Xtest)[:, 1]
    roc = roc_auc_score(test["label"], pred)
    pr = average_precision_score(test["label"], pred)
    names = np.concatenate([
        np.char.add("d: ", vec_d.get_feature_names_out().astype(str)),
        np.char.add("t: ", vec_t.get_feature_names_out().astype(str)),
    ])
    coefs = pd.DataFrame({"feature": names, "coef": clf.coef_.ravel()})
    coefs["tower"] = np.where(coefs["feature"].str.startswith("d: "), "disease", "target")

    # Distinct-entity document frequency: a token that occurs in only a
    # handful of diseases (e.g. a rare disease's own name, "cicatricial
    # alopecia") acts as a near-categorical identity feature and inherits
    # that disease's small-sample positive rate, not a generalizable signal.
    # Reuse the fitted vectorizers (transform, not refit) on one row per
    # unique disease / target so document frequency counts entities, not pairs.
    dedup_dis = train.drop_duplicates("diseaseId")
    dedup_tgt_idx = train["targetId"].drop_duplicates().index
    Xd_doc = vec_d.transform(dedup_dis["disease_text"])
    Xt_doc = vec_t.transform(tgt_train.loc[dedup_tgt_idx])
    entity_df = np.concatenate([
        np.asarray((Xd_doc > 0).sum(axis=0)).ravel(),
        np.asarray((Xt_doc > 0).sum(axis=0)).ravel(),
    ])
    coefs["entity_df"] = entity_df
    return coefs, roc, pr


def plot(coefs: pd.DataFrame, roc: float, pr: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for ax, tower, title in zip(
        axes, ["disease", "target"], ["A. Disease-tower tokens", "B. Target-tower tokens"]
    ):
        sub = coefs[(coefs.tower == tower) & (coefs.entity_df >= MIN_ENTITY_DF)].sort_values("coef")
        top = pd.concat([sub.head(15), sub.tail(15)])
        labels = top["feature"].str.slice(3)
        colors = [C_POS if c > 0 else C_NEG for c in top["coef"]]
        ax.barh(range(len(top)), top["coef"], color=colors, height=0.72)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="#666666", linewidth=0.8)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=12)
        ax.set_xlabel("Logistic-regression coefficient", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"Token-level surrogate importance (temporal training frame; "
        f"target-disjoint holdout ROC-AUC {roc:.3f}, PR-AUC {pr:.3f})",
        fontsize=11,
    )
    for out in (OUTPUTS / "S10_token_importance.pdf", OUTPUTS / "S10_token_importance.png"):
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out)


def plot_leak_comparison(coefs_full: pd.DataFrame, coefs_stripped: pd.DataFrame):
    """Target-tower tokens, full model vs. leak-tokens-stripped, side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for ax, coefs, title in zip(
        axes,
        [coefs_full, coefs_stripped],
        ["A. Full model (with leak tokens)", "B. Leak tokens stripped"],
    ):
        sub = coefs[(coefs.tower == "target") & (coefs.entity_df >= MIN_ENTITY_DF)].sort_values("coef")
        top = pd.concat([sub.head(15), sub.tail(15)])
        labels = top["feature"].str.slice(3)
        colors = [C_POS if c > 0 else C_NEG for c in top["coef"]]
        ax.barh(range(len(top)), top["coef"], color=colors, height=0.72)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="#666666", linewidth=0.8)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=12)
        ax.set_xlabel("Logistic-regression coefficient", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
    for out in (OUTPUTS / "S10b_leak_token_removed.pdf", OUTPUTS / "S10b_leak_token_removed.png"):
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ottree", action="store_true", help="also retrain OTTree and print native importance")
    args = ap.parse_args()

    df = load_frame()
    train_tids, test_tids = train_test_split(
        df["targetId"].unique(), test_size=0.1, random_state=SEED, shuffle=True
    )
    train, test = df[df.targetId.isin(train_tids)], df[df.targetId.isin(test_tids)]
    print(f"train {train.shape} test {test.shape} pos-rate {train.label.mean():.4f}")

    rows = []
    for strip in (False, True):
        coefs, roc, pr = fit_variant(train, test, strip_leak=strip)
        tag = "stripped" if strip else "full"
        print(f"[{tag}] holdout ROC {roc:.4f} PR {pr:.4f}")
        coefs["variant"], coefs["roc"], coefs["pr"] = tag, roc, pr
        rows.append(coefs)
        if not strip:
            plot(coefs, roc, pr)
    plot_leak_comparison(rows[0], rows[1])
    pd.concat(rows).to_csv(RESULTS / "s10_token_importance.csv", index=False)
    print("wrote", RESULTS / "s10_token_importance.csv")

    if args.ottree:
        from catboost import CatBoostClassifier, Pool

        features = ["disease_text", "target_text", "diseaseId"]
        pool = Pool(
            data=train[features], label=train["label"],
            text_features=["disease_text", "target_text"], cat_features=["diseaseId"],
        )
        model = CatBoostClassifier(depth=8, eval_metric="AUC", random_seed=SEED, verbose=False)
        model.fit(pool)
        imp = pd.DataFrame({
            "feature": features,
            "prediction_values_change": model.get_feature_importance(pool),
        })
        print(imp)
        imp.to_csv(RESULTS / "s10_ottree_native_importance.csv", index=False)
        print("wrote", RESULTS / "s10_ottree_native_importance.csv")


if __name__ == "__main__":
    main()

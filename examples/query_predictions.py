"""Query the released OTRec prediction files without loading the model.

Usage (from the repository root):
    python examples/query_predictions.py --disease obesity
    python examples/query_predictions.py --target GIPR
    python examples/query_predictions.py --disease "lung cancer" --top 20 --include-known

Reads Outputs/S1-DL_novel_predictions.csv (novel candidates only) or, with
--include-known, Outputs/S2-DL_novel+known_candidates.csv (novel + known
clinical associations). Requires only pandas.
"""
import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--disease", help="disease name (case-insensitive substring match)")
    ap.add_argument("--target", help="gene symbol (case-insensitive exact match)")
    ap.add_argument("--top", type=int, default=10, help="rows to show per match (default 10)")
    ap.add_argument("--include-known", action="store_true",
                    help="query S2 (novel + known associations) instead of S1 (novel only)")
    args = ap.parse_args()
    if not args.disease and not args.target:
        ap.error("give --disease and/or --target")

    fname = ("S2-DL_novel+known_candidates.csv" if args.include_known
             else "S1-DL_novel_predictions.csv")
    df = pd.read_csv(REPO / "Outputs" / fname)

    if args.disease:
        hits = df[df.diseaseName.str.contains(args.disease, case=False, na=False)]
        if hits.empty:
            print(f"No disease name containing '{args.disease}' in {fname}.")
        for did, g in hits.groupby("diseaseId"):
            g = g.sort_values("score", ascending=False)
            print(f"\n{g.diseaseName.iloc[0]} ({did}) — {len(g)} predictions")
            print(g.head(args.top)[["targetSymbol", "targetId", "score"]]
                  .to_string(index=False))

    if args.target:
        hits = df[df.targetSymbol.str.upper() == args.target.upper()]
        if hits.empty:
            print(f"No target symbol '{args.target}' in {fname}.")
        else:
            hits = hits.sort_values("score", ascending=False)
            print(f"\n{args.target.upper()} — predicted for {hits.diseaseId.nunique()} diseases")
            print(hits.head(args.top)[["diseaseName", "diseaseId", "score"]]
                  .to_string(index=False))


if __name__ == "__main__":
    main()

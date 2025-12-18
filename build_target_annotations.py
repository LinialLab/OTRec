#!/usr/bin/env python
#### additional meta attributes, used in results analysis/eda

import argparse
import os
import time
from typing import Optional

import requests
import pandas as pd
from tqdm.auto import tqdm

# --- constants --------------------------------------------------------------

ENSEMBL_REST = "https://rest.ensembl.org"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
PDBE_BEST_STRUCT = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures"

STRING_ALIAS_URL = (
    "https://stringdb-downloads.org/download/protein.aliases.v12.0/"
    "9606.protein.aliases.v12.0.txt.gz"
)
STRING_LINKS_URL = (
    "https://stringdb-downloads.org/download/protein.links.v12.0/"
    "9606.protein.links.v12.0.txt.gz"
)


# --- helpers ----------------------------------------------------------------


def load_unique_targets(path: str) -> pd.DataFrame:
    """Read input predictions and keep one row per target."""
    df = pd.read_csv(path)
    if not {"targetId", "targetSymbol"} <= set(df.columns):
        raise ValueError("Input CSV must contain 'targetId' and 'targetSymbol' columns")
    t = df[["targetId", "targetSymbol"]].drop_duplicates().reset_index(drop=True)
    print(f"[load] Loaded {len(t)} unique targets from {path}")
    return t


# ---------------- Ensembl: paralogs ----------------------------------------


def fetch_paralog_count_ensembl(ensembl_id: str,
                                retries: int = 3,
                                sleep: float = 0.1) -> int:
    """
    Count same-species paralogs for a human Ensembl gene.

    Uses /homology/id/human/:id?type=paralogues;format=condensed
    and counts homologies labelled 'within_species_paralog'.
    """
    if not isinstance(ensembl_id, str) or not ensembl_id:
        return 0

    ext = f"/homology/id/human/{ensembl_id}?type=paralogues;format=condensed"
    url = ENSEMBL_REST + ext
    headers = {"Content-Type": "application/json"}

    for _ in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=30)
        except requests.RequestException:
            time.sleep(sleep)
            continue

        if r.status_code == 200:
            try:
                j = r.json()
            except ValueError:
                return 0

            try:
                homologies = j["data"][0].get("homologies", []) or []
            except (KeyError, IndexError, TypeError):
                return 0

            cnt = 0
            for h in homologies:
                htype = h.get("homology_type") or h.get("type")
                if htype == "within_species_paralog":
                    cnt += 1
            return cnt

        if r.status_code in (429, 503):
            time.sleep(sleep)
            continue

        return 0

    return 0



def add_paralog_counts(targets: pd.DataFrame) -> pd.DataFrame:
    print("[ensembl] Fetching paralog counts (within_species_paralog) per target")
    counts = []
    for ensg in tqdm(targets["targetId"], desc="Ensembl paralogs"):
        counts.append(fetch_paralog_count_ensembl(ensg))
    targets["paralogCount"] = counts
    return targets


# ---------------- UniProt: symbol → accession ------------------------------


def fetch_uniprot_for_symbol(
    symbol: str, retries: int = 3, sleep: float = 0.3
) -> Optional[str]:
    """Return a UniProt accession for a human gene symbol (first hit)."""
    if not isinstance(symbol, str) or not symbol:
        return None

    params = {
        "query": f"gene_exact:{symbol} AND organism_id:9606",
        "fields": "accession",
        "format": "tsv",
        "size": "1",
    }

    for _ in range(retries):
        try:
            r = requests.get(UNIPROT_SEARCH, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(sleep)
            continue

        if r.status_code == 200:
            text = r.text.strip()
            if not text:
                return None
            lines = text.splitlines()
            if len(lines) < 2:
                return None
            # first column in first data row is the accession
            first_row = lines[1].split("\t")
            if not first_row:
                return None
            acc = first_row[0].strip()
            return acc or None
        elif r.status_code in (429, 503):
            time.sleep(sleep)
            continue
        else:
            return None

    return None


def add_uniprot_ids(targets: pd.DataFrame) -> pd.DataFrame:
    print("[uniprot] Mapping gene symbols -> UniProt accessions")
    symbols = sorted(
        {s for s in targets["targetSymbol"].dropna().unique() if isinstance(s, str) and s}
    )

    sym_to_acc: dict[str, Optional[str]] = {}
    for sym in tqdm(symbols, desc="UniProt (by symbol)"):
        sym_to_acc[sym] = fetch_uniprot_for_symbol(sym)

    targets["uniprot_id"] = targets["targetSymbol"].map(sym_to_acc)
    n_ok = targets["uniprot_id"].notna().sum()
    n_unique = len({u for u in targets["uniprot_id"].dropna().unique()})
    print(
        f"[uniprot] Got UniProt for {n_ok}/{len(targets)} target rows "
        f"({n_unique} unique accessions)"
    )
    return targets


# ---------------- PDBe: hasPDB ---------------------------------------------


def has_pdb_for_uniprot(acc: str, retries: int = 3, sleep: float = 0.3) -> int:
    """Return 1 if PDBe has at least one best_structure for this UniProt accession."""
    if not isinstance(acc, str) or not acc:
        return 0

    url = f"{PDBE_BEST_STRUCT}/{acc}"
    headers = {"Accept": "application/json"}

    for _ in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=30)
        except requests.RequestException:
            time.sleep(sleep)
            continue

        if r.status_code == 200:
            try:
                j = r.json()
            except ValueError:
                return 0

            # PDBe returns a dict keyed by UniProt accession; any non-empty
            # mapping means at least one PDB structure exists.
            if isinstance(j, dict) and j:
                return 1
            return 0
        elif r.status_code == 404:
            return 0
        elif r.status_code in (429, 503):
            time.sleep(sleep)
            continue
        else:
            return 0

    return 0


def add_pdb_flags(targets: pd.DataFrame) -> pd.DataFrame:
    print("[pdbe] Checking PDB coverage via best_structures")
    uniq_uniprot = sorted(
        {u for u in targets["uniprot_id"].dropna().unique() if isinstance(u, str) and u}
    )
    pdb_map: dict[str, int] = {}
    for acc in tqdm(uniq_uniprot, desc="PDBe best_structures"):
        pdb_map[acc] = has_pdb_for_uniprot(acc)

    targets["hasPDB"] = targets["uniprot_id"].map(pdb_map).fillna(0).astype(int)
    # For compatibility with earlier EDA code that looked at hasStructure
    targets["hasStructure"] = targets["hasPDB"]
    return targets


# ---------------- STRING: aliases + links → degree -------------------------


def download_string_files(workdir: str) -> tuple[str, str]:
    os.makedirs(workdir, exist_ok=True)
    alias_path = os.path.join(workdir, "9606.protein.aliases.v12.0.txt.gz")
    links_path = os.path.join(workdir, "9606.protein.links.v12.0.txt.gz")

    for url, path in ((STRING_ALIAS_URL, alias_path), (STRING_LINKS_URL, links_path)):
        if not os.path.exists(path):
            print(f"[download] Downloading {url} -> {path}")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        else:
            print(f"[download] Found existing file: {path}")

    return alias_path, links_path


def add_string_ppi(
    targets: pd.DataFrame, workdir: str, score_threshold: int = 700
) -> pd.DataFrame:
    alias_path, links_path = download_string_files(workdir)

    # --- map targets -> STRING protein IDs via alias file -------------------
    print("[string] Loading alias table")
    aliases = pd.read_csv(alias_path, sep="\t", header=0)

    # Normalise column name: file uses '#string_protein_id'
    if "#string_protein_id" in aliases.columns:
        aliases = aliases.rename(columns={"#string_protein_id": "string_protein_id"})
    if "string_protein_id" not in aliases.columns or "alias" not in aliases.columns:
        raise RuntimeError(f"Unexpected columns in alias file: {aliases.columns}")

    aliases = aliases[["string_protein_id", "alias"]].dropna()
    aliases["alias"] = aliases["alias"].astype(str)
    alias_to_id = aliases.drop_duplicates()

    # Map by gene symbol
    sym_map = alias_to_id[
        alias_to_id["alias"].isin(targets["targetSymbol"].astype(str))
    ].copy()
    sym_map = sym_map.rename(columns={"alias": "targetSymbol"})
    sym_map = sym_map.drop_duplicates(subset=["targetSymbol"])

    # Map by UniProt accession, if we have it
    if "uniprot_id" in targets.columns:
        unip_map = alias_to_id[
            alias_to_id["alias"].isin(targets["uniprot_id"].dropna().astype(str))
        ].copy()
        unip_map = unip_map.rename(columns={"alias": "uniprot_id"})
        unip_map = unip_map.drop_duplicates(subset=["uniprot_id"])
    else:
        unip_map = pd.DataFrame(columns=["uniprot_id", "string_protein_id"])

    tmp = targets.merge(sym_map, on="targetSymbol", how="left")
    if not unip_map.empty:
        tmp = tmp.merge(
            unip_map, on="uniprot_id", how="left", suffixes=("", "_from_uniprot")
        )
        # Prefer mapping via UniProt if present, otherwise via symbol
        tmp["string_protein_id"] = tmp["string_protein_id_from_uniprot"].combine_first(
            tmp["string_protein_id"]
        )
        tmp = tmp.drop(columns=["string_protein_id_from_uniprot"])

    targets["string_protein_id"] = tmp["string_protein_id"]

    # --- compute degree from links file ------------------------------------
    print("[string] Loading links and computing degrees")
    # STRING default is space-separated; fall back to tab if needed.
    try:
        links = pd.read_csv(links_path, sep=" ")
        if "combined_score" not in links.columns:
            raise ValueError("Bad separator")
    except Exception:
        links = pd.read_csv(links_path, sep="\t")

    if not {"protein1", "protein2", "combined_score"} <= set(links.columns):
        raise RuntimeError(f"Unexpected columns in links file: {links.columns}")

    links_filt = links[links["combined_score"] >= score_threshold]

    all_proteins = pd.concat(
        [links_filt["protein1"], links_filt["protein2"]],
        ignore_index=True,
    )
    degree = (
        all_proteins.value_counts()
        .rename_axis("string_protein_id")
        .reset_index(name="ppiDegree")
    )

    targets = targets.merge(degree, on="string_protein_id", how="left")
    targets["ppiDegree"] = targets["ppiDegree"].fillna(0).astype(int)
    return targets


# --- main ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build target-level annotations: paralogs, PDB coverage, STRING PPI."
    )
    parser.add_argument(
        "--input-targets", required=True, help="CSV with at least targetId,targetSymbol."
    )
    parser.add_argument("--output", required=True, help="Output CSV for annotations.")
    parser.add_argument(
        "--workdir", required=True, help="Directory to store STRING download files."
    )
    parser.add_argument(
        "--score-threshold",
        type=int,
        default=700,
        help="STRING combined_score threshold (default: 700 = high confidence).",
    )
    args = parser.parse_args()

    targets = load_unique_targets(args.input_targets)
    targets = add_paralog_counts(targets)
    targets = add_uniprot_ids(targets)
    targets = add_pdb_flags(targets)
    targets = add_string_ppi(targets, args.workdir, score_threshold=args.score_threshold)

    out_cols = [
        "targetId",
        "targetSymbol",
        "uniprot_id",
        "hasStructure",
        "hasPDB",
        "string_protein_id",
        "ppiDegree",
        "paralogCount",
    ]
    out_cols = [c for c in out_cols if c in targets.columns]

    targets[out_cols].to_csv(args.output, index=False)
    print(f"[done] Wrote {len(targets)} rows to {args.output}")


if __name__ == "__main__":
    main()

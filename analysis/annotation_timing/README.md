# Annotation-timing sensitivity: 22.02-native rebuild

Backing artifacts for the manuscript's annotation-timing sensitivity results
(Section "Cold-Start and Feature Analyses", Limitations): the temporal
experiment's clinical labels and auxiliary score come from OTP Release 22.02,
while annotation text comes from a shared later snapshot. These runs rebuild
the annotation text from Release 22.02 itself, eliminating post-2022
annotation content by construction, and measure the effect.

## Construction

- `build_2202_features.py` — rebuilds `disease_text` / `target_text` from raw
  22.02 tables (`diseases`, `targets`, `diseaseToPhenotype`), porting the
  exact text-construction recipe of `0-OT-PreProcess_Recc.ipynb`.
- `test_build_2202_features.py` — port-fidelity gate: the same functions, fed
  the newer raw snapshot, must reproduce the released
  `copy_proc/{disease_df,target_df}.parquet` text columns **exactly**
  (1.0000 match on 38,959 diseases / 17,065 targets). This test also pins the
  inclusion of the `constraint` field, which the released artifact contains
  although the shipped notebook line comments it out.
- `temporal_2202_features.py` — seed-parametrized temporal retrain (identical
  protocol to `run_temporal_repeated.py`) with the 22.02-native text.
  Seeds 42, 43, 44, 45.
- `missingness_audit.py`, `final_s9_numbers.py` — coverage-stratified
  aggregation; `final_s9_numbers.log` holds the printed results.

## Result

Release 22.02's ontology contains 18,468 of the evaluation cohort's 38,959
disease terms; 58% of test pairs (235,076; 8,844 positives) have 2022-era
disease text ("covered"). On the covered pairs:

| Configuration | ROC-AUC | PR-AUC |
|---|---|---|
| Stored reference run (shared later snapshot) | 0.8540 | 0.2094 |
| Clinical-precedence tokens stripped | 0.8451 | 0.1989 |
| Fully 22.02-native text (4 seeds) | 0.8470 ± 0.0095 | 0.1994 ± 0.0130 |

Per-seed covered-subset ROC deltas (reference − native): 0.0021, 0.0213,
0.0022, 0.0024. Pooled numbers for the native runs (0.819 ± 0.016) are
depressed by the 42% of pairs whose disease terms have no 2022 text (blank
input), which measures entity coverage, not information leakage; per-subset
values incl. annotation-free baselines are in `s9_table_numbers.csv` and
`s9_baselines_per_subset.csv`.

## Reproduction

Raw 22.02 tables: `https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/22.02/output/etl/parquet/`
(tables `diseases`, `targets`, `diseaseToPhenotype`; `known_drug` and
`association_overall_direct` as used by the main temporal pipeline).

```
python test_build_2202_features.py    # port-fidelity gate (requires newer raw snapshot)
python build_2202_features.py         # writes disease_df_2202 / target_df_2202
python temporal_2202_features.py 42   # one retrain per seed: 42 43 44 45
python missingness_audit.py
python final_s9_numbers.py
```

# Quick example

Inspect the released predictions without installing the model:

```bash
pip install pandas
python examples/query_predictions.py --disease obesity
```

Prints the highest-scoring predicted targets for each matching disease.

```bash
python examples/query_predictions.py --target GIPR
```

Prints the diseases where GIPR is a predicted candidate.

Both commands query `Outputs/S1-DL_novel_predictions.csv`, the novel
disease–target predictions released with the manuscript (Open Targets 25.06).
Add `--include-known` to query `Outputs/S2-DL_novel+known_candidates.csv`,
which also contains the known clinical associations.

For interactive use, see the [web demo](https://huggingface.co/spaces/GrimSqueaker/OTRec).

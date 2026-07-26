#!/bin/bash
cd "$(dirname "$(readlink -f "$0")")/../../.."   # workspace root (parent of OTRec)
for S in 43 44 45; do
  for R in R1 R4; do
    echo "########## START $R seed$S $(date +%H:%M:%S) ##########"
    .venv/bin/python OTRec/rebuttal/scripts/ablation_temporal.py --rung $R --seed $S \
      --out ablation_seed_replication.csv 2>&1 \
      | grep -vE "oneDNN|cpu_feature|To enable|absl::|TF-TRT|external/local_xla"
    echo "########## DONE $R seed$S $(date +%H:%M:%S) ##########"
  done
done
echo "SEED_REPLICATION_COMPLETE"

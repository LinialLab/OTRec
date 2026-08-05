#!/bin/bash
cd "$(dirname "$(readlink -f "$0")")/../../.."   # workspace root (parent of OTRec)
for R in R1 R2 R3 R4 R5; do
  echo "########## START $R $(date +%H:%M:%S) ##########"
  .venv/bin/python OTRec/rebuttal/scripts/ablation_temporal.py --rung $R 2>&1 | grep -vE "oneDNN|cpu_feature|To enable|absl::|TF-TRT|external/local_xla"
  echo "########## DONE $R $(date +%H:%M:%S) ##########"
done
echo "ALL_RUNGS_COMPLETE"

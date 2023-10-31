#!/bin/env bash

# execute in directory that contains all available alignments:

for file in "CALM1_HUMAN_1_b0.5" "MTH3_HAEAESTABILIZED_1_b0.5" "BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105" "RL401_YEAST_1_b0.5" "BRCA1_HUMAN_1_b0.5" "BRCA1_HUMAN_BRCT_1_b0.3" "TRPC_THEMA_1_b0.5" "parEparD_3"; do
    if [[ -f "$file.a2m" ]]; then
        hmmbuild "$file.hmm" "$file.a2m"
    fi
done
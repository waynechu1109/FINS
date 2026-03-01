#!/bin/bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# dtu_90_41 dtu_110_43 dtu_65_47 dtu_73_42 dtu_82_14 dtu_114_32 dtu shoes dtu_73_42 dtu_108_32 dtu_118_60

for file in dtu_65_47 ; do
    for lr in 3.2e-3; do
        echo -e "\033[34m$file: TEST_${lr}\033[0m"
        ./scripts/experiment.sh "TEST_${lr}" "$lr" "vggt" "$file" schedule false
    done
done
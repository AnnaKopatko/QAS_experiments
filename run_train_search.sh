#!/bin/bash

set -e

echo "=== Batch 1: No Controller (3 runs each) ==="

COMMON_ARGS="--epochs 400 \
             --warmup_epochs 200 \
             --n_experts 5 \
             --n_search 500 \
             --ea_pop_size 25 \
             --ea_gens 20 \
             --searcher evolution \
             --finetune_epochs 150 \
             --device default \
             --noise \
             --log_experiment
             --use_aging"

MOLS=("H2" "LiH" "BeH2")
LAYERS=(4 8 8)

# 3 repetitions
for RUN in 1 2 3; do
    echo "=== Repetition $RUN ==="

    for i in ${!MOLS[@]}; do
        MOL=${MOLS[$i]}
        N_LAYERS=${LAYERS[$i]}

        echo "Running $MOL (layers=$N_LAYERS), run $RUN..."

        python train_search.py \
            $COMMON_ARGS \
            --mol_name $MOL \
            --n_layers $N_LAYERS \
            --seed $RUN \
            --expr_tag two_gates_RyRz_aging_run_$RUN

    done
done

echo "=== Batch 1 Finished ==="
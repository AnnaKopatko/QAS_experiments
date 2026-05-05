#!/bin/bash

set -e

COMMON_ARGS="--n_experts 5 \
             --n_search 500 \
             --ea_pop_size 50 \
             --ea_gens 20 \
             --searcher evolution \
             --finetune_epochs 150 \
             --device default \
             --log_experiment"

MOLS=("LiH")
LAYERS=(4)
EPOCHS=(400)

# 3 repetitions
for RUN in 1 2 3; do
    echo "=== Repetition $RUN ==="

    for i in ${!MOLS[@]}; do
        MOL=${MOLS[$i]}
        N_LAYERS=${LAYERS[$i]}
        N_EPOCHS=${EPOCHS[$i]}

        echo "Running $MOL (layers=$N_LAYERS, epochs=$N_EPOCHS), run $RUN..."

        NEXT_RUN=$((RUN))

        python train_search.py \
            $COMMON_ARGS \
            --mol_name "$MOL" \
            --n_layers "$N_LAYERS" \
            --epochs "$N_EPOCHS" \
            --seed "$RUN" \
            --expr_tag "2cnot3rot_dropout_${NEXT_RUN}"

    done
done

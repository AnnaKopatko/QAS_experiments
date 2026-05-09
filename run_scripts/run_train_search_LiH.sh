# #!/bin/bash

# set -e

# COMMON_ARGS="--n_experts 5 \
#              --n_search 500 \
#              --ea_pop_size 100 \
#              --ea_gens 40 \
#              --searcher evolution \
#              --finetune_epochs 150 \
#              --warmup_epochs 400
#              --device default \
#              --log_experiment"

# MOLS=("H20")
# LAYERS=(4)
# EPOCHS=(600)

# # 3 repetitions
# for RUN in 1 2 3; do
#     echo "=== Repetition $RUN ==="

#     for i in ${!MOLS[@]}; do
#         MOL=${MOLS[$i]}
#         N_LAYERS=${LAYERS[$i]}
#         N_EPOCHS=${EPOCHS[$i]}

#         echo "Running $MOL (layers=$N_LAYERS, epochs=$N_EPOCHS), run $RUN..."

#         NEXT_RUN=$((RUN))

#         python train_search.py \
#             $COMMON_ARGS \
#             --mol_name "$MOL" \
#             --n_layers "$N_LAYERS" \
#             --epochs "$N_EPOCHS" \
#             --seed "$RUN" \
#             --cnot_dropout_prob 0 \
#             --r_dropout_prob 0 \
#             --mutation_prob 0.25 \
#             --use_controller false \
#             --block_structure \
#             --expr_tag "block_structure_${NEXT_RUN}"

#     done
# done

#!/bin/bash

set -e

COMMON_ARGS="--n_experts 5 \
             --n_search 500 \
             --ea_pop_size 50 \
             --ea_gens 2 \
             --searcher evolution \
             --finetune_epochs 150 \
             --warmup_epochs 200
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
            --cnot_dropout_prob 0 \
            --r_dropout_prob 0 \
            --mutation_prob 0.25 \
            --use_controller false \
            --block_structure \
            --expr_tag "block_structure_${NEXT_RUN}"

    done
done
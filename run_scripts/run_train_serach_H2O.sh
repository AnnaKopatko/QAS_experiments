
set -e

COMMON_ARGS="--n_experts 5 \
             --n_search 500 \
             --ea_pop_size 100 \
             --ea_gens 40 \
             --searcher evolution \
             --finetune_epochs 150 \
             --warmup_epochs 400
             --device default \
             --log_experiment"

MOLS=("H2O")
LAYERS=(4)
EPOCHS=(600)

# 3 repetitions
for RUN in 1; do
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
            --cnot_dropout_prob 0.125 \
            --r_dropout_prob 0.125 \
            --mutation_prob 0.25 \
            --use_controller true \
            --expr_tag "2cnot3rot_dropout_RC_controller_1"

    done
done
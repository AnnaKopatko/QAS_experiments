#!/bin/bash

set -e

COMMON_ARGS="--epochs 150 \
             --lr 0.2 \
             --device default \
             --log_experiment"

MOL="LiH"
ARCH="single_double"
TAG="classic_vqe_run"

for RUN in 1 2 3; do
    echo "=== Repetition $RUN ==="
    echo "Running $MOL (architecture=$ARCH, tag=${TAG}_${RUN})..."

    python train.py \
        $COMMON_ARGS \
        --mol_name "$MOL" \
        --architecture "$ARCH" \
        --seed "$RUN" \
        --expr_tag "${TAG}_${ARCH}_${RUN}"

done

#!/usr/bin/env bash
set -euo pipefail

python3 train_search.py --mol_name H2
python3 train_search.py --mol_name LiH
python3 train_search.py --mol_name BeH2

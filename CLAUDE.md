# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**QAS_modern** is a Quantum Architecture Search (QAS) framework for discovering optimal quantum circuit architectures for Variational Quantum Eigensolver (VQE) applications. It uses evolutionary algorithms to search the space of quantum circuit designs while training neural parameters to minimize molecular ground-state energy (H2, LiH, BeH2).

## Running the Code

### Architecture Search (primary workflow)
```bash
python train_search.py \
  --epochs 400 \
  --warmup_epochs 200 \
  --n_layers 8 \
  --n_experts 5 \
  --n_search 500 \
  --ea_pop_size 25 \
  --ea_gens 20 \
  --mutation_prob 0.25 \
  --searcher evolution \
  --mol_name LiH \
  --device default \
  --noise \
  --seed 0
```

Key arguments:
- `--mol_name`: `H2`, `LiH`, `BeH2`
- `--searcher`: `evolution` or `random`
- `--device`: `default` (PennyLane), `ibmq-sim`, `ibmq`
- `--use_controller`: enforce CNOT adjacency constraints
- `--use_aging`: apply aging mechanism in evolution

### Fixed Architecture Training
```bash
python train.py --mol_name LiH --architecture single_double --epochs 150 --lr 0.2
```
`--architecture`: `single_double` or `uccsd`

### Batch Runs
```bash
bash run_train_search.sh
```
Runs multiple experiments over H2, LiH, BeH2 with varying layer counts.

### Minimal Demo
```bash
python simple_vqe.py
```

## Architecture

```
train_search.py / train.py          # Entry points
│
├── models/
│   ├── search_space.py             # Defines all valid gate/CNOT combos (Cartesian product)
│   ├── circuit_search_model.py     # Multi-expert parameter storage and training
│   └── circuit_model.py            # Fixed-architecture circuit trainer
│
├── custom_evolution/
│   ├── custom_evoltuion.py         # Evolutionary sampler (selection, crossover, mutation)
│   ├── architecture_controller.py  # Constraint: no repeated CNOT in adjacent layers
│   └── utils.py                    # Evolution utilities
│
├── utils/
│   ├── molecule_utils.py           # Hamiltonian loading, Hartree-Fock state
│   ├── circut_utils.py             # Circuit builders (UCCSD, Single/Double excitations)
│   ├── vis_and_logging.py          # Plots and artifact saving
│   └── utils.py                    # Architecture evaluation, expert selection
│
└── configs/config.yaml             # Molecule coords/energies, noise params, search space gates
```

### Search Workflow
1. **Warmup** (`warmup_epochs`): train parameters on random architectures to initialize experts
2. **Search** (`n_search` iterations): evolutionary algorithm discovers good architectures
   - Evaluate fitness using best-matching expert parameters
   - Binary tournament selection → two-point crossover → polynomial mutation
3. **Finetuning** (`finetune_epochs`): train the best found architecture from scratch

### Key Concepts

**Architecture encoding**: A list of integers like `[5, 12, 3]` — each integer indexes into the search space, selecting a rotation gate config + CNOT pattern for that layer.

**Expert system**: Multiple independent parameter sets (one per rotation-gate configuration). During search, `expert_evaluator` picks the best expert for each candidate architecture, enabling efficient exploration without re-training from scratch.

**Search space**: Defined in `configs/config.yaml` and `models/search_space.py`. It is the Cartesian product of {rotation gate types} × {CNOT wire pairs}. Rotation gates default to `[RY, RZ]`; CNOT pairs are molecule-dependent.

**Constraint system** (`ArchitectureController`): If layer `n` uses CNOT `(i,j)`, layer `n+1` must not reuse `(i,j)`. Activated with `--use_controller`.

## Configuration

`configs/config.yaml` controls:
- Molecule definitions (geometry, basis set, reference energies for H2/LiH/BeH2)
- Noise model parameters (single-qubit 5%, two-qubit 20% depolarizing)
- Search space: valid rotation gates and CNOT wire pairs

## Experiment Outputs

Each run creates a timestamped directory under `experiments/`:
- `args.txt` — full CLI arguments as JSON
- `records.json` — energy curves and final result
- `energy_convergence.png` — training convergence plot
- `circuit_*.png` — circuit diagrams (high-level and decomposed)
- `circuit_gate_info.json` — gate count breakdown

Experiment tracking uses **Aim** (`.aim/` directory). View runs with:
```bash
aim up
```

## Dependencies

No `requirements.txt` exists. Key packages:
- `pennylane` — quantum circuit framework
- `qiskit`, `qiskit-aer` — simulator backends
- `aim` — experiment tracking
- `pyyaml`, `numpy`, `matplotlib`

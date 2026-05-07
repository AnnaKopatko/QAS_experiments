# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**QAS_modern** is a Quantum Architecture Search (QAS) framework for discovering optimal quantum circuit architectures for Variational Quantum Eigensolver (VQE) applications. It uses evolutionary algorithms to search the space of quantum circuit designs while training variational parameters to minimize molecular ground-state energy (H2, LiH, BeH2, H2O).

## Running the Code

### Architecture Search (primary workflow)
```bash
python train_search.py \
  --epochs 400 \
  --warmup_epochs 200 \
  --n_layers 4 \
  --n_experts 5 \
  --n_search 500 \
  --ea_pop_size 50 \
  --ea_gens 20 \
  --mutation_prob 0.125 \
  --searcher evolution \
  --mol_name LiH \
  --device default \
  --seed 0
```

Key arguments:
- `--mol_name`: `H2`, `LiH`, `BeH2`, `H2O`
- `--searcher`: `evolution` or `random`
- `--device`: `default` (PennyLane), `ibmq-sim`, `ibmq`
- `--noise`: enable depolarizing noise model (on by default)
- `--use_controller`: enforce no-repeated-CNOT constraint in adjacent layers (on by default; pass `False` to disable)
- `--use_aging`: apply aging mechanism (AmoebaNet-style) in evolution
- `--block_structure`: use Du et al. block search space (all qubits get gates, all neighbor CNOTs independent)
- `--sandwich_layers`: split rotations around CNOTs (first-half Rs → CNOTs → second-half Rs)
- `--cnot_dropout_prob`: per-layer probability of dropping one CNOT during mutation (default 0.125)
- `--r_dropout_prob`: per-layer probability of dropping one R gate during mutation (default 0.125)
- `--lr`: Adam optimizer learning rate (default 0.2)
- `--expr_tag`: string tag added to Aim run name for experiment grouping
- `--finetune_epochs`: epochs for fine-tuning best architecture from scratch (default 150)

### Fixed Architecture Training
```bash
python train.py --mol_name LiH --architecture single_double --epochs 150 --lr 0.2
```
`--architecture`: `single_double` or `uccsd`

### Batch Runs
```bash
bash run_train_search.sh   # runs evolution search experiments (currently LiH, block_structure)
bash run_block_experiments.sh
```

### Minimal Demo
```bash
python simple_vqe.py
```

## Architecture

```
train_search.py / train.py          # Entry points
│
├── models/
│   ├── search_space.py             # SearchSpace and BlockSearchSpace (Cartesian product + dynamic extensions)
│   ├── circuit_search_model.py     # Multi-expert parameter storage and training
│   └── circuit_model.py            # Fixed-architecture circuit trainer
│
├── custom_evolution/               # Active evolutionary engine (replaces evolution/)
│   ├── custom_evoltuion.py         # EvolutionSampler: selection, crossover, mutation, dropout
│   ├── architecture_controller.py  # Constraint: no repeated CNOT in adjacent layers
│   └── utils.py                    # arch_to_key helper
│
├── evolution/                      # Legacy NSGA-Net sampler (unused; kept for reference)
│   ├── evolution_sampler.py
│   └── nsganet.py
│
├── utils/
│   ├── molecule_utils.py           # Hamiltonian loading, Hartree-Fock state
│   ├── circut_utils.py             # Circuit builders (UCCSD, Single/Double excitations)
│   ├── vis_and_logging.py          # Plots and artifact saving
│   └── utils.py                    # Architecture evaluation, expert selection
│
└── configs/config.yaml             # Molecule coords/energies, noise params, search space config
```

### Search Workflow
1. **Warmup** (`warmup_epochs`): train parameters on random architectures to initialize experts
2. **Search** (`n_search` generations × population): evolutionary algorithm discovers good architectures
   - Evaluate fitness using `expert_evaluator` (best-matching expert parameters)
   - Binary tournament selection → two-point crossover → polynomial mutation
   - Optional CNOT dropout and R-gate dropout mutations create dynamic arch variants
3. **Finetuning** (`finetune_epochs`): train the best found architecture from scratch with Adam; best-seen params tracked with patience

### Key Concepts

**Architecture encoding**: A list of integers like `[5, 12, 3]` — each integer indexes into the search space, selecting a rotation gate config + CNOT pattern for that layer.

**Search space types**:
- `SearchSpace` (default): Cartesian product of `n_rotations_per_layer` rotations (from a subset of qubits) × `n_cnots_per_layer` CNOT pairs. Currently `n_rotations_per_layer=3`, `n_cnots_per_layer=2`.
- `BlockSearchSpace` (`--block_structure`): Du et al. 2022 design — all N qubits get one gate (G^N rotation configs) × all 2^(N−1) CNOT subsets of nearest-neighbor pairs. Activated with `--block_structure`.

**Dynamic extensions**: CNOT dropout and R-gate dropout mutations register novel architectures at runtime into `search_space._dynamic_extensions` beyond the static search space. These are keyed by `(Rs, CNOTs)` tuples and reuse the parent's rotation parameter slot for a warm start.

**Expert system**: `n_experts` independent parameter sets, each of shape `(n_layers, len(Rs_space), max_rot)`. During search, `expert_evaluator` picks the best expert for each candidate architecture. Parameter sharing: architectures with the same rotation config share the same parameter slot (`r_idx = search_space.get_r_idx(idx)`).

**Constraint system** (`ArchitectureController`): If layer `n` uses CNOT `(i,j)`, layer `n+1` must not reuse `(i,j)`. Pre-computes valid follower sets for all blocks. Activated by default (`--use_controller True`); crossover and mutation operators are constraint-aware (repair after crossover, constrained sampling during mutation).

**Optimizer**: Adam (`qml.AdamOptimizer`) with configurable `--lr` (default 0.2). QNG options (`--qng_lam`, `--qng_approx`) are parsed but not used in the current training loop.

## Configuration

`configs/config.yaml` controls:
- Molecule definitions (geometry, basis set, reference energies for H2/LiH/BeH2/H2O)
- Noise model parameters (single-qubit 5%, two-qubit 20% depolarizing)
- Search space: `valid_Rs` (`RY`, `RZ`), `n_rotations_per_layer`, `n_cnots_per_layer`

## Experiment Outputs

`train_search.py` creates a timestamped directory under `experiments/`:
- `args.txt` — full CLI arguments as JSON
- `nas_result_sorted.txt` — all evaluated architectures sorted by energy
- `nas_result_with_energy.txt` — detailed results with deviation and expert index
- `best_architecture_decoded.txt` — human-readable layer-by-layer breakdown
- `best_finetuned_params.npy` — best fine-tuned parameters
- `energy_convergence.png` — fine-tuning convergence plot
- `final_circuit.png` — circuit diagram
- `circuit_specs.txt` / `circuit_gate_info.json` — gate count and depth

`train.py` saves under `experiments/classic_archs_new/`:
- `records.json`, `energy_convergence.png`, `architecture_highlevel.png`, `architecture_decomposed.png`, `circuit_gate_info.json`

Experiment tracking uses **Aim** (`.aim/` directory). View runs with:
```bash
aim up
```

Helper scripts `inspect_aim.py` and `visualize_aim_runs.py` are available for querying Aim runs programmatically.

## Dependencies

No `requirements.txt` exists. Key packages:
- `pennylane` — quantum circuit framework
- `qiskit`, `qiskit-aer` — simulator backends and noise models
- `aim` — experiment tracking
- `pyyaml`, `numpy`, `matplotlib`

# Evolution Algorithm Analysis — QAS_modern

## 1. Bugs and Issues Found

### Issue 1 — Constrained mutation doesn't repair downstream genes (`use_controller=True` only)

`constrained_mutation` in `architecture_controller.py:241-251` ensures each mutated gene is valid given its predecessor. But it does **not** repair subsequent *unmutated* genes.

Example: arch `[A, B, C, D]` where C is valid after B. If B is mutated to B', C stays unchanged and may now violate the constraint against B'. Fix: after mutating gene `j`, scan `j+1 … n_layers-1` and repair any gene that became invalid.

---

### Issue 2 — Redundant energy evaluation in `test_subnet_evolution` ✅ Fixed

`expert_evaluator` already ran `cost(model.params)` for the best expert (stored as `best_loss`). The original code then re-evaluated with an extra `cost()` call, wasting one circuit evaluation per architecture and introducing an additional noise realization.

**Fix applied:** `expert_evaluator` now returns `(best_idx, best_loss)`. All four call sites in `train_search.py` updated to unpack both values and use the returned energy directly.

---

### Issue 3 — Mutation probability

With `mutation_prob=0.25` and `n_layers=8`, the expected mutations per individual is `0.25 × 8 = 2`, effectively randomising most offspring. The standard recommendation is `1/n_layers`.

| n_layers | Recommended mutation_prob |
|---|---|
| 4 | 0.25 ✓ (correct as-is) |
| 8 | 0.125 |

**For n_layers=4, 0.25 is exactly right.**

---

### Issue 4 — Population size too small for large search spaces

With `pop_size=25` and large search spaces (LiH, BeH2, H2O), evolution degrades toward random sampling because no two individuals share genetic material by ancestry.

---

### Issue 5 — High quantum noise degrades selection pressure

With 20% 2-qubit depolarizing noise, energy estimates are noisy. Tournament selection may promote architectures that got a lucky noise sample rather than genuinely better ones.

---

### Issue 6 — `search_iter` never incremented in evolution mode (logging bug)

In `train_search.py`, `search_iter = 0` is set before the if/else branch. In the evolution path it is never incremented, so all Aim metrics are logged at step 0.

---

## 2. Search Space Sizes

| Molecule | Qubits | Rotation configs (rs_size) | n_blocks | Total space (4 layers) |
|---|---|---|---|---|
| H2 | 4 | 32 | 128 | 128⁴ ≈ 2.7×10⁸ |
| LiH | 6 | 160 | 960 | 960⁴ ≈ 8.5×10¹¹ |
| BeH2 | 8 | 448 | 3584 | 3584⁴ ≈ 1.6×10¹⁴ |
| H2O | 8 | 448 | 3584 | 3584⁴ ≈ 1.6×10¹⁴ |

Search space = `C(n_qubits, 3) × 2³ × n_qubits` with config: `n_rotations_per_layer=3`, `valid_Rs=[RY,RZ]`, `n_cnots_per_layer=1`.

---

## 3. Why Evolution Didn't Outperform Random Search

### Root cause: expert parameters are too sparsely trained

The expert model stores parameters indexed by **rotation config** (`r_idx = block_idx // n_cnots`), shared across all CNOT variants of the same rotation config. With `warmup_epochs=200` and 5 experts, each expert gets ~40 training epochs.

**Coverage per expert per layer (LiH, rs_size=160):**
- Unique rotation configs trained: ~35 / 160 = **22%**
- Untrained (random init): **78%**

Architectures whose rotation configs were never trained return essentially random energies. Tournament selection then picks winners based on noise rather than architecture quality — evolution behaves like random sampling regardless of population size.

**This is why increasing `pop_size` from 25 → 50 alone didn't help.** More individuals sampled from the same flat/noisy landscape doesn't fix the signal quality.

---

## 4. Warmup Epochs — Original Estimate (90% coverage)

Targeting 90% coverage of rotation configs per expert per layer:

```
warmup_needed = n_experts × log(0.1) / log(1 - 1/rs_size)
```

| Molecule | rs_size | Warmup (90%) |
|---|---|---|
| H2 | 32 | 362 |
| LiH | 160 | 1836 |
| BeH2 | 448 | 5152 |
| H2O | 448 | 5152 |

---

## 5. Warmup Epochs — Revised Estimate (weight-sharing argument)

With weight sharing, tournament selection acts as a self-filter: it pushes the population toward architectures with trained rotation configs over the first few generations. You don't need full coverage upfront — you need enough warmup so that the initial population contains at least **~5 seed individuals** where all `n_layers` positions land on trained rotation configs.

**Required coverage:**
```
c = (5 / pop_size)^(1 / n_layers)
warmup_needed = n_experts × log(1 - c) / log(1 - 1/rs_size)
```

| Molecule | pop_size | Coverage target | New warmup | Old warmup | Reduction |
|---|---|---|---|---|---|
| H2 | 25 | 67% | **173** | 362 | 2.1× |
| LiH | 50 | 56% | **658** | 1836 | 2.8× |
| BeH2 | 100 | 47% | **1432** | 5152 | 3.6× |
| H2O | 100 | 47% | **1432** | 5152 | 3.6× |

These are approximate lower bounds. In practice you may need slightly more if partially-trained configs introduce too much noise. If running time allows, targeting coverage between the new and old estimates is safe.

---

## 6. Recommended Run Parameters

### H2
```bash
python train_search.py \
  --mol_name H2 --n_layers 4 \
  --warmup_epochs 173 --epochs 200 \
  --ea_pop_size 25 --ea_gens 20 \
  --mutation_prob 0.25 --searcher evolution
```

### LiH
```bash
python train_search.py \
  --mol_name LiH --n_layers 4 \
  --warmup_epochs 658 --epochs 700 \
  --ea_pop_size 50 --ea_gens 20 \
  --mutation_prob 0.25 --searcher evolution
```

### BeH2 / H2O
```bash
python train_search.py \
  --mol_name BeH2 --n_layers 4 \
  --warmup_epochs 1432 --epochs 1500 \
  --ea_pop_size 100 --ea_gens 20 \
  --mutation_prob 0.25 --searcher evolution
```

---

## 7. Hardware Acceleration Notes

### GPU (lightning.gpu / qiskit-aer-gpu)

For **non-noisy runs** (`--device default`), the PennyLane device can be swapped:
```python
# train_search.py line 204 / train.py line 156
dev = qml.device("lightning.qubit", wires=qubits)  # fast C++ CPU
dev = qml.device("lightning.gpu", wires=qubits)    # NVIDIA CUDA (requires pennylane-lightning[gpu])
```

For **noisy runs** (`--noise`), the code uses `qiskit.aer`. The GPU equivalent is:
```bash
pip install qiskit-aer-gpu
```
No code change needed — it replaces `qiskit-aer` automatically.

### Why GPU won't help at current molecule sizes

Your qubit counts are small (H2=4, LiH=6, BeH2=8). GPU acceleration for statevector/density matrix simulation only pays off above ~20 qubits. Below that, the overhead of launching CUDA kernels per circuit evaluation costs more than it saves — especially in the evolutionary search, which runs hundreds of short evaluations (`n_search=500`).

### Practical speedup options for current scale

1. **Parallelize circuit evaluations across CPU cores** — each candidate in the population is independent and can be evaluated in parallel using `multiprocessing` or `concurrent.futures`.
2. **Reduce `n_search` / `ea_pop_size`** during experimentation runs.
3. GPU becomes worthwhile if scaling to molecules with 20+ qubits.

---

## 8. Key Formulas

```python
import math

def warmup_epochs_needed(n_qubits, pop_size, n_layers=4, n_experts=5, seeds=5):
    rs_size = math.comb(n_qubits, 3) * (2 ** 3)
    c_target = (seeds / pop_size) ** (1.0 / n_layers)
    k_per_expert = math.log(1 - c_target) / math.log(1 - 1/rs_size)
    return int(k_per_expert * n_experts)

# Examples
print(warmup_epochs_needed(n_qubits=4,  pop_size=25))   # H2   → 173
print(warmup_epochs_needed(n_qubits=6,  pop_size=50))   # LiH  → 658
print(warmup_epochs_needed(n_qubits=8,  pop_size=100))  # H2O  → 1432
```

"""
custom_evolution.py  (updated — constrained architecture search)
================================================================
Identical to the original EvolutionSampler except that every operator
(initialisation, crossover, mutation) goes through an ArchitectureController
that enforces:

    Constraint: if layer n uses CNOT (i,j), layer n+1 must NOT use CNOT (i,j).

Pass ``architecture_controller=None`` to revert to the unconstrained behaviour.
"""

from __future__ import annotations

import numpy as np
import time

from .utils import arch_to_key

# Import the controller — adjust the import path to match your project layout
from .architecture_controller import ArchitectureController


# =============================================================================
# Step 1: Population Initialization
# =============================================================================

def initialize_population(
    pop_size: int,
    n_layers: int,
    n_blocks: int,
    controller: ArchitectureController | None = None,
) -> np.ndarray:
    """Generate an initial random population, obeying constraints if provided."""
    if controller is not None:
        return np.array([controller.random_architecture(n_layers) for _ in range(pop_size)])
    return np.random.randint(0, n_blocks, size=(pop_size, n_layers))


# =============================================================================
# Step 2: Evaluation  (unchanged)
# =============================================================================

def evaluate_population(
    population: np.ndarray,
    eval_func,
    cache: dict,
) -> np.ndarray:
    scores = np.full(len(population), np.nan)
    for i, arch in enumerate(population):
        key = arch_to_key(arch)
        if key in cache:
            scores[i] = cache[key]
        else:
            score = eval_func(arch)
            cache[key] = score
            scores[i] = score
        print(f"==evaluation subnet:{key} score:{scores[i]:.6f}")
    return scores


# =============================================================================
# Step 3: Binary Tournament Selection  (unchanged)
# =============================================================================

def binary_tournament_selection(
    population: np.ndarray,
    scores: np.ndarray,
    n_select: int,
) -> np.ndarray:
    pop_size = len(population)
    selected = []
    for _ in range(n_select):
        a, b = np.random.choice(pop_size, size=2, replace=False)
        if scores[a] < scores[b]:
            winner = a
        elif scores[b] < scores[a]:
            winner = b
        else:
            winner = np.random.choice([a, b])
        selected.append(population[winner].copy())
    return np.array(selected)


# =============================================================================
# Step 4: Two-Point Crossover  (now controller-aware)
# =============================================================================

def two_point_crossover(
    parents: np.ndarray,
    n_layers: int,
    controller: ArchitectureController | None = None,
) -> np.ndarray:
    """Two-point crossover; offspring are repaired if a controller is given."""
    if controller is not None:
        return controller.constrained_crossover_and_repair(parents, n_layers)

    # --- original unconstrained path ---
    n_parents = len(parents)
    offspring = np.empty_like(parents)
    for i in range(0, n_parents - 1, 2):
        p1, p2 = parents[i].copy(), parents[i + 1].copy()
        cut1, cut2 = sorted(np.random.choice(n_layers + 1, size=2, replace=False))
        offspring[i]     = np.concatenate([p1[:cut1], p2[cut1:cut2], p1[cut2:]])
        offspring[i + 1] = np.concatenate([p2[:cut1], p1[cut1:cut2], p2[cut2:]])
    if n_parents % 2 == 1:
        offspring[-1] = parents[-1].copy()
    return offspring


# =============================================================================
# Step 5: Polynomial Mutation  (now controller-aware)
# =============================================================================

def polynomial_mutation(
    offspring: np.ndarray,
    n_blocks: int,
    mutation_prob: float | None = None,
    controller: ArchitectureController | None = None,
) -> np.ndarray:
    """Mutate offspring; uses constrained mutation if controller is provided."""
    if controller is not None:
        return controller.constrained_mutation(offspring, mutation_prob)

    # --- original unconstrained path ---
    mutated = offspring.copy()
    n_individuals, n_layers = mutated.shape
    if mutation_prob is None:
        mutation_prob = 1.0 / n_layers
    for i in range(n_individuals):
        for j in range(n_layers):
            if np.random.rand() < mutation_prob:
                mutated[i, j] = np.random.randint(0, n_blocks)
    return mutated


# =============================================================================
# Step 5b: CNOT Dropout Mutation
# =============================================================================

def cnot_dropout_mutation(
    offspring: np.ndarray,
    search_space,
    dropout_prob: float,
) -> np.ndarray:
    """For each layer, with probability dropout_prob, remove one random CNOT.

    The dropped architecture is registered in search_space._dynamic_extensions
    so it can be evaluated and fine-tuned like any other architecture.
    Rotation parameters are inherited from the parent's r_idx (no new param slots).
    """
    mutated = offspring.copy()
    n_individuals, n_layers = mutated.shape
    for i in range(n_individuals):
        for j in range(n_layers):
            if np.random.rand() >= dropout_prob:
                continue
            idx = int(mutated[i, j])
            Rs, CNOTs = search_space.get_arch_elem(idx)
            if len(CNOTs) <= 1:
                continue  # need at least 2 CNOTs to drop one
            drop_k = np.random.randint(len(CNOTs))
            CNOTs_reduced = tuple(c for k, c in enumerate(CNOTs) if k != drop_k)
            parent_r_idx = search_space.get_r_idx(idx)
            new_idx = search_space.register_cnot_dropout_arch(Rs, CNOTs_reduced, parent_r_idx)
            mutated[i, j] = new_idx
    return mutated


# =============================================================================
# Step 5c: R-Gate Dropout Mutation
# =============================================================================

def r_dropout_mutation(
    offspring: np.ndarray,
    search_space,
    dropout_prob: float,
) -> np.ndarray:
    """For each layer, with probability dropout_prob, remove one random R gate.

    Requires at least 2 R gates in the layer so the layer isn't left empty.
    The dropped variant is registered in search_space._dynamic_extensions
    reusing the parent's r_idx as a warm-start parameter slot.
    """
    mutated = offspring.copy()
    n_individuals, n_layers = mutated.shape
    for i in range(n_individuals):
        for j in range(n_layers):
            if np.random.rand() >= dropout_prob:
                continue
            idx = int(mutated[i, j])
            Rs, CNOTs = search_space.get_arch_elem(idx)
            if len(Rs) <= 1:
                continue  # need at least 2 R gates to drop one
            drop_k = np.random.randint(len(Rs))
            Rs_reduced = [r for k, r in enumerate(Rs) if k != drop_k]
            parent_r_idx = search_space.get_r_idx(idx)
            new_idx = search_space.register_r_dropout_arch(Rs_reduced, CNOTs, parent_r_idx)
            mutated[i, j] = new_idx
    return mutated


# =============================================================================
# Step 6: Duplicate Elimination  (unchanged)
# =============================================================================

def eliminate_duplicates(
    offspring: np.ndarray,
    population: np.ndarray,
    n_blocks: int,
    controller: ArchitectureController | None = None,
) -> np.ndarray:
    n_layers = offspring.shape[1]
    existing_keys = {arch_to_key(arch) for arch in population}
    cleaned = []
    for arch in offspring:
        key = arch_to_key(arch)
        if key not in existing_keys:
            cleaned.append(arch)
            existing_keys.add(key)

    while len(cleaned) < len(offspring):
        if controller is not None:
            new_arch = controller.random_architecture(n_layers)
        else:
            new_arch = np.random.randint(0, n_blocks, size=n_layers)
        key = arch_to_key(new_arch)
        if key not in existing_keys:
            cleaned.append(new_arch)
            existing_keys.add(key)

    return np.array(cleaned[:len(offspring)])


# =============================================================================
# Step 7: Survival Selection  (unchanged)
# =============================================================================

def survival_selection(
    population: np.ndarray,
    scores: np.ndarray,
    pop_size: int,
    generation_added: np.ndarray | None = None,  # NEW
    aging: bool = False,                          # NEW
) -> tuple:
    if aging and generation_added is not None:
        # Remove oldest: sort by age (ascending = oldest first), keep youngest pop_size
        oldest_first = np.argsort(generation_added)          # oldest at index 0
        survivor_indices = oldest_first[len(population) - pop_size:]  # keep the youngest
    else:
        # Original: remove worst performers
        random_keys = np.random.rand(len(scores))
        sorted_indices = np.lexsort((random_keys, scores))
        survivor_indices = sorted_indices[:pop_size]

    return population[survivor_indices], scores[survivor_indices], \
           generation_added[survivor_indices] if generation_added is not None else None


# =============================================================================
# Main Evolutionary Loop
# =============================================================================

class EvolutionSampler:
    """
    Evolutionary architecture sampler with optional constraint enforcement.

    Parameters
    ----------
    search_space : SearchSpace, optional
        If provided, an :class:`ArchitectureController` is built automatically
        and all operators become constraint-aware.  Pass ``None`` (default)
        for the original unconstrained behaviour.
    """

    def __init__(
        self,
        pop_size: int = 50,
        n_gens: int = 20,
        n_layers: int = 3,
        n_blocks: int = 12,
        aim_run=None,
        search_space=None,
        use_controller=False,
        mutation_prob=None,
        aging: bool = True,
        cnot_dropout_prob: float = 0.0,
        r_dropout_prob: float = 0.0,
    ):
        self.aging = aging
        self.pop_size  = pop_size
        self.n_gens    = n_gens
        self.n_layers  = n_layers
        self.n_blocks  = n_blocks
        self.aim_run   = aim_run
        self.mutation_prob = mutation_prob
        self.cnot_dropout_prob = cnot_dropout_prob
        self.r_dropout_prob = r_dropout_prob
        self.search_space_obj = search_space  # kept for dropout mutations

        if search_space is not None and use_controller:
            self.controller = ArchitectureController(search_space)
            print(self.controller.constraint_summary())
        else:
            self.controller = None

        self.subnet_eval_dict: dict = {}
        self.subnet_topk: list = []

    # ------------------------------------------------------------------

    def sample(self, eval_func) -> list:
        cache = self.subnet_eval_dict
        ctrl  = self.controller        # shorthand

        evolution_start = time.perf_counter()

        # STEP 1 — initialise population (constrained if controller present)
        population = initialize_population(
            self.pop_size, self.n_layers, self.n_blocks, controller=ctrl
        )
        print(f"Initialized population: {self.pop_size} individuals, "
              f"{self.n_layers} layers, {self.n_blocks} blocks"
              + (" [CONSTRAINED]" if ctrl else ""))
        
        # After initializing population (Step 1):
        generation_added = np.zeros(self.pop_size, dtype=int)  # all born at gen 0


        # STEP 2 — evaluate initial population
        scores = evaluate_population(population, eval_func, cache)

        # MAIN LOOP
        for gen in range(1, self.n_gens + 1):
            print(f"\n==Generation {gen}/{self.n_gens}")

            # STEP 3 — parent selection
            parents = binary_tournament_selection(population, scores, n_select=self.pop_size)

            # STEP 4 — crossover (+ repair if constrained)
            offspring = two_point_crossover(parents, self.n_layers, controller=ctrl)

            # STEP 5 — mutation (constraint-aware if controller present)
            offspring = polynomial_mutation(offspring, self.n_blocks, controller=ctrl, mutation_prob=self.mutation_prob)

            # STEP 5b — CNOT dropout (creates dynamic arch variants; disabled when prob=0)
            if self.cnot_dropout_prob > 0 and self.search_space_obj is not None:
                offspring = cnot_dropout_mutation(offspring, self.search_space_obj, self.cnot_dropout_prob)

            # STEP 5c — R-gate dropout (disabled when prob=0)
            if self.r_dropout_prob > 0 and self.search_space_obj is not None:
                offspring = r_dropout_mutation(offspring, self.search_space_obj, self.r_dropout_prob)

            # STEP 6 — duplicate elimination (uses constrained sampling for fill)
            offspring = eliminate_duplicates(offspring, population, self.n_blocks, controller=ctrl)

            # STEP 7 — evaluate offspring
            offspring_scores = evaluate_population(offspring, eval_func, cache)
            
            #Inside the loop, after evaluating offspring (Step 7):
            offspring_generation = np.full(len(offspring), gen, dtype=int)  # tag with current gen

            # Step 8 — combine:
            combined_pop        = np.vstack([population, offspring])
            combined_scores     = np.concatenate([scores, offspring_scores])
            combined_generations = np.concatenate([generation_added, offspring_generation])  # NEW

            population, scores, generation_added = survival_selection(
                combined_pop, combined_scores, self.pop_size,
                generation_added=combined_generations,
                aging=self.aging,
            )

            print(f"  Best score this gen: {scores.min():.6f}")

            evolution_end  = time.perf_counter()
            evolution_time = evolution_end - evolution_start
            print(f"  Cumulative evolution time: {evolution_time:.4f}s")

            if self.aim_run is not None:
                self.aim_run["evolution_time"] = evolution_time

        # FINAL — sort all evaluated architectures
        sorted_results   = sorted(cache.items(), key=lambda x: x[1])
        self.subnet_topk = [key for key, _ in sorted_results[:10]]
        self.subnet_eval_dict = cache

        print(f"\n✓ Evolution complete. Evaluated {len(cache)} unique architectures.")
        print(f"  Top architecture: {self.subnet_topk[0]} (score: {sorted_results[0][1]:.6f})")

        return sorted_results
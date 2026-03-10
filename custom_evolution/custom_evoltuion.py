
import numpy as np
import time
from .utils import arch_to_key
# =============================================================================
# Step 1: Population Initialization
# Mirrors: FloatRandomSampling in pymoo, then rounded to integers
# =============================================================================

def initialize_population(pop_size: int, n_layers: int, n_blocks: int) -> np.ndarray:
    """Generate an initial random population of architectures.
    
    Each architecture is a 1D array of length n_layers, where each value
    is a random integer in [0, n_blocks - 1] representing a block choice.
    
    Mirrors: pymoo's RandomSampling() followed by rounding to valid integer indices.
    
    Args:
        pop_size: Number of individuals in the population
        n_layers: Number of layers per architecture
        n_blocks: Number of block options per layer
        
    Returns:
        2D array of shape (pop_size, n_layers) with integer block indices
    """
    # Random integers in [0, n_blocks - 1] for each individual and layer
    return np.random.randint(0, n_blocks, size=(pop_size, n_layers))


# =============================================================================
# Step 2: Evaluation
# Mirrors: NAS._evaluate() in the original code
# Caches results to avoid re-evaluating the same architecture.
# =============================================================================

def evaluate_population(
    population: np.ndarray,
    eval_func,
    cache: dict,
) -> np.ndarray:
    """Evaluate all architectures in the population, using cache where possible.
    
    Mirrors: NAS._evaluate() — iterates over each individual, checks cache,
    calls eval_func for new architectures, stores results in cache.
    
    NOTE: eval_func should return a score where LOWER is better
    (same convention as the original: score = |energy - exact_value|).
    
    Args:
        population: 2D array of shape (pop_size, n_layers)
        eval_func: Function that takes a 1D arch array and returns a scalar score
        cache: Dict mapping arch key -> score (modified in place)
        
    Returns:
        1D array of scores for each individual in population
    """
    scores = np.full(len(population), np.nan)

    for i, arch in enumerate(population):
        key = arch_to_key(arch)

        if key in cache:
            # Use cached result — same as the result_dict check in NAS._evaluate()
            scores[i] = cache[key]
        else:
            # Evaluate and cache — same as calling eval_func(x[i]) in NAS._evaluate()
            score = eval_func(arch)
            cache[key] = score
            scores[i] = score

        print(f"==evaluation subnet:{key} score:{scores[i]:.6f}")

    return scores


# =============================================================================
# Step 3: Binary Tournament Selection
# Mirrors: binary_tournament() + TournamentSelection in nsganet.py
#
# For SINGLE objective:
#   - dominance check simplifies to: lower score = dominates
#   - crowding distance is not needed (only relevant for multi-objective)
#   - so: just compare scores directly, random tie-break
# =============================================================================

def binary_tournament_selection(
    population: np.ndarray,
    scores: np.ndarray,
    n_select: int
) -> np.ndarray:
    """Select parents using binary tournament selection.
    
    For each selection, randomly pick 2 individuals and return the one
    with the lower score (lower = better). Ties broken randomly.
    
    Mirrors: TournamentSelection(func_comp=binary_tournament) in nsganet.py.
    For single-objective search, dominance check reduces to simple score comparison.
    
    Args:
        population: 2D array of shape (pop_size, n_layers)
        scores: 1D array of scores (lower is better)
        n_select: How many parents to select
        
    Returns:
        2D array of shape (n_select, n_layers) — selected parent architectures
    """
    pop_size = len(population)
    selected = []

    for _ in range(n_select):
        # Pick 2 random competitors (with replacement allowed, same as pymoo)
        a, b = np.random.choice(pop_size, size=2, replace=False)

        # Compare scores — lower wins (mirrors dominance check for single objective)
        if scores[a] < scores[b]:
            winner = a
        elif scores[b] < scores[a]:
            winner = b
        else:
            # Tie — pick randomly (mirrors return_random_if_equal=True)
            winner = np.random.choice([a, b])

        selected.append(population[winner].copy())

    return np.array(selected)


# =============================================================================
# Step 4: Two-Point Crossover
# Mirrors: PointCrossover(n_points=2) in nsganet.py
#
# Pick 2 random cut points, swap the middle segment between parents.
# Example with n_layers=6, cut at 2 and 4:
#   parent_a: [A0 A1 | A2 A3 | A4 A5]
#   parent_b: [B0 B1 | B2 B3 | B4 B5]
#   child_1:  [A0 A1   B2 B3   A4 A5]
#   child_2:  [B0 B1   A2 A3   B4 B5]
# =============================================================================

def two_point_crossover(
    parents: np.ndarray,
    n_layers: int
) -> np.ndarray:
    """Apply two-point crossover to produce offspring from pairs of parents.
    
    Parents are paired sequentially: (0,1), (2,3), etc.
    Each pair produces 2 children by swapping the segment between 2 cut points.
    
    Mirrors: PointCrossover(n_points=2) in pymoo.
    
    Args:
        parents: 2D array of shape (n_parents, n_layers) — must be even count
        n_layers: Number of layers per architecture
        
    Returns:
        2D array of shape (n_parents, n_layers) — offspring architectures
    """
    n_parents = len(parents)
    offspring = np.empty_like(parents)

    # Process pairs of parents
    for i in range(0, n_parents - 1, 2):
        p1 = parents[i].copy()
        p2 = parents[i + 1].copy()

        # Pick 2 distinct cut points
        cut1, cut2 = sorted(np.random.choice(n_layers + 1, size=2, replace=False))

        # Swap the middle segment (same as 2-point crossover)
        c1 = np.concatenate([p1[:cut1], p2[cut1:cut2], p1[cut2:]])
        c2 = np.concatenate([p2[:cut1], p1[cut1:cut2], p2[cut2:]])

        offspring[i] = c1
        offspring[i + 1] = c2

    # If odd number of parents, last one passes through unchanged
    if n_parents % 2 == 1:
        offspring[-1] = parents[-1].copy()

    return offspring


# =============================================================================
# Step 5: Polynomial Mutation
# Mirrors: PolynomialMutation(eta=3) in nsganet.py
#
# For integer NAS problems, polynomial mutation on integers simplifies to:
# with probability (1/n_layers), replace a layer's block index with a random one.
# This is the standard interpretation for discrete/integer NAS search.
# =============================================================================

def polynomial_mutation(
    offspring: np.ndarray,
    n_blocks: int,
    mutation_prob: float = None
) -> np.ndarray:
    """Apply mutation to offspring architectures.
    
    Each gene (layer index) is mutated with probability mutation_prob.
    Mutated genes are replaced with a uniformly random block index.
    
    Mirrors: PolynomialMutation(eta=3) from pymoo — for integer variables,
    this reduces to random replacement with probability 1/n_layers.
    
    Args:
        offspring: 2D array of shape (n_offspring, n_layers)
        n_blocks: Number of valid block choices per layer [0, n_blocks-1]
        mutation_prob: Per-gene mutation probability.
                      Defaults to 1/n_layers (standard NAS convention).
        
    Returns:
        2D array of mutated offspring (same shape)
    """
    mutated = offspring.copy()
    n_individuals, n_layers = mutated.shape

    if mutation_prob is None:
        mutation_prob = 1.0 / n_layers  # Same default as pymoo's PolynomialMutation

    for i in range(n_individuals):
        for j in range(n_layers):
            if np.random.rand() < mutation_prob:
                # Replace with a random block index (uniform over valid range)
                mutated[i, j] = np.random.randint(0, n_blocks)

    return mutated


# =============================================================================
# Step 6: Duplicate Elimination
# Mirrors: eliminate_duplicates=True in nsganet()
#
# Remove any architectures that already exist in the current population.
# If we remove too many, fill back up with random new individuals.
# =============================================================================

def eliminate_duplicates(
    offspring: np.ndarray,
    population: np.ndarray,
    n_blocks: int
) -> np.ndarray:
    """Remove offspring that are duplicates of existing population members.
    
    Converts each architecture to a string key for fast lookup.
    Fills any gaps left by removed duplicates with fresh random architectures.
    
    Mirrors: eliminate_duplicates=True in the nsganet() factory function.
    
    Args:
        offspring: 2D array of candidate offspring
        population: 2D array of current population (used to check for duplicates)
        n_blocks: Number of valid block choices (for generating replacements)
        
    Returns:
        2D array of same shape as offspring, with duplicates replaced
    """
    n_layers = offspring.shape[1]

    # Build a set of existing architecture keys for O(1) lookup
    existing_keys = {arch_to_key(arch) for arch in population}

    cleaned = []
    for arch in offspring:
        key = arch_to_key(arch)
        if key not in existing_keys:
            cleaned.append(arch)
            existing_keys.add(key)  # Also prevent duplicates within offspring
        # If duplicate, skip it (will be replaced below)

    # Fill gaps with random new architectures
    while len(cleaned) < len(offspring):
        new_arch = np.random.randint(0, n_blocks, size=n_layers)
        key = arch_to_key(new_arch)
        if key not in existing_keys:
            cleaned.append(new_arch)
            existing_keys.add(key)

    return np.array(cleaned[:len(offspring)])


# =============================================================================
# Step 7: Survival Selection
# Mirrors: RankAndCrowdingSurvival in nsganet.py
#
# For SINGLE objective, non-dominated sorting + crowding simplifies to:
# just sort by score and keep the top N. No fronts, no crowding needed.
# =============================================================================

def survival_selection(
    population: np.ndarray,
    scores: np.ndarray,
    pop_size: int
) -> tuple:
    """Select survivors by keeping the top-N lowest scoring architectures.
    
    For single-objective optimization, RankAndCrowdingSurvival reduces to
    a simple sort-and-truncate: lower score = better = survives.
    
    Mirrors: RankAndCrowdingSurvival._do() from nsganet.py, simplified for
    single objective (no crowding distance needed).
    
    Args:
        population: 2D array of shape (current_pop_size, n_layers)
        scores: 1D array of scores for each individual
        pop_size: How many survivors to keep
        
    Returns:
        Tuple of (survivors array, survivor scores array)
    """
    # Sort by score ascending (lower = better), with random tie-breaking
    random_keys = np.random.rand(len(scores))
    sorted_indices = np.lexsort((random_keys, scores))  # Same tie-breaking as randomized_argsort

    # Keep top pop_size individuals
    survivor_indices = sorted_indices[:pop_size]

    return population[survivor_indices], scores[survivor_indices]


# =============================================================================
# Main Evolutionary Loop
# Mirrors: minimize() from pymoo — this is the explicit loop that was hidden.
# =============================================================================

class EvolutionSampler:
    """
    Pymoo-free evolutionary sampler for quantum/neural architecture search.

    Implements the same algorithm as the original NSGA-Net based sampler:
    - Random initialization
    - Binary tournament selection
    - Two-point crossover
    - Polynomial mutation
    - Duplicate elimination
    - Top-N survival (single-objective simplification of rank+crowding)

    All steps are explicit — no hidden minimize() calls or abstract Problem classes.

    Attributes:
        pop_size: Number of individuals per generation
        n_gens: Number of generations to evolve
        n_layers: Number of layers per architecture
        n_blocks: Number of block choices per layer
        subnet_eval_dict: Cache of all evaluated architectures {key: score}
        subnet_topk: Top 10 architecture keys after search (best first)
    """

    def __init__(
        self,
        pop_size: int = 50,
        n_gens: int = 20,
        n_layers: int = 3,
        n_blocks: int = 12, #n_blocks is the search space!!
        aim_run = None
    ):
        self.pop_size = pop_size
        self.n_gens = n_gens
        self.n_layers = n_layers
        self.n_blocks = n_blocks

        # Populated after sample() is called
        self.subnet_eval_dict = {}
        self.subnet_topk = []
        self.aim_run = aim_run


    def sample(self, eval_func) -> list:
        """
        Run the full evolutionary search.

        This is the explicit version of pymoo's minimize() call.
        Each step mirrors what pymoo was doing internally.

        Args:
            eval_func: Function(arch: np.ndarray) -> float
                       Takes a 1D architecture array, returns scalar score.
                       LOWER score = BETTER (e.g. |energy - exact_value|).

        Returns:
            List of (key, score) tuples sorted best to worst.
        """
        cache = self.subnet_eval_dict  # Shared cache — avoids re-evaluation
        
        # --------------------------------------------------------------
        # Evolution timer start
        # --------------------------------------------------------------
        evolution_start = time.perf_counter()

        # ------------------------------------------------------------------
        # STEP 1: Initialize random population
        # Mirrors: RandomSampling() in pymoo
        # ------------------------------------------------------------------
        population = initialize_population(self.pop_size, self.n_layers, self.n_blocks)
        print(f"Initialized population: {self.pop_size} individuals, "
              f"{self.n_layers} layers, {self.n_blocks} blocks")

        # ------------------------------------------------------------------
        # STEP 2: Evaluate initial population
        # Mirrors: algorithm._evaluate() on the initial pop in minimize()
        # ------------------------------------------------------------------
        scores = evaluate_population(population, eval_func, cache)

        # ------------------------------------------------------------------
        # MAIN LOOP — this is the explicit minimize() loop
        # Mirrors: the generation loop inside pymoo's minimize()
        # ------------------------------------------------------------------
        for gen in range(1, self.n_gens + 1):
            print(f"\n==Finished generation: {gen}")

            # --------------------------------------------------------------
            # STEP 3: Parent selection via binary tournament
            # Mirrors: TournamentSelection(func_comp=binary_tournament)
            # Select pop_size parents (same count as population)
            # --------------------------------------------------------------
            parents = binary_tournament_selection(population, scores, n_select=self.pop_size)

            # --------------------------------------------------------------
            # STEP 4: Two-point crossover to create offspring
            # Mirrors: PointCrossover(n_points=2)
            # --------------------------------------------------------------
            offspring = two_point_crossover(parents, self.n_layers)

            # --------------------------------------------------------------
            # STEP 5: Polynomial mutation
            # Mirrors: PolynomialMutation(eta=3)
            # Default prob = 1/n_layers
            # --------------------------------------------------------------
            offspring = polynomial_mutation(offspring, self.n_blocks)

            # --------------------------------------------------------------
            # STEP 6: Eliminate duplicates
            # Mirrors: eliminate_duplicates=True in nsganet()
            # --------------------------------------------------------------
            offspring = eliminate_duplicates(offspring, population, self.n_blocks)

            # --------------------------------------------------------------
            # STEP 7: Evaluate offspring
            # Mirrors: NAS._evaluate() called on offspring in minimize()
            # --------------------------------------------------------------
            offspring_scores = evaluate_population(offspring, eval_func, cache)

            # --------------------------------------------------------------
            # STEP 8: Combine population + offspring, select survivors
            # Mirrors: RankAndCrowdingSurvival (simplified for single objective)
            # pymoo combines pop + offspring, then selects top pop_size
            # --------------------------------------------------------------
            combined_pop = np.vstack([population, offspring])
            combined_scores = np.concatenate([scores, offspring_scores])

            population, scores = survival_selection(combined_pop, combined_scores, self.pop_size)

            # Print best score this generation
            print(f"  Best score this gen: {scores[0]:.6f}")
            
            # --------------------------------------------------------------
            # Evolution timer end
            # --------------------------------------------------------------
            evolution_end = time.perf_counter()
            evolution_time = evolution_end - evolution_start

            print(f"\n✓ Evolution runtime: {evolution_time:.4f} seconds")

            if self.aim_run is not None:
                self.aim_run["evolution_time"] = evolution_time
        # ------------------------------------------------------------------
        # FINAL: Sort all evaluated architectures and extract top-k
        # Mirrors: the sorted_subnet / subnet_topk logic in original sample()
        # ------------------------------------------------------------------
        sorted_results = sorted(cache.items(), key=lambda x: x[1])  # ascending score

        self.subnet_topk = [key for key, _ in sorted_results[:10]]
        self.subnet_eval_dict = cache

        print(f"\n✓ Evolution complete. Evaluated {len(cache)} unique architectures.")
        print(f"  Top architecture: {self.subnet_topk[0]} (score: {sorted_results[0][1]:.6f})")

        return sorted_results
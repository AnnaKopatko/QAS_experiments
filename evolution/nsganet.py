"""
NSGA-Net: Neural Architecture Search using NSGA-II Genetic Algorithm.

This module implements a customized version of the NSGA-II (Non-dominated Sorting 
Genetic Algorithm II) specifically adapted for neural architecture search. It includes
custom tournament selection, crowding distance calculation, and survival mechanisms.

Based on NSGA-II from https://github.com/msu-coinlab/pymoo
"""

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2 as GeneticAlgorithm

from pymoo.docs import parse_doc_string
from pymoo.core.individual import Individual

from pymoo.core.survival import Survival
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling as RandomSampling


from pymoo.operators.selection.tournament import compare, TournamentSelection


from pymoo.util.dominator import Dominator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np

def randomized_argsort(values):
    """
    Sort array indices with randomized tie-breaking.
    
    Equivalent to old pymoo.util.randomized_argsort. When multiple values are equal,
    their relative ordering is randomized rather than arbitrary.
    
    Args:
        values (np.array): Array of values to sort
        
    Returns:
        np.array: Indices that would sort the array, with random tie-breaking
    """
    # Generate random keys for tie-breaking
    random_keys = np.random.rand(len(values))
    
    # Sort by value first, then by random key to break ties
    # lexsort sorts by the last key first, so random_keys breaks ties in values
    return np.lexsort((random_keys, values))



# =========================================================================================================
# Implementation
# based on nsga2 from https://github.com/msu-coinlab/pymoo
# =========================================================================================================


class NSGANet(GeneticAlgorithm):
    """
    NSGA-Net algorithm for neural architecture search.
    
    Extends the standard NSGA-II genetic algorithm with customizations specific
    to neural architecture search problems. Uses tournament selection based on
    dominance and crowding distance.
    
    Attributes:
        tournament_type (str): Type of tournament selection to use
                              ('comp_by_dom_and_crowding' or 'comp_by_rank_and_crowding')
        individual (Individual): Template individual with rank and crowding attributes
    """

    def __init__(self, **kwargs):
        """
        Initialize the NSGA-Net algorithm.
        
        Args:
            **kwargs: Keyword arguments passed to parent NSGA2 algorithm
        """
        # Initialize individuals with rank (infinity) and crowding distance (-1)
        kwargs['individual'] = Individual(rank=np.inf, crowding=-1)
        super().__init__(**kwargs)

        # Set tournament selection to compare by dominance and crowding distance
        self.tournament_type = 'comp_by_dom_and_crowding'



# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------


def binary_tournament(pop, P, algorithm, **kwargs):
    """
    Perform binary tournament selection between pairs of individuals.
    
    Selects the better individual from each pair based on constraint violation,
    dominance relation, rank, and crowding distance (in that priority order).
    
    Args:
        pop (Population): Current population of individuals
        P (np.array): Array of shape (n_tournaments, 2) containing pairs of indices
                     to compete in tournaments
        algorithm: Algorithm object containing tournament_type attribute
        **kwargs: Additional keyword arguments (unused)
        
    Returns:
        np.array: Array of shape (n_tournaments, 1) containing indices of winners
        
    Raises:
        ValueError: If P does not have exactly 2 columns (not binary tournament)
    """
    # Verify this is binary tournament (exactly 2 competitors)
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")

    # Get the tournament selection strategy from algorithm
    tournament_type = algorithm.tournament_type
    
    # Initialize array to store winners (NaN = no winner selected yet)
    S = np.full(P.shape[0], np.nan)

    # Run each tournament
    for i in range(P.shape[0]):

        # Get indices of the two competing individuals
        a, b = P[i, 0], P[i, 1]

        # PRIORITY 1: If at least one solution violates constraints, prefer the feasible one
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # Both solutions are feasible - use tournament type to decide
        else:

            # Tournament based on dominance relation
            if tournament_type == 'comp_by_dom_and_crowding':
                # Check if one solution dominates the other
                rel = Dominator.get_relation(pop[a].F, pop[b].F)
                if rel == 1:    # a dominates b
                    S[i] = a
                elif rel == -1: # b dominates a
                    S[i] = b
                # If rel == 0 (non-dominated), S[i] remains NaN and crowding decides below

            # Tournament based on rank comparison
            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, pop[a].rank, b, pop[b].rank,
                               method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # PRIORITY 2: If rank or domination relation didn't make a decision,
            # use crowding distance (prefer more isolated solutions)
            if np.isnan(S[i]):
                S[i] = compare(a, pop[a].get("crowding"), b, pop[b].get("crowding"),
                               method='larger_is_better', return_random_if_equal=True)

    # Return winners as column vector of integers
    return S[:, None].astype(int)


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------


class RankAndCrowdingSurvival(Survival):
    """
    Survival selection based on non-dominated sorting and crowding distance.
    
    Selects survivors from a population using:
    1. Non-dominated sorting to assign ranks (lower rank = better)
    2. Crowding distance to maintain diversity within each rank
    
    When a front must be split, individuals with larger crowding distance
    (more isolated in objective space) are preferred to maintain diversity.
    
    Attributes:
        filter_infeasible (bool): Whether to filter infeasible solutions (inherited)
    """

    def __init__(self) -> None:
        """
        Initialize the survival selection operator.
        
        Sets filter_infeasible to True to remove constraint-violating solutions.
        """
        super().__init__(True)

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        """
        Execute survival selection to choose which individuals survive.
        
        Uses non-dominated sorting and crowding distance to select n_survive
        individuals from the population.
        
        Args:
            problem: Optimization problem instance
            pop (Population): Current population
            *args: Additional positional arguments (unused)
            n_survive (int): Number of individuals to select for survival
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            Population: Selected survivors
        """
        # Get number of survivors (pymoo passes it by keyword)
        n_survive = int(n_survive)

        # Get the objective space values from all individuals
        F = pop.get("F")

        # Initialize list to store indices of survivors
        survivors = []

        # Perform non-dominated sorting to organize population into fronts
        # Stop early if we've ranked enough individuals
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

        # Process each front
        for k, front in enumerate(fronts):
            # Calculate crowding distance for all individuals in this front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # Assign rank and crowding distance to each individual in the front
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # Check if adding this entire front would exceed n_survive
            if len(survivors) + len(front) > n_survive:
                # This front must be split - select individuals with highest crowding distance
                I = randomized_argsort(crowding_of_front)
                # Take only enough to reach n_survive (most crowded individuals)
                I = I[:(n_survive - len(survivors))]
            else:
                # Take the entire front
                I = np.arange(len(front))

            # Add selected individuals from this front to survivors
            survivors.extend(front[I])

        # Return the population containing only survivors
        return pop[survivors]



def calc_crowding_distance(F):
    """
    Calculate crowding distance for a set of points in objective space.
    
    Crowding distance measures how isolated a point is from its neighbors.
    Higher values indicate the point is in a less crowded region, which helps
    maintain diversity in the population. Boundary points get infinite distance.
    
    Args:
        F (np.array): 2D array of shape (n_points, n_objectives) containing
                     objective values for each point
                     
    Returns:
        np.array: 1D array of crowding distances for each point
    """
    # Use large number instead of actual infinity for numerical stability
    infinity = 1e+14

    n_points = F.shape[0]
    n_obj = F.shape[1]

    # Special case: 2 or fewer points get infinite crowding distance
    if n_points <= 2:
        return np.full(n_points, infinity)
    else:

        # Sort indices for each objective separately
        I = np.argsort(F, axis=0, kind='mergesort')

        # Rearrange F according to sorted indices for each objective
        F = F[I, np.arange(n_obj)]

        # Calculate distance to neighbors for each objective
        # Add infinity boundaries at top and -infinity at bottom
        dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) \
               - np.concatenate([np.full((1, n_obj), -np.inf), F])

        # Find where distances are zero (duplicate objective values)
        index_dist_is_zero = np.where(dist == 0)

        # Calculate distance to last non-duplicate point
        dist_to_last = np.copy(dist)
        for i, j in zip(*index_dist_is_zero):
            dist_to_last[i, j] = dist_to_last[i - 1, j]

        # Calculate distance to next non-duplicate point
        dist_to_next = np.copy(dist)
        for i, j in reversed(list(zip(*index_dist_is_zero))):
            dist_to_next[i, j] = dist_to_next[i + 1, j]

        # Normalize distances by the range of each objective
        norm = np.max(F, axis=0) - np.min(F, axis=0)
        norm[norm == 0] = np.nan  # Avoid division by zero
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # Replace NaN values (from zero range) with 0.0
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # Unsort to get back to original ordering
        J = np.argsort(I, axis=0)
        
        # Sum distances for all objectives and normalize by number of objectives
        crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    # Replace any infinite values with the large number
    crowding[np.isinf(crowding)] = infinity

    return crowding


# =========================================================================================================
# Interface
# =========================================================================================================


def nsganet(
        pop_size=100,
        sampling=RandomSampling(),
        selection=TournamentSelection(func_comp=binary_tournament),
        crossover=PointCrossover(n_points=2),
        mutation=PolynomialMutation(eta=3),

        eliminate_duplicates=True,
        n_offsprings=None,
        **kwargs):
    """
    Create an NSGA-Net algorithm instance for neural architecture search.
    
    Factory function that configures and returns an NSGA-Net algorithm with
    specified parameters. NSGA-Net is a variant of NSGA-II adapted for NAS problems.

    Parameters
    ----------
    pop_size : int, optional (default: 100)
        Population size - number of individuals in each generation
        
    sampling : Sampling, optional (default: RandomSampling())
        Sampling strategy for generating initial population
        
    selection : Selection, optional (default: TournamentSelection with binary_tournament)
        Parent selection operator for choosing individuals to reproduce
        
    crossover : Crossover, optional (default: PointCrossover with 2 points)
        Crossover operator for combining parent architectures
        
    mutation : Mutation, optional (default: PolynomialMutation with eta=3)
        Mutation operator for introducing variation in offspring
        
    eliminate_duplicates : bool, optional (default: True)
        Whether to eliminate duplicate individuals from the population
        
    n_offsprings : int or None, optional (default: None)
        Number of offspring to generate each generation.
        If None, uses the population size
        
    **kwargs : dict
        Additional keyword arguments passed to NSGANet constructor

    Returns
    -------
    nsganet : NSGANet
        Configured NSGA-Net algorithm object ready for optimization

    """

    return NSGANet(pop_size=pop_size,
                   sampling=sampling,
                   selection=selection,
                   crossover=crossover,
                   mutation=mutation,
                   survival=RankAndCrowdingSurvival(),
                   eliminate_duplicates=eliminate_duplicates,
                   n_offsprings=n_offsprings,
                   **kwargs)


# Parse and format the docstring for pymoo documentation system
parse_doc_string(nsganet)
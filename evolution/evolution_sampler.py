"""
Neural Architecture Search (NAS) using NSGA-Net evolutionary algorithm.

This module implements an evolutionary sampler for optimizing neural network architectures
using the NSGA-Net algorithm. It searches for optimal subnet configurations by evolving
a population of architectures over multiple generations.
"""

import numpy as np
from evolution import nsganet as engine
from pymoo.core.problem import Problem
import mlflow 
from pymoo.optimize import minimize


class EvolutionSampler:
    """
    Evolutionary sampler for neural architecture search.
    
    Uses NSGA-Net (evolutionary algorithm) to search for optimal network architectures
    by evolving a population of subnets over multiple generations.
    
    Attributes:
        pop_size (int): Size of the population in each generation
        n_gens (int): Number of generations to evolve
        n_layers (int): Number of layers/variables in each architecture
        n_blocks (int): Number of possible blocks to choose from for each layer
    """
    
    def __init__(self, pop_size=50, n_gens=20, n_layers=3, n_blocks=12):
        """
        Initialize the evolutionary sampler.
        
        Args:
            pop_size (int): Population size (default: 50)
            n_gens (int): Number of generations (default: 20)
            n_layers (int): Number of layers in the architecture (default: 3)
            n_blocks (int): Number of block options per layer (default: 12)
        """
        self.pop_size = pop_size
        self.n_gens = n_gens
        self.n_layers = n_layers
        self.n_blocks = n_blocks

    def sample(self, eval_func=None):
        """
        Run the evolutionary search to find optimal architectures.
        
        Executes the NSGA-Net algorithm to evolve a population of subnet architectures,
        evaluating each using the provided evaluation function.
        
        Args:
            eval_func (callable): Function to evaluate architecture performance.
                                Should take architecture array and return a scalar metric.
        
        Returns:
            list: Sorted list of (architecture_string, energy) tuples, 
                  ordered from best to worst performance
        """
        # Dictionary to cache evaluation results and avoid re-evaluating duplicates
        subnet_eval_dict = {}
        
        # Number of offspring to generate (None uses default from algorithm)
        n_offspring = None #40
        
        # Setup NAS search problem
        # Each architecture is represented by n_layers variables
        n_var = self.n_layers
        
        # Lower bound: each layer can select block index 0
        lb = np.zeros(n_var)
        
        # Upper bound: each layer can select up to block index (n_blocks - 1)
        ub = np.zeros(n_var) + self.n_blocks - 1

        # Create the NAS optimization problem
        nas_problem = NAS(n_var=n_var, n_obj=1, n_constr=0, lb=lb, ub=ub,
                            eval_func=eval_func,
                            result_dict=subnet_eval_dict)

        # Configure the NSGA-Net evolutionary algorithm
        method = engine.nsganet(pop_size=self.pop_size,
                                n_offsprings=n_offspring,
                                eliminate_duplicates=True)

        # Run the evolutionary optimization
        res = minimize(nas_problem,
                        method,
                        callback=lambda algorithm: self.generation_callback(algorithm),
                        termination=('n_gen', self.n_gens))

        # Extract top 10 architectures based on performance
        subnet_topk = []
        
        # Sort all evaluated subnets by their energy (lower is better)
        sorted_subnet = sorted(subnet_eval_dict.items(), key=lambda i: i[1])
        
        # Extract just the architecture strings (keys)
        sorted_subnet_key = [x[0] for x in sorted_subnet]
        
        # Select top 10 best performing architectures
        subnet_topk = sorted_subnet_key[:10]
        
        # Store results as instance attributes for later access
        self.subnet_topk = subnet_topk
        self.subnet_eval_dict = subnet_eval_dict
        
        return sorted_subnet


    def generation_callback(self, algorithm):
        """
        Callback function called after each generation.
        
        Prints generation progress information during the evolutionary search.
        
        Args:
            algorithm: The pymoo algorithm object containing current generation state
        """
        gen = algorithm.n_gen
        pop_var = algorithm.pop.get("X")  # Population architectures
        pop_obj = algorithm.pop.get("F")  # Population objective values
        print(f'==Finished generation: {gen}')


# ---------------------------------------------------------------------------------------------------------
# Define your NAS Problem
# ---------------------------------------------------------------------------------------------------------
class NAS(Problem):
    """
    Neural Architecture Search optimization problem for pymoo.
    
    Defines the NAS problem as a single-objective optimization where architectures
    are evaluated and compared based on their performance metric (energy).
    
    Attributes:
        xl (np.array): Lower bounds for each decision variable
        xu (np.array): Upper bounds for each decision variable
        _n_evaluated (int): Counter tracking total number of architectures evaluated
        eval_func (callable): User-provided function to evaluate architecture performance
        result_dict (dict): Cache storing evaluation results with architecture strings as keys
    """
    
    def __init__(self, n_var=20, n_obj=1, n_constr=0, lb=None, ub=None, eval_func=None, result_dict=None):
        """
        Initialize the NAS problem.
        
        Args:
            n_var (int): Number of decision variables (architecture layers)
            n_obj (int): Number of objectives to optimize (default: 1)
            n_constr (int): Number of constraints (default: 0)
            lb (np.array): Lower bounds for each variable
            ub (np.array): Upper bounds for each variable
            eval_func (callable): Function to evaluate architecture performance
            result_dict (dict): Dictionary to cache evaluation results
        """
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr)
        
        # Set variable bounds
        self.xl = lb
        self.xu = ub
        
        # Counter for tracking total number of architecture evaluations
        self._n_evaluated = 0
        
        # Evaluation function provided by user
        self.eval_func = eval_func
        
        # Cache for storing evaluation results to avoid redundant evaluations
        self.result_dict = result_dict

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate a batch of architectures.
        
        Called by the optimization algorithm to evaluate objective values for
        a population of architectures. Caches results to avoid re-evaluation.
        
        Args:
            x (np.array): 2D array where each row is an architecture to evaluate
            out (dict): Output dictionary to store objective values
            *args, **kwargs: Additional arguments (unused)
        """
        # Initialize objectives array with NaN values
        objs = np.full((x.shape[0], self.n_obj), np.nan)

        # Evaluate each architecture in the batch
        for i in range(x.shape[0]):
            # Assign unique ID to this architecture
            arch_id = self._n_evaluated + 1

            # NOTE: All objectives are assumed to be MINIMIZED
            # Convert architecture array to string for use as cache key
            key = str(x[i])
            
            # Check if this architecture has already been evaluated
            if self.result_dict.get(key) is not None:
                # Retrieve cached result
                energy = self.result_dict[key]
            else:
                # Evaluate new architecture
                energy = self.eval_func(x[i])
                
                # Cache the result
                self.result_dict[key] = energy

            print('==evaluation subnet:{} energy:{}'.format(key, energy))

            # Store the objective value (energy - lower is better)
            objs[i, 0] = energy 
            
            # Log metric to MLflow for tracking
            if True:
                mlflow.log_metric("evolution_energy", float(energy), i)# lower energy is better
            
            # If multiple objectives are needed, add them here:
            # objs[i, 1] = ...  # additional objectives if needed

            # Increment evaluation counter
            self._n_evaluated += 1
            
        # Set the objective values in the output dictionary
        out["F"] = objs
        
        # If your NAS problem has constraints, use the following line to set constraints:
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
def do_every_generations(algorithm):
    """
    Callback function to execute custom logic after each generation.
    
    This function is called by the optimization algorithm after each generation
    completes. It can be used to print statistics, save checkpoints, or log metrics.
    
    Args:
        algorithm: The pymoo algorithm object with access to current generation state
    """
    # Extract current generation number
    gen = algorithm.n_gen
    
    # Get population decision variables (architectures)
    pop_var = algorithm.pop.get("X")
    
    # Get population objective values (performance metrics)
    pop_obj = algorithm.pop.get("F")
    
    # Print generation number
    print(gen)

    # Additional generation information can be printed here:
    # print(gen, pop_var, pop_obj)

    # Report generation info to files (custom implementation would go here)

def main():
    """
    Example usage of the NAS evolutionary search.
    
    Demonstrates how to set up and run a basic NAS experiment using the
    NSGA-Net algorithm with default parameters.
    """
    # Hyper parameters for the evolutionary search
    pop_size = 50      # Population size per generation
    n_gens = 20        # Number of generations to evolve
    n_offspring = 40   # Number of offspring to generate each generation

    # Setup NAS search problem
    n_var = 20  # Number of variables (layers) in each architecture
    
    # Lower bounds: each layer can select block 0
    lb = np.zeros(n_var)
    
    # Upper bounds: each layer can select up to block 4
    ub = np.zeros(n_var) + 4

    # Create the NAS problem instance
    nas_problem = NAS(n_var=n_var, n_obj=1, n_constr=0, lb=lb, ub=ub)

    # Configure the NSGA-Net evolutionary algorithm
    method = engine.nsganet(pop_size=pop_size,
                            n_offsprings=n_offspring,
                            eliminate_duplicates=True)

    # Run the optimization
    res = minimize(nas_problem,
                   method,
                   callback=do_every_generations,
                   termination=('n_gen', n_gens))
    
    # Print available attributes of the result object
    print(dir(res))
    
    # Print size of final population
    print(len(res.pop))
    
    # Print top 10 solutions (objective value and architecture)
    for pop in res.pop[:10]:
        print(pop.F, pop.X)
        
    return res


if __name__ == "__main__":
    main()
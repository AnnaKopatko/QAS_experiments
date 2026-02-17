"""
Circuit Search Model for Quantum Architecture Search

This module implements the CircuitSearchModel class, which is the core component
for Neural Architecture Search (NAS) in quantum circuits. Unlike CircuitModel
which trains a single fixed architecture, this model maintains multiple "expert"
parameter sets and can dynamically switch between different architectures during
the search process.

Key Concept - Experts:
    The search model uses multiple independent parameter sets called "experts".
    Each expert learns to optimize different subsets of the architecture space.
    During search, the best expert is selected for each architecture based on
    its performance, allowing efficient exploration of the search space.
"""

import pennylane as qml
from pennylane import numpy as np
from .circuit_model import qas_layer


class CircuitSearchModel():
    """
    Quantum circuit model for architecture search with multiple experts.
    
    This model is designed for the search phase of QAS, where we want to:
    1. Try many different architectures efficiently
    2. Learn good parameters for each architecture
    3. Identify the best-performing architectures
    
    Key Features:
    - Maintains n_experts independent parameter sets
    - Each expert specializes in different rotation gate configurations
    - Supports dynamic architecture switching during training
    - Efficient parameter sharing across similar architectures
    
    Architecture Encoding:
        Each architecture is encoded as a list of integers (subnet).
        Example: [5, 12, 3] means:
            - Layer 0: Use architecture #5 from NAS_search_space
            - Layer 1: Use architecture #12 from NAS_search_space  
            - Layer 2: Use architecture #3 from NAS_search_space
            
    Parameter Organization:
        params_space shape: (n_experts, n_layers, len(Rs_space), n_qubits)
        - n_experts: Number of independent expert parameter sets
        - n_layers: Circuit depth
        - len(Rs_space): Number of unique rotation gate configurations
        - n_qubits: Number of qubits
        
        The rotation gate index is extracted from the subnet index:
            r_idx = subnet[j] // len(CNOTs_space)
        This works because NAS_search_space = Rs_space × CNOTs_space
        
    Attributes:
        dev: PennyLane quantum device
        n_qubits (int): Number of qubits
        n_layers (int): Number of circuit layers
        n_experts (int): Number of expert parameter sets
        search_space (SearchSpace): Instance of the SearchSpace class
        params_space (np.ndarray): All expert parameters, 
            shape (n_experts, n_layers, len(Rs_space), n_qubits)
        params (np.ndarray): Current active parameters for selected subnet/expert
        basis_state (list): Initial quantum state
        subnet (list): Currently active architecture (set by get_params)
        expert_idx (int): Currently active expert (set by get_params)
    """
    
    def __init__(self, dev, search_space, n_qubits=3, n_layers=3, n_experts=5, basis_state=None):
        """
        Initialize the circuit search model with multiple experts.
        
        Args:
            dev: PennyLane quantum device
            search_space (SearchSpace): Instance of the SearchSpace class
            n_qubits (int): Number of qubits in the circuit
            n_layers (int): Number of layers (circuit depth)
            n_experts (int): Number of independent expert parameter sets to maintain.
                Higher values allow better exploration but use more memory.
                Typical range: 3-10
            basis_state (list, optional): Initial quantum state. 
                Default: |000...0⟩
                Example: [1, 0, 1, 0] for Hartree-Fock state
        """
        self.dev = dev
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.search_space = search_space
        
        # Initialize parameter space for all experts
        # Shape: (n_experts, n_layers, len(Rs_space), n_qubits)
        # Randomly initialized in range [0, 2π]
        # 
        # Why len(Rs_space) in dimension 2?
        # - Rs_space contains all unique rotation gate configurations
        # - Each expert learns parameters for every possible rotation configuration
        # - CNOT patterns don't need parameters (they're just connectivity)
        # - This enables parameter sharing: similar architectures share rotation parameters
        self.params_space = np.random.uniform(0, np.pi * 2, 
                                             (n_experts, n_layers, len(self.search_space.Rs_space), n_qubits))
        
        # Current active parameters (set by get_params)
        self.params = None
        
        # Initial quantum state (e.g., Hartree-Fock for chemistry)
        self.basis_state = basis_state
    
    
    def get_params(self, subnet, expert_idx):
        """
        Retrieve parameters for a specific architecture (subnet) and expert.
        
        This method extracts the relevant parameters from the full parameter space
        based on the rotation gate configuration of each layer in the subnet.
        
        Important!: Each subnet index encodes both rotation gates AND CNOT pattern.
        We decode the rotation gate index and fetch those specific parameters.
        
        Args:
            subnet (list of int): Architecture specification, one index per layer.
                Each index points to an element in NAS_search_space.
                Example: [5, 12, 3] for a 3-layer circuit
            expert_idx (int): Which expert's parameters to use (0 to n_experts-1)
            
        Returns:
            np.ndarray: Parameters for this subnet/expert combination,
                shape (n_layers, n_qubits)
        """
        # Store current selection for later use in set_params
        self.subnet = subnet
        self.expert_idx = expert_idx
        
        params = []
        for j in range(self.n_layers):
            # Decode the rotation gate index from the subnet index
            # NAS_search_space is organized as: Rs_space × CNOTs_space
            # So dividing by CNOTs_space size gives us the Rs_space index
            r_idx = subnet[j] // len(self.search_space.CNOTs_space)
            
            # Convert to int for indexing (avoid floating point issues)
            ei, jj, ri = int(expert_idx), int(j), int(r_idx)
            
            # Extract parameters for this layer
            # [ei, jj, ri:ri+1] gives shape (1, n_qubits)
            params.append(self.params_space[ei, jj, ri:ri+1])

        # Concatenate all layers: final shape (n_layers, n_qubits)
        return np.concatenate(params, axis=0)

    def set_params(self, params):
        """
        Update the parameter space with new optimized parameters.
        
        After an optimization step, we need to write the updated parameters
        back into the parameter space for the current subnet and expert.
        This updates ONLY the parameters that were used (rotation configs),
        leaving other parameters unchanged.
        
        Args:
            params (np.ndarray): Updated parameters from optimizer,
                shape (n_layers, n_qubits)
        """
        for j in range(self.n_layers):
            # Decode the rotation gate index (same logic as get_params)
            r_idx = self.subnet[j] // len(self.search_space.CNOTs_space)
            
            # Write updated parameters back to the parameter space
            # Only update the specific rotation configuration that was used
            self.params_space[self.expert_idx, j, r_idx:r_idx+1] = params[j, :]

    def __call__(self, params, wires):
        """
        Make the model callable for use in QNodes.
        
        This builds the quantum circuit using the current subnet architecture
        and the provided parameters.
        
        Args:
            params (np.ndarray): Circuit parameters, shape (n_layers, n_qubits)
            wires (list or range): Qubit wires to use
        """
        self.circuit_search(params, wires=wires,
                       n_qubits=self.n_qubits,
                       n_layers=self.n_layers,
                       arch=self.subnet)
        
    def circuit_search(self, params, wires=[0,1,2,3], n_qubits=3, n_layers=3, arch=[]):
        """
        Prepares the initial state and then builds the circuit layer by layer
        according to the specified architecture.
        
        Args:
            params (np.ndarray): Circuit parameters, shape (n_layers, n_qubits)
            wires (list): Qubit wire labels
            n_qubits (int): Number of qubits
            n_layers (int): Number of layers
            arch (list of int): Architecture specification (subnet).
                Each element is an index into NAS_search_space
                
        Circuit Construction:
            1. Prepare initial basis state (e.g., Hartree-Fock for chemistry)
            2. For each layer j:
                a. Look up architecture: idx = arch[j]
                b. Retrieve (Rs, CNOTs) from NAS_search_space[idx]
                c. Build layer using qas_layer with these gates/connectivity
                
        Example:
            arch = [5, 12, 3]
            
            # Layer 0: architecture #5
            Rs_0, CNOTs_0 = NAS_search_space[5]
            # Build: RY/RZ gates according to Rs_0, CNOTs according to CNOTs_0
            
            # Layer 1: architecture #12
            Rs_1, CNOTs_1 = NAS_search_space[12]
            # Build: RY/RZ gates according to Rs_1, CNOTs according to CNOTs_1
            
            # Layer 2: architecture #3
            Rs_2, CNOTs_2 = NAS_search_space[3]
            # Build: RY/RZ gates according to Rs_2, CNOTs according to CNOTs_2
        """
        # Prepare initial state
        # For chemistry problems, this is typically the Hartree-Fock state
        if self.basis_state is None:
            self.basis_state = [0] * len(wires)

        qml.BasisState(np.array(self.basis_state), wires=wires)

        # Build circuit layer by layer using the NAS architecture
        for j in range(n_layers):
            # Get architecture specification for this layer
            idx = int(arch[j])
            
            # Retrieve rotation gates and CNOT pattern from search space
            # NAS_search_space[idx] = (Rs, CNOTs)
            # Rs: tuple of rotation gates (e.g., (RY, RZ, RY, RZ))
            # CNOTs: list of wire pairs (e.g., [[0,1], [2,3]])
            Rs, CNOTs = self.search_space.NAS_search_space[idx]
            
            # Build this layer with specified gates and connectivity
            qas_layer(params, j, n_qubits, Rs, CNOTs)
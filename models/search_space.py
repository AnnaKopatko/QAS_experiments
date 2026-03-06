"""
Search Space Definition for Quantum Architecture Search (QAS)

This module defines the Neural Architecture Search (NAS) space for quantum circuits.
It specifies the available rotation gates and CNOT configurations that can be combined
to form different quantum circuit architectures.
"""

import pennylane as qml
import itertools
import yaml


class SearchSpace:
    """
    Defines the Neural Architecture Search (NAS) space for quantum circuits.
    
    The search space is constructed by combining:
    1. Rotation gate configurations (Rs_space) - different combinations of rotation gates
    2. CNOT configurations (CNOTs_space) - different entanglement patterns
    
    Each architecture in the search space is a tuple (Rs, CNOTs) where:
    - Rs: tuple of rotation gates (one per qubit)
    - CNOTs: list of CNOT connections between qubits
    
    Attributes:
        valid_Rs (list): Available single-qubit rotation gates (loaded from config)
        valid_CNOTs (tuple): Valid CNOT wire pairs (nearest-neighbor only)
        n_qubits (int): Number of qubits (determines Rs_space size)
        Rs_space (list): All possible combinations of rotation gates
        CNOTs_space (list): All possible CNOT connectivity patterns (including subsets)
        NAS_search_space (list): Complete search space as Cartesian product of Rs_space × CNOTs_space
    """
    
    def __init__(self, config, n_qubits=None, run=None):
        """
        Initialize the search space by loading configuration and generating all 
        possible combinations of rotation gates and CNOT patterns.
        
        Args:
            config: The configuration
            n_qubits (int, optional): Number of qubits. If None, falls back to
                SearchSpace.num_repeats from config for backward compatibility.
        """
        # Extract search space configuration
        search_cfg = config.get("SearchSpace", {})
        
        # Define valid single-qubit rotation gates from config
        # Map gate names to PennyLane gate functions
        gate_mapping = {
            "RY": qml.RY,
            "RZ": qml.RZ,
            "RX": qml.RX
        }
        valid_Rs_names = search_cfg.get("valid_Rs", ["RY", "RZ"])
        self.valid_Rs = [gate_mapping[name] for name in valid_Rs_names]

        # Define valid CNOT wire pairs as nearest-neighbor connections only
        # Example for 5 qubits: (0,1), (1,2), (2,3), (3,4)
    
        self.n_qubits = n_qubits 
        self.Rs_space = [(gate, wire) for gate in self.valid_Rs for wire in range(self.n_qubits)]
        
        # Build the CNOT connectivity space
        # Generate all possible subsets of CNOT connections (including empty set)
        self.CNOTs_space =  [
            (i, (i + 1) % self.n_qubits)
            for i in range(self.n_qubits)
        ]
        
        # Build the complete NAS search space
        # Each element is a tuple (Rs_configuration, CNOTs_configuration)
        # Total size: len(Rs_space) × len(CNOTs_space)
        self.NAS_search_space = list(itertools.product(self.Rs_space, self.CNOTs_space))
        
        if run:
            try:
                run['search_space_info'] = {
                    "n_qubits": self.n_qubits,
                    "valid_Rs": valid_Rs_names,
                    "valid_CNOTs": list(self.CNOTs_space),
                    "nas_search_space_size": len(self.NAS_search_space)
                }
            except Exception as e:
                print(f"⚠️ Failed to log search space info: {e}")
                
    
    def __len__(self):
        """
        Return the total number of architectures in the search space.
        
        Returns:
            int: Size of the search space
        """
        return len(self.NAS_search_space)
    
    def __getitem__(self, idx):
        """
        Access a specific architecture configuration by index.
        
        Args:
            idx (int): Index of the architecture in the search space
            
        Returns:
            tuple: (Rs, CNOTs) where Rs is a tuple of rotation gates and 
                   CNOTs is a list of wire pairs for CNOT gates
        """
        return self.NAS_search_space[idx]

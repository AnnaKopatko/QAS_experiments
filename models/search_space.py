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
        import itertools

        n_rot = search_cfg.get("n_rotations_per_layer", 1)

        self.Rs_space = []

        qubits = list(range(self.n_qubits))

        # choose qubits without repetition
        for qubit_combo in itertools.combinations(qubits, n_rot):

            # choose gates (repetition allowed)
            for gates in itertools.product(self.valid_Rs, repeat=n_rot):

                layer = [(gate, wire) for gate, wire in zip(gates, qubit_combo)]

                self.Rs_space.append(layer)
                
        # Build the CNOT connectivity space
        # Each element is a tuple of n_cnots_per_layer wire pairs.
        # n_cnots_per_layer=0 yields one option: () — no CNOTs.
        n_cnots = search_cfg.get("n_cnots_per_layer", 1)
        valid_cnot_pairs = [(i, (i + 1) % self.n_qubits) for i in range(self.n_qubits)]
        self.CNOTs_space = list(itertools.combinations(valid_cnot_pairs, n_cnots))
        
        # Build the complete NAS search space
        # Each element is a tuple (Rs_configuration, CNOTs_configuration)
        # Total size: len(Rs_space) × len(CNOTs_space)
        self.NAS_search_space = list(itertools.product(self.Rs_space, self.CNOTs_space))

        # Storage for architectures created on-the-fly during evolution (e.g. CNOT dropout).
        # Populated only during the search phase; never touched during warmup.
        # Maps new_idx -> (Rs, CNOTs, r_idx) where r_idx is inherited from the parent.
        self._dynamic_extensions: dict = {}
        self._dynamic_key_to_idx: dict = {}
        self._next_dynamic_idx: int = len(self.NAS_search_space)

        if run:
            try:
                run['search_space_info'] = {
                    "n_qubits": self.n_qubits,
                    "valid_Rs": valid_Rs_names,
                    "valid_CNOTs": list(self.CNOTs_space),
                    "nas_search_space_size": len(self.NAS_search_space),
                    "rs_space_size": len(self.Rs_space),
                    "cnots_space_size": len(self.CNOTs_space),
                    "n_rotations_per_layer": n_rot,
                    "n_cnots_per_layer": n_cnots,
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

    # ------------------------------------------------------------------
    # Dynamic extension helpers (used by CNOT dropout during evolution)
    # ------------------------------------------------------------------

    def register_cnot_dropout_arch(self, Rs, CNOTs_reduced, parent_r_idx: int) -> int:
        """Register a CNOT-dropped architecture and return its index.

        If an identical (Rs, CNOTs_reduced) pair was already registered the
        existing index is returned, so the dict never accumulates duplicates.
        """
        key = (
            tuple((g.__name__, w) for g, w in Rs),
            tuple(CNOTs_reduced),
        )
        if key in self._dynamic_key_to_idx:
            return self._dynamic_key_to_idx[key]
        new_idx = self._next_dynamic_idx
        self._dynamic_extensions[new_idx] = (list(Rs), tuple(CNOTs_reduced), parent_r_idx)
        self._dynamic_key_to_idx[key] = new_idx
        self._next_dynamic_idx += 1
        return new_idx

    def get_arch_elem(self, idx: int):
        """Return (Rs, CNOTs) for any index — static or dynamic."""
        if idx < len(self.NAS_search_space):
            return self.NAS_search_space[idx]
        entry = self._dynamic_extensions[idx]
        return (entry[0], entry[1])

    def get_r_idx(self, idx: int) -> int:
        """Return the rotation-parameter slot index for any architecture index."""
        if idx < len(self.NAS_search_space):
            return idx // len(self.CNOTs_space)
        return self._dynamic_extensions[idx][2]


class BlockSearchSpace:
    """
    Block search space from Du et al. 2022.

    Each layer:
    - ALL N qubits receive one of G rotation gates → G^N rotation configs
    - Each of the N-1 nearest-neighbor pairs is independently on/off → 2^(N-1) CNOT configs

    Total per-layer options: G^N × 2^(N-1)

    For N=4: 2^4 × 2^3 = 128
    For N=6: 2^6 × 2^5 = 2,048
    For N=8: 2^8 × 2^7 = 32,768
    """

    def __init__(self, config, n_qubits, run=None):
        search_cfg = config.get("SearchSpace", {})

        gate_mapping = {"RY": qml.RY, "RZ": qml.RZ, "RX": qml.RX}
        valid_Rs_names = search_cfg.get("valid_Rs", ["RY", "RZ"])
        self.valid_Rs = [gate_mapping[name] for name in valid_Rs_names]
        self.n_qubits = n_qubits

        n_rot = search_cfg.get("n_rotations_per_layer", 1)
        n_cnots = search_cfg.get("n_cnots_per_layer", 1)

        # Rs_space: G^N combinations — every qubit gets one gate
        self.Rs_space = []
        for gates in itertools.product(self.valid_Rs, repeat=n_qubits):
            layer = [(gate, wire) for wire, gate in enumerate(gates)]
            self.Rs_space.append(layer)

        # CNOTs_space: all 2^(N-1) subsets of the N-1 nearest-neighbor pairs
        neighbor_pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        self.CNOTs_space = []
        for r in range(len(neighbor_pairs) + 1):
            for combo in itertools.combinations(neighbor_pairs, r):
                self.CNOTs_space.append(combo)

        self.NAS_search_space = list(itertools.product(self.Rs_space, self.CNOTs_space))

        if run:
            try:
                run['search_space_info'] = {
                    "type": "block",
                    "n_qubits": n_qubits,
                    "valid_Rs": valid_Rs_names,
                    "rs_space_size": len(self.Rs_space),
                    "cnots_space_size": len(self.CNOTs_space),
                    "nas_search_space_size": len(self.NAS_search_space),
                    "n_rotations_per_layer": n_rot,
                    "n_cnots_per_layer": n_cnots,
                }
            except Exception as e:
                print(f"⚠️ Failed to log search space info: {e}")

    def __len__(self):
        return len(self.NAS_search_space)

    def __getitem__(self, idx):
        return self.NAS_search_space[idx]

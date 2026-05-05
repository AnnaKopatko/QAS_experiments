"""
Circuit Model for Fixed Quantum Architectures

This module provides the CircuitModel class and supporting functions for building
and executing quantum circuits with either regular (fixed) or NAS-discovered architectures.

Key Components:
- layer(): Standard parameterized layer with RY gates and linear CNOT chain
- qas_layer(): Flexible layer used in NAS with configurable gates and connectivity
- circuit(): Generic circuit builder that works for both regular and NAS circuits
- CircuitModel: Main class for instantiating and running quantum circuits
"""

import pennylane as qml
from pennylane import numpy as np

def qas_layer(params, j, arch_elem, sandwich=False):
    """
    One NAS layer:
        - multiple rotations
        - zero or more CNOTs

    Args:
        params: shape (n_layers, n_rotations)
        j: layer index
        arch_elem: element from NAS_search_space
                   ([(gate, wire), ...], ((control, target), ...))
                   The CNOT tuple may be empty (no entanglement).
        sandwich: if True, split rotations around CNOTs:
                  first-half Rs → CNOTs → second-half Rs.
                  Adds expressivity without changing the search space.
    """

    Rs, cnots = arch_elem

    if not sandwich or not cnots:
        for k, (gate_cls, wire) in enumerate(Rs):
            gate_cls(params[j][k], wires=wire)
        for control, target in cnots:
            qml.CNOT(wires=[control, target])
    else:
        mid = len(Rs) // 2
        for k, (gate_cls, wire) in enumerate(Rs[:mid]):
            gate_cls(params[j][k], wires=wire)
        for control, target in cnots:
            qml.CNOT(wires=[control, target])
        for k, (gate_cls, wire) in enumerate(Rs[mid:]):
            gate_cls(params[j][mid + k], wires=wire)



# ======================================================
# Circuit Model Class
# ======================================================
class CircuitModel:
    """
    Quantum circuit model for fixed architectures.
    
    This class is used AFTER architecture search to train a specific circuit.
    It can work with either:
    1. Regular circuits (standard RY + CNOT chain)
    2. A fixed NAS architecture discovered during search
    
    The model maintains the circuit parameters and provides a callable interface
    for integration with PennyLane optimizers and QNodes.
    
    Attributes:
        dev: PennyLane quantum device (e.g., default.qubit, qiskit.aer)
        search_space (SearchSpace): Instance of the SearchSpace class
        n_qubits (int): Number of qubits
        n_layers (int): Number of circuit layers
        basis_state (list): Initial quantum state
        arch (str or list): Architecture specification (empty string for regular circuit)
        params (np.ndarray): Trainable circuit parameters, shape (n_layers, n_qubits)
    """
    
    def __init__(self, dev, search_space, n_qubits=3, n_layers=3, arch="", basis_state=None, sandwich=False):
        """
        Initialize the circuit model.
        
        Args:
            dev: PennyLane device to execute the circuit
            search_space (SearchSpace): Instance of the SearchSpace class
            n_qubits (int): Number of qubits in the circuit
            n_layers (int): Number of layers (depth of the circuit)
            arch (str): Architecture string (e.g., "5-12-3" for NAS, "" for regular)
            basis_state (list, optional): Initial state. Default: |000...0⟩
        """
        self.dev = dev
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.basis_state = basis_state
        self.search_space = search_space
        self.sandwich = sandwich

        # Parse and validate architecture if NAS mode is used
        if arch != "":
            # Convert string "5-12-3" to list [5, 12, 3]
            self.arch = [int(x) for x in arch.split("-")]
            
            # Sanity check: architecture length must match number of layers
            assert len(self.arch) == n_layers, \
                f"Architecture length {len(self.arch)} != number of layers {n_layers}"
            
        else:
            # Empty string indicates regular circuit (not NAS)
            self.arch = ""

        # Initialize parameters randomly in range [0, 2π]
        # Shape: (n_layers, n_qubits) - one parameter per gate per layer
        max_rot = max(len(r) for r in self.search_space.Rs_space)

        self.params = np.random.uniform(
            0, 2 * np.pi,
            (n_layers, max_rot)
        )

    def __call__(self, params=None, wires=None):
        """
        Execute the circuit with given parameters.
        
        Args:
            params (np.ndarray, optional): Circuit parameters. 
                If None, uses self.params
            wires (list or range, optional): Qubit wires to use.
                If None, uses range(self.n_qubits)

        """
        if params is None:
            params = self.params
        if wires is None:
            wires = range(self.n_qubits)
            
        # Build and execute the circuit
        self.circuit(params, wires)
        
    # ======================================================
    # Generic Circuit Builder
    # ======================================================
    def circuit(self, params, wires):
        """
        This function can build two types of circuits:
        1. Regular circuit (arch=None or ""): Uses standard layer() for all layers
        2. NAS circuit (arch provided): Uses qas_layer() with architecture from search space
        
        The circuit always starts by preparing an initial basis state, then applies
        n_layers sequential layers of parameterized gates.
        
        Args:
            params (np.ndarray): Parameters for the circuit, shape (n_layers, n_qubits)
            wires (list or range): Wire labels for the qubits
            n_qubits (int): Number of qubits
            n_layers (int): Number of layers in the circuit
            arch (str or list, optional): Architecture specification for NAS.
                If provided as string like "5-12-3", each number is an index into NAS_search_space.
                If None or "", uses regular layer() instead.
            basis_state (array-like, optional): Initial computational basis state.
                Example: [1, 0, 1, 0] prepares |1010⟩
                Default: |000...0⟩ (all zeros)
        """
        # Default basis state: |000...0⟩
        if self.basis_state is None:
            self.basis_state = np.zeros(self.n_qubits, dtype=int)

        # Initialize the quantum state
        qml.BasisState(np.array(self.basis_state), wires=wires)

        for j in range(self.n_layers):

            idx = int(self.arch[j])
            arch_elem = self.search_space.get_arch_elem(idx)
            qas_layer(params, j, arch_elem, sandwich=self.sandwich)

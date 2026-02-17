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

# ======================================================
# Regular Layer (works for any n_qubits)
# ======================================================
def layer(params, j, n_qubits):
    """
    Standard quantum circuit layer with fixed structure.
    
    This creates a basic ansatz layer consisting of:
    1. RY rotation on each qubit (parameterized)
    2. Linear chain of CNOT gates connecting adjacent qubits
    
    Architecture:
        RY(θ₀) -- CNOT -- RY(θ₁) -- CNOT -- RY(θ₂) -- CNOT -- RY(θ₃)
                   |                |                |
                   ----------------+----------------+
    
    Args:
        params (np.ndarray): Parameter array of shape (n_layers, n_qubits)
        j (int): Current layer index
        n_qubits (int): Number of qubits in the circuit
        
    Note:
        This is used for baseline/regular circuits, NOT for NAS circuits.
    """
    # Apply parameterized RY rotation to each qubit
    for i in range(n_qubits):
        qml.RY(params[j, i], wires=i)

    # Apply CNOT chain across all adjacent qubit pairs
    # This creates entanglement: qubit i is control, qubit i+1 is target
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])


# ======================================================
# QAS Layer (for NAS circuits)
# ======================================================
def qas_layer(params, j, n_qubits, Rs=None, CNOTs=None):
    """
    Quantum Architecture Search (QAS) layer with configurable structure.
    
    Unlike the standard layer(), this allows customization of:
    1. Which rotation gates to use (RY, RZ, or mix)
    2. Which qubits are entangled via CNOT gates
    
    This flexibility is essential for NAS, where different architectures
    are tried during the search process.
    
    Args:
        params (np.ndarray): Parameter array of shape (n_layers, n_qubits)
        j (int): Current layer index
        n_qubits (int): Number of qubits in the circuit
        Rs (list of callables, optional): List of rotation gates to apply.
            If fewer than n_qubits gates are provided, they will cycle.
            Default: [RY, RY, RY, RY]
        CNOTs (list of lists, optional): List of [control, target] wire pairs.
            Example: [[0,1], [2,3]] creates CNOTs between qubits 0-1 and 2-3.
            Default: Linear chain [[0,1], [1,2], ..., [n_qubits-2, n_qubits-1]]
    """
    # Default rotation gates: RY on all qubits
    if Rs is None:
        Rs = [qml.RY] * n_qubits
        
    # Default CNOT connectivity: linear chain
    if CNOTs is None:
        CNOTs = [[i, (i + 1) % n_qubits] for i in range(n_qubits - 1)]

    # Apply rotation gates to each qubit
    # If len(Rs) < n_qubits, cycle through the available gates
    for i in range(n_qubits):
        gate = Rs[i % len(Rs)]   # Modulo ensures we don't go out of bounds
        gate(params[j, i], wires=i)

    # Apply CNOT gates according to the specified connectivity
    for conn in CNOTs:
        qml.CNOT(wires=conn)



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
    
    def __init__(self, dev, search_space, n_qubits=3, n_layers=3, arch="", basis_state=None):
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


        # Parse and validate architecture if NAS mode is used
        if arch != "":
            # Convert string "5-12-3" to list [5, 12, 3]
            self.arch = [int(x) for x in arch.split("-")]
            
            # Sanity check: architecture length must match number of layers
            assert len(self.arch) == n_layers, \
                f"Architecture length {len(self.arch)} != number of layers {n_layers}"
            
            # Print architecture details for debugging/logging
            print("----NAS circuit----")
            for i, idx in enumerate(self.arch):
                Rs, CNOTs = self.search_space.NAS_search_space[idx]
                print(f"------Layer {i}------")
                print(f"Rs: {Rs}")
                print(f"CNOTs: {CNOTs}")
        else:
            # Empty string indicates regular circuit (not NAS)
            self.arch = ""

        # Initialize parameters randomly in range [0, 2π]
        # Shape: (n_layers, n_qubits) - one parameter per gate per layer
        self.params = np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits))

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

        # Build the circuit layer by layer
        for j in range(self.n_layers):
            if self.arch is None or self.arch == "":
                # Regular circuit: use standard layer
                layer(params, j, self.n_qubits)
            else:
                # NAS circuit: use architecture from search space
                # arch[j] is the index into NAS_search_space for this layer
                idx = int(self.arch[j])
                Rs, CNOTs = self.search_space.NAS_search_space[idx]  # Retrieve (rotation gates, CNOT pattern)
                qas_layer(params, j, self.n_qubits, Rs, CNOTs)

import pennylane as qml
from pennylane import numpy as np
import itertools


valid_Rs =  [qml.RY, qml.RZ]
valid_CNOTs = ([0, 1], [1, 2], [2, 3])

Rs_space = list(itertools.product(valid_Rs, valid_Rs, valid_Rs, valid_Rs))
CNOTs_space = [[y for y in CNOTs if y is not None] for CNOTs in list(itertools.product(*([x, None] for x in valid_CNOTs)))]
NAS_search_space = list(itertools.product(Rs_space, CNOTs_space))


### FOR THE REGULAR CIRCUT ↓↓↓
### FOR THE REGULAR CIRCUT ↓↓↓
### FOR THE REGULAR CIRCUT ↓↓↓


# ======================================================
# Regular Layer (works for any n_qubits)
# ======================================================
def layer(params, j, n_qubits):
    """Standard circuit layer: RY + chain CNOTs."""
    for i in range(n_qubits):
        qml.RY(params[j, i], wires=i)

    # CNOT chain across all adjacent pairs
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])


# ======================================================
# Generic Circuit Builder
# ======================================================
def circuit(params, wires, n_qubits, n_layers, arch  =None, basis_state=None):
    """Build either a regular or NAS circuit."""
    # Default basis state = all zeros
    if basis_state is None:
        basis_state = np.zeros(n_qubits, dtype=int)

    qml.BasisState(np.array(basis_state), wires=wires)

    for j in range(n_layers):
        if arch is None or arch == "":
            layer(params, j, n_qubits)
        else:
            idx = int(arch[j])
            Rs, CNOTs = NAS_search_space[idx]
            qas_layer(params, j, n_qubits, Rs, CNOTs)


# ======================================================
# Circuit Model
# ======================================================
class CircuitModel:
    def __init__(self, dev, n_qubits=3, n_layers=3, arch="", basis_state=None):
        self.dev = dev
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.basis_state = basis_state

        # Parse architecture if NAS mode is used
        if arch != "":
            self.arch = [int(x) for x in arch.split("-")]
            assert len(self.arch) == n_layers, "Architecture length != number of layers"
            print("----NAS circuit----")
            for i, idx in enumerate(self.arch):
                Rs, CNOTs = NAS_search_space[idx]
                print(f"------Layer {i}------")
                print(f"Rs: {Rs}")
                print(f"CNOTs: {CNOTs}")
        else:
            self.arch = ""

        # Initialize parameters
        self.params = np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits))

    def __call__(self, params=None, wires=None):
        if params is None:
            params = self.params
        if wires is None:
            wires = range(self.n_qubits)
        circuit(params, wires, self.n_qubits, self.n_layers, self.arch, self.basis_state)

### FOR THE SEACH CIRCUT ↓↓↓
### FOR THE SEACH CIRCUT ↓↓↓
### FOR THE SEACH CIRCUT ↓↓↓

def qas_layer(params, j, n_qubits, Rs=None, CNOTs=None):
    """
    Apply one layer of parameterized rotations + entangling CNOTs.
    Rs: list of rotation gates (default repeats RY)
    CNOTs: list of entangling connections
    """
    if Rs is None:
        Rs = [qml.RY] * n_qubits
    if CNOTs is None:
        CNOTs = [[i, (i + 1) % n_qubits] for i in range(n_qubits - 1)]

    for i in range(n_qubits):
        gate = Rs[i % len(Rs)]   # cycle Rs if fewer than n_qubits
        gate(params[j, i], wires=i)

    for conn in CNOTs:
        qml.CNOT(wires=conn)



class CircuitSearchModel():
    def __init__(self, dev, n_qubits=3, n_layers=3, n_experts=5, basis_state=None):
        self.dev = dev
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_experts = n_experts
        '''init params'''
        # randomly initialize parameters from a normal distribution
        self.params_space = np.random.uniform(0, np.pi * 2, (n_experts, n_layers, len(Rs_space), n_qubits))
        self.params = None
        self.basis_state = basis_state
    
    
    def get_params(self, subnet, expert_idx):
        self.subnet = subnet
        self.expert_idx = expert_idx
        params = []
        for j in range(self.n_layers):
            #we get rid of cnot information
            r_idx = subnet[j] // len(CNOTs_space)
            ei, jj, ri = int(expert_idx), int(j), int(r_idx)
            params.append(self.params_space[ei, jj, ri:ri+1])

        return np.concatenate(params, axis=0)

    def set_params(self, params):
        for j in range(self.n_layers):
            r_idx = self.subnet[j] // len(CNOTs_space)
            self.params_space[self.expert_idx, j, r_idx:r_idx+1] = params[j, :]

    def __call__(self, params, wires):
        self.circuit_search(params, wires=wires,
                       n_qubits=self.n_qubits,
                       n_layers=self.n_layers,
                       arch=self.subnet)
        
    def circuit_search(self, params, wires=[0,1,2,3], n_qubits=3, n_layers=3, arch=[]):
        """
        Quantum circuit with a configurable initial basis state.
        """
        # If no custom basis state is provided, start from |000...0>
        if self.basis_state is None:
            self.basis_state = [0] * len(wires)

        # Prepare the initial state
        qml.BasisState(np.array(self.basis_state), wires=wires)

        # Build NAS circuit layer by layer
        for j in range(n_layers):
            idx = int(arch[j])
            qas_layer(params, j, n_qubits,
                    NAS_search_space[idx][0],
                    NAS_search_space[idx][1])

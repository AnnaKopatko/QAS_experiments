import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms import decompose
from functools import partial


# ======================================================
# Circuit Builders
# ======================================================
def build_single_double_circuit(dev, hamiltonian, hf_state, n_qubits):
    """Single DoubleExcitation gate circuit."""
    @qml.qnode(dev, interface="autograd")
    def circuit(param):
        qml.BasisState(hf_state, wires=range(n_qubits))
        qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
        return qml.expval(hamiltonian)
    
    def cost(param):
        return circuit(param)
    
    # Initial parameter
    init_param = np.array(0.0, requires_grad=True)
    
    return circuit, cost, init_param


def build_uccsd_circuit(dev, hamiltonian, hf_state, n_qubits, n_electrons):
    """Full UCCSD ansatz circuit."""
    # Generate excitations
    singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
    n_params = len(singles) + len(doubles)
    
    print(f"Number of single excitations: {len(singles)}")
    print(f"Number of double excitations: {len(doubles)}")
    print(f"Total parameters: {n_params}")
    
    def uccsd_ansatz(params, wires):
        idx = 0
        # Single excitations
        for s in singles:
            qml.SingleExcitation(params[idx], wires=[wires[i] for i in s])
            idx += 1
        # Double excitations
        for d in doubles:
            qml.DoubleExcitation(params[idx], wires=[wires[i] for i in d])
            idx += 1
    
    @qml.qnode(dev, interface="autograd")
    def circuit(params):
        qml.BasisState(hf_state, wires=range(n_qubits))
        uccsd_ansatz(params, wires=range(n_qubits))
        return qml.expval(hamiltonian)
    
    def cost(params):
        return circuit(params)
    
    # Initial parameters (small random values)
    init_params = np.random.randn(n_params, requires_grad=True) * 0.01
    
    return circuit, cost, init_params, singles, doubles


def build_decomposed_circuit(dev, hamiltonian, hf_state, n_qubits, architecture, param, singles=None, doubles=None):
    """
    Build a fully decomposed circuit (using allowed gates) for visualization and specs.
    This version uses the modern PennyLane transform API with functools.partial.
    """
    allowed_gates = {qml.RX, qml.RY, qml.RZ, qml.CNOT}

    if architecture == "single_double":
        # --- Single DoubleExcitation ---
        @partial(decompose, gate_set=allowed_gates)
        @qml.qnode(dev, interface="autograd")
        def decomposed_circuit(p):
            qml.BasisState(hf_state, wires=range(n_qubits))
            qml.DoubleExcitation(p, wires=[0, 1, 2, 3])
            return qml.expval(hamiltonian)

    else:
        # --- Full UCCSD (Singles + Doubles) ---
        @partial(decompose, gate_set=allowed_gates)
        @qml.qnode(dev, interface="autograd")
        def decomposed_circuit(params):
            qml.BasisState(hf_state, wires=range(n_qubits))
            idx = 0
            for s in singles:
                qml.SingleExcitation(params[idx], wires=s)
                idx += 1
            for d in doubles:
                qml.DoubleExcitation(params[idx], wires=d)
                idx += 1
            return qml.expval(hamiltonian)

    return decomposed_circuit
import pennylane as qml
from pennylane import numpy as np


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
    """Build decomposed circuit for visualization based on architecture."""
    if architecture == 'single_double':
        @qml.qnode(dev)
        def decomposed_circuit(p):
            qml.BasisState(hf_state, wires=range(n_qubits))
            for op in qml.DoubleExcitation(p, wires=[0, 1, 2, 3]).decomposition():
                op.queue()
            return qml.expval(hamiltonian)
    else:  # uccsd
        @qml.qnode(dev)
        def decomposed_circuit(params):
            qml.BasisState(hf_state, wires=range(n_qubits))
            idx = 0
            # Decompose single excitations
            for s in singles:
                for op in qml.SingleExcitation(params[idx], wires=s).decomposition():
                    op.queue()
                idx += 1
            # Decompose double excitations
            for d in doubles:
                for op in qml.DoubleExcitation(params[idx], wires=d).decomposition():
                    op.queue()
                idx += 1
            return qml.expval(hamiltonian)
    
    return decomposed_circuit


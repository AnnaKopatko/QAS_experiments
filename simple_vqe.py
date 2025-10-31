import pennylane as qml
from pennylane import numpy as np
from pennylane.qchem.operations import single_excitation, double_excitation

# ---- Molecule definition (LiH) ----
symbols = ["Li", "H"]
coordinates = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.6]
])

# Build molecule and Hamiltonian
molecule = qml.qchem.Molecule(symbols, coordinates)
H, n_qubits = qml.qchem.molecular_hamiltonian(
    molecule,
    active_electrons=2,
    active_orbitals=4
)

print(f"Number of qubits: {n_qubits}")
print("Hamiltonian:\n", H)

# ---- Hartree–Fock reference ----
n_electrons = 2
hf_state = qml.qchem.hf_state(n_electrons, orbitals=n_qubits)
print("HF state:", hf_state)

# ---- Generate excitations ----
singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
print(f"Number of single excitations: {len(singles)}")
print(f"Number of double excitations: {len(doubles)}")

# ---- Device ----
dev = qml.device("default.qubit", wires=n_qubits)

# ---- Manual UCCSD ansatz ----
def uccsd_ansatz(params, wires):
    idx = 0
    # single excitations
    for s in singles:
        single_excitation(params[idx], s, wires=wires)
        idx += 1
    # double excitations
    for d in doubles:
        double_excitation(params[idx], d, wires=wires)
        idx += 1

# ---- QNode ----
@qml.qnode(dev)
def circuit(params):
    qml.BasisState(hf_state, wires=range(n_qubits))
    uccsd_ansatz(params, wires=range(n_qubits))
    return qml.expval(H)

def cost_fn(params):
    return circuit(params)

# ---- Initialize and optimize ----
n_params = len(singles) + len(doubles)
theta = np.zeros(n_params, requires_grad=True)

opt = qml.GradientDescentOptimizer(stepsize=0.4)
max_iterations = 100
conv_tol = 1e-6

energies = [cost_fn(theta)]
for n in range(max_iterations):
    theta, prev_energy = opt.step_and_cost(cost_fn, theta)
    curr_energy = cost_fn(theta)
    energies.append(curr_energy)
    conv = np.abs(energies[-1] - energies[-2])
    if n % 2 == 0:
        print(f"Step = {n:3d},  Energy = {curr_energy:.8f} Ha")
    if conv <= conv_tol:
        break

print("\nFinal ground-state energy = {:.8f} Ha".format(curr_energy))
print("Optimal parameters = ", theta)

# ---- Plot ----
try:
    import matplotlib.pyplot as plt
    plt.plot(energies, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.title("VQE Optimization for LiH (Manual UCCSD)")
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Could not plot results: {e}")

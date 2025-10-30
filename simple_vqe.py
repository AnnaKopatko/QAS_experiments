import pennylane as qml
from pennylane import numpy as np

# ---- Molecule definition ----
symbols = ["H", "H"]
coordinates = np.array([
    [0.0, 0.0, -0.6614],
    [0.0, 0.0,  0.6614]
])

molecule = qml.qchem.Molecule(symbols, coordinates)
H, qubits = qml.qchem.molecular_hamiltonian(molecule)

print(f"Number of qubits: {qubits}")
print("Hamiltonian:\n", H)

# ---- Hartree–Fock state ----
electrons = 2
hf_state = qml.qchem.hf_state(electrons, qubits)
print("HF state:", hf_state)

# ---- Device ----
dev = qml.device("default.qubit", wires=qubits)

# ---- Define circuit ----
@qml.qnode(dev)
def circuit(param):
    qml.BasisState(hf_state, wires=range(qubits))
    qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
    return qml.expval(H)

def cost_fn(param):
    return circuit(param)

# ---- Optimization ----
opt = qml.GradientDescentOptimizer(stepsize=0.4)
theta = np.array(0.0, requires_grad=True)

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
print("Optimal circuit parameter θ = {:.6f}".format(theta))

# ---- Visualization ----
try:
    import matplotlib.pyplot as plt
    plt.plot(energies, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.title("VQE Optimization for H₂")
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Could not plot results: {e}")

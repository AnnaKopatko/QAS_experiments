import yaml
import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem


def load_molecule_and_hf(mol_cfg):
    """Load molecule info, build Hamiltonian, and matching HF state."""
    symbols = mol_cfg["symbols"]
    coordinates = np.array(mol_cfg["coordinates"])
    basis = mol_cfg.get("basis", "sto-3g")

    # --- Build molecule and Hamiltonian ---
    molecule = qml.qchem.Molecule(symbols, coordinates)
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(molecule, active_electrons=mol_cfg["n_electrons"],
    active_orbitals=mol_cfg["active_orbitals"])

    # --- Correct HF state length ---
    n_electrons = mol_cfg["n_electrons"]
    n_orbitals = mol_cfg["active_orbitals"]

    # Double orbitals for spin — PennyLane expects qubits = 2 * n_orbitals
    n_spin_orbitals = 2 * n_orbitals
    hf_state = qchem.hf_state(
        electrons=n_electrons,
        orbitals=n_spin_orbitals
    )

    # print(f"\n=== Molecule: {mol_name} ===")
    print(f"Total qubits: {qubits}")
    print(f"Electrons: {n_electrons}, Orbitals: {n_orbitals}")
    print(f"HF basis state ({len(hf_state)} qubits): {hf_state}")
    print("Hamiltonian:\n", hamiltonian)

    return hamiltonian, qubits, hf_state


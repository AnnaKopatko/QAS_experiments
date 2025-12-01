import matplotlib.pyplot as plt
import json
import os
import pennylane as qml
import mlflow

# ==========================================================
# Helper Functions
# ==========================================================

def plot_energy_convergence(energies, exact_value, args):
    """Plot and save the VQE or NAS energy convergence curve."""
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(energies) + 1), energies, marker="o", label="Energy")
    plt.axhline(y=exact_value, color="r", linestyle="--", label="FCI Energy")

    # Title adapts based on presence of 'architecture'
    if hasattr(args, "architecture"):
        title = f"VQE Convergence for {args.mol_name} ({args.architecture})"
    else:
        title = f"Energy Convergence for {args.mol_name}"

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(args.save, "energy_convergence.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved convergence plot to {path}")
    return path



def save_circuit_diagram(circuit, param, save_path, title):
    """Draw and save a QNode circuit diagram."""
    drawer = qml.draw_mpl(circuit)
    fig, _ = drawer(param)
    fig.suptitle(title, fontsize=12)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved circuit diagram to {save_path}")
    return save_path


def extract_and_save_specs(decomposed_circuit, param, save_dir):
    """Extract circuit specs, print them, and save gate info."""
    specs = qml.specs(decomposed_circuit)(param)
    resources = specs["resources"]

    # Extract
    num_wires = resources.num_wires
    num_gates = resources.num_gates
    depth = resources.depth
    gate_types = dict(resources.gate_types)
    gate_sizes = dict(resources.gate_sizes)

    print("\n=== Decomposed Circuit Resources ===")
    print(f"Number of wires: {num_wires}")
    print(f"Total gates: {num_gates}")
    print(f"Depth: {depth}")
    print(f"Gate types: {gate_types}")
    print(f"Gate sizes: {gate_sizes}")

    # Save gate breakdown
    resources_path = os.path.join(save_dir, "circuit_gate_info.json")
    with open(resources_path, "w") as f:
        json.dump(
            {"gate_types": gate_types, "gate_sizes": gate_sizes}, f, indent=2
        )
    print(f"Saved gate type and size info to {resources_path}")

    return num_wires, num_gates, depth, resources_path


def log_to_mlflow(args, plots, specs_path, num_wires, num_gates, depth):
    """Log all generated artifacts and parameters to MLflow."""
    if not args.log_experiment:
        return

    # Log parameters
    mlflow.log_params({
        "num_wires": num_wires,
        "num_gates": num_gates,
        "depth": depth
    })

    # Log artifacts
    mlflow.log_artifact(specs_path)
    for path in plots:
        mlflow.log_artifact(path)

    print("✅ Logged all plots and specs to MLflow.")



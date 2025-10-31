import argparse
import json
import os
import time
import yaml
import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import mlflow
from utils.molecule_utils import load_molecule_and_hf
from utils.circut_utils import build_single_double_circuit, build_uccsd_circuit, build_decomposed_circuit




# ======================================================
# Argument Parsing
# ======================================================
def get_args():
    parser = argparse.ArgumentParser("Single-Circuit VQE")

    parser.add_argument('--epochs', type=int, default=100, help='number of optimization steps')
    parser.add_argument('--lr', type=float, default=0.2, help='optimizer learning rate')
    parser.add_argument('--device', type=str, default='default', choices=['default', 'ibmq-sim', 'ibmq'],
                        help='backend device')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--noise', action='store_true', default=False, help='use noise model')
    parser.add_argument('--log_experiment', action='store_true', default=False, help='enable MLflow logging')
    
    # NEW: Architecture selection
    parser.add_argument('--architecture', type=str, default='single_double', 
                        choices=['single_double', 'uccsd'],
                        help='Circuit architecture: single_double (one DoubleExcitation) or uccsd (full UCCSD ansatz)')

    parser.add_argument('--mol_name', type=str, default='H2',
                        choices=['H2', 'LiH'],
                        help='Select molecule to simulate (H2 or LiH)')

    args = parser.parse_args()

    experiments_root = "experiments"
    base_dir = os.path.join(experiments_root, "classic_archs")
    os.makedirs(base_dir, exist_ok=True)
    args.save = os.path.join(base_dir, f"eval-{args.mol_name}-{args.architecture}-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(args.save, exist_ok=True)

    with open(os.path.join(args.save, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    return args


# ======================================================
# Main
# ======================================================
def main():
    args = get_args()
    np.random.seed(args.seed)

    # Optional noise import
    if args.noise or args.device in ['ibmq-sim', 'ibmq']:
        import qiskit
        import qiskit_aer.noise as noise

    # -------------------------
    # Load molecule and HF state
    # -------------------------
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    mol_name = args.mol_name
    mol_cfg = cfg["Molecules"][mol_name]
    hamiltonian, n_qubits, hf_state = load_molecule_and_hf(mol_cfg=mol_cfg)
    n_electrons = mol_cfg.get("n_electrons", 2)  # Add this to your config

    # -------------------------
    # Start MLflow run (if enabled)
    # -------------------------
    if args.log_experiment:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("QAS_VQE_Experiments")
        noise_id = "noise" if args.noise else "no_noise"
        run_name = f"{args.mol_name}_{args.architecture}_{noise_id}_run_{time.strftime('%Y%m%d-%H%M%S')}"
        mlflow.start_run(run_name=run_name)

        mlflow.log_params({
            "epochs": args.epochs,
            "lr": args.lr,
            "device": args.device,
            "seed": args.seed,
            "noise": args.noise,
            "molecule": args.mol_name,
            "architecture": args.architecture,
            "n_qubits": n_qubits,
            "n_electrons": n_electrons,
            "basis": mol_cfg.get("basis", "sto-3g")
        })

    # -------------------------
    # Noise / device setup
    # -------------------------
    noise_cfg = cfg["Noise"]

    if args.device in ['ibmq-sim', 'ibmq']:
        from qiskit import IBMQ
        account_key = ''
        assert account_key != '', 'Please fill in your IBMQ account key.'
        IBMQ.save_account(account_key, overwrite=True)
        provider = IBMQ.enable_account(account_key)
        if args.device == 'ibmq':
            dev = qml.device('qiskit.ibmq', wires=n_qubits, backend='ibmq_ourense', provider=provider)
        else:
            backend = provider.get_backend('ibmq_ourense')
            noise_model = noise.NoiseModel().from_backend(backend)
            dev = qml.device('qiskit.aer', wires=n_qubits, noise_model=noise_model)
    else:
        if args.noise:
            prob_1 = noise_cfg["prob_1q"]
            prob_2 = noise_cfg["prob_2q"]
            if args.log_experiment:
                mlflow.log_params({"prob_1q": prob_1, "prob_2q": prob_2})
            error_1 = noise.depolarizing_error(prob_1, 1)
            error_2 = noise.depolarizing_error(prob_2, 2)
            noise_model = noise.NoiseModel()
            noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
            noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
            dev = qml.device('qiskit.aer', wires=n_qubits, noise_model=noise_model)
        else:
            dev = qml.device("default.qubit", wires=n_qubits)

    # -------------------------
    # Build circuit based on architecture
    # -------------------------
    singles, doubles = None, None
    
    if args.architecture == 'single_double':
        circuit, cost, param = build_single_double_circuit(dev, hamiltonian, hf_state, n_qubits)
        print("\n=== Using Single DoubleExcitation Architecture ===")
    else:  # uccsd
        circuit, cost, param, singles, doubles = build_uccsd_circuit(
            dev, hamiltonian, hf_state, n_qubits, n_electrons
        )
        print("\n=== Using Full UCCSD Architecture ===")

    # -------------------------
    # Optimization
    # -------------------------
    opt = qml.AdamOptimizer(stepsize=args.lr)
    exact_value = mol_cfg["exact_energy"]
    energies = []

    print("\n=== Training Start ===")
    start_time = time.time()
    for epoch in range(args.epochs):
        param = opt.step(cost, param)
        energy = cost(param)
        energies.append(float(energy))
        deviation = abs(energy - exact_value)

        if args.log_experiment:
            mlflow.log_metric("energy", float(energy), step=epoch)
            mlflow.log_metric("deviation", float(deviation), step=epoch)

        if epoch % 5 == 0:
            print(f"Step {epoch+1:3d}: Energy = {energy:.8f} Ha, ΔE = {deviation:.8f}")

    duration = time.time() - start_time
    print("\n=== Training Complete ===")
    print(f"Final energy = {energies[-1]:.8f} Ha")
    print(f"Deviation from FCI = {abs(energies[-1] - exact_value):.8f} Ha")

    # -------------------------
    # Save / log results
    # -------------------------
    results = {
        "energies": energies,
        "final_energy": float(energies[-1]),
        "final_param": param.tolist() if hasattr(param, 'tolist') else float(param),
        "deviation": float(abs(energies[-1] - exact_value)),
        "architecture": args.architecture
    }
    with open(os.path.join(args.save, "records.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.save}/records.json")
    
    if args.log_experiment:
        mlflow.log_artifact(os.path.join(args.save, "records.json"))

    # -------------------------
    # Visualization
    # -------------------------
    try:
        import matplotlib.pyplot as plt

        # ---- Energy convergence ----
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(energies)+1), energies, marker='o', label='Energy')
        plt.axhline(y=exact_value, color='r', linestyle='--', label='FCI Energy')
        plt.title(f"VQE Convergence for {args.mol_name} ({args.architecture})")
        plt.xlabel("Iteration")
        plt.ylabel("Energy (Ha)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        energy_plot_path = os.path.join(args.save, "energy_convergence.png")
        plt.savefig(energy_plot_path, dpi=300)
        plt.close()
        print(f"Saved convergence plot to {energy_plot_path}")

        # ---- High-level circuit visualization ----
        drawer = qml.draw_mpl(circuit)
        fig, _ = drawer(param)
        highlevel_path = os.path.join(args.save, "architecture_highlevel.png")
        fig.savefig(highlevel_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved high-level circuit diagram to {highlevel_path}")

        # ---- Decomposed circuit visualization ----
        decomposed_circuit = build_decomposed_circuit(
            dev, hamiltonian, hf_state, n_qubits, args.architecture, param, singles, doubles
        )
        drawer_decomp = qml.draw_mpl(decomposed_circuit)
        fig, _ = drawer_decomp(param)
        decomposed_path = os.path.join(args.save, "architecture_decomposed.png")
        fig.savefig(decomposed_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved decomposed circuit diagram to {decomposed_path}")

        # ---- Log plots to MLflow ----
        if args.log_experiment:
            mlflow.log_artifact(energy_plot_path)
            mlflow.log_artifact(highlevel_path)
            mlflow.log_artifact(decomposed_path)
            print("Logged all plots to MLflow successfully.")

    except Exception as e:
        print(f"Visualization skipped: {e}")
        
    if args.log_experiment:
        mlflow.end_run()


if __name__ == "__main__":
    main()
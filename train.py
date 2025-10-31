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

    parser.add_argument('--mol_name', type=str, default='LiH',
                        choices=['H2', 'LiH'],
                        help='Select molecule to simulate (H2 or LiH)')

    args = parser.parse_args()

    experiments_root = "experiments"
    base_dir = os.path.join(experiments_root, "classic_archs")
    os.makedirs(base_dir, exist_ok=True)
    args.save = os.path.join(base_dir, f"eval-{args.mol_name}-{time.strftime('%Y%m%d-%H%M%S')}")
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
        # ---- Define H₂ molecule ----
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Select molecule (H2 or Li3)
    mol_name = args.mol_name
    mol_cfg = cfg["Molecules"][mol_name]
    hamiltonian, n_qubits, hf_state = load_molecule_and_hf(mol_cfg=mol_cfg)

    # -------------------------
    # Start MLflow run (if enabled)
    # -------------------------
    if args.log_experiment:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("QAS_VQE_Experiments")
        run_name = f"{args.mol_name}_{args.noise}_run_{time.strftime('%Y%m%d-%H%M%S')}"
        mlflow.start_run(run_name=run_name)

        mlflow.log_params({
            "epochs": args.epochs,
            "lr": args.lr,
            "device": args.device,
            "seed": args.seed,
            "noise": args.noise,
            "molecule": args.mol_name,
            "n_qubits": n_qubits,
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
    # Circuit definition
    # -------------------------
    @qml.qnode(dev, interface="autograd")
    def circuit(param):
        qml.BasisState(hf_state, wires=range(n_qubits))
        #wires=[0, 1, 2, 3] will work for any molecule, since DoubleExcitation always needs 4 qubits.
        qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
        return qml.expval(hamiltonian)

    def cost(param):
        return circuit(param)

    # -------------------------
    # Optimization
    # -------------------------
    opt = qml.AdamOptimizer(stepsize=args.lr)
    exact_value = mol_cfg["exact_energy"]
    param = np.array(0.0, requires_grad=True)
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
        "final_param": float(param),
        "deviation": float(abs(energies[-1] - exact_value))
    }
    with open(os.path.join(args.save, "records.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.save}/records.json")

    if args.log_experiment:
        mlflow.log_artifact(os.path.join(args.save, "records.json"))
        mlflow.end_run()


if __name__ == "__main__":
    main()
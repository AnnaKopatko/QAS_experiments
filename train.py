import argparse
import json
import os
import time
import yaml
import pennylane as qml
from pennylane import numpy as np
from aim import Run, Image as AimImage, Text as AimText

from utils.molecule_utils import load_molecule_and_hf
from utils.circut_utils import (
    build_single_double_circuit,
    build_uccsd_circuit,
    build_decomposed_circuit
)
from utils.vis_and_logging import (
    plot_energy_convergence,
    save_circuit_diagram,
    extract_and_save_specs
)


# ======================================================
# Argument Parsing
# ======================================================
def get_args():
    parser = argparse.ArgumentParser("Single-Circuit VQE")

    parser.add_argument('--epochs', type=int, default=150, help='number of optimization steps')
    parser.add_argument('--lr', type=float, default=0.2, help='optimizer learning rate')
    parser.add_argument('--device', type=str, default='default',
                        choices=['default', 'ibmq-sim', 'ibmq'],
                        help='backend device')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--noise', action='store_true', default=True,
                        help='use noise model')
    parser.add_argument('--log_experiment', action='store_true', default=True,
                        help='enable Aim logging')
    parser.add_argument('--aim_repo', type=str, default='.aim', help='Aim repository path')
    parser.add_argument('--architecture', type=str, default='single_double',
                        choices=['single_double', 'uccsd'],
                        help='Circuit architecture')

    parser.add_argument('--mol_name', type=str, default='LiH',
                        choices=['H2', 'LiH'],
                        help='Molecule to simulate')

    args = parser.parse_args()

    # Output directory
    experiments_root = "experiments"
    base_dir = os.path.join(experiments_root, "classic_archs_new")
    os.makedirs(base_dir, exist_ok=True)
    args.save = os.path.join(
        base_dir,
        f"eval-{args.mol_name}-{args.architecture}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
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

    # Optional noise imports
    if args.noise or args.device in ['ibmq-sim', 'ibmq']:
        import qiskit
        import qiskit_aer.noise as noise

    # Load config
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    mol_cfg = cfg["Molecules"][args.mol_name]
    hamiltonian, n_qubits, hf_state = load_molecule_and_hf(mol_cfg=mol_cfg)
    n_electrons = mol_cfg.get("n_electrons", 2)

    # ======================================================
    # Aim Experiment Logging
    # ======================================================
    run = None
    if args.log_experiment:
        print("Logging to Aim...")
        noise_id = "noise" if args.noise else "no_noise"
        run_name = f"classic_{args.architecture}_{args.mol_name}_{noise_id}"
        try:
            run = Run(
                repo=args.aim_repo,
                experiment=f"QAS_Experiment"
            )
            run['name'] = run_name
            run["hparams"] = {
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
            }
            print("✅ Aim logging initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize Aim: {e}")
            run = None

    # ======================================================
    # Device and noise configuration
    # ======================================================
    noise_cfg = cfg["Noise"]

    if args.device in ['ibmq-sim', 'ibmq']:
        from qiskit import IBMQ
        account_key = ''
        assert account_key != '', 'Please fill in your IBMQ account key.'

        IBMQ.save_account(account_key, overwrite=True)
        provider = IBMQ.enable_account(account_key)

        if args.device == 'ibmq':
            dev = qml.device('qiskit.ibmq', wires=n_qubits,
                             backend='ibmq_ourense', provider=provider)
        else:
            backend = provider.get_backend('ibmq_ourense')
            noise_model = noise.NoiseModel().from_backend(backend)
            dev = qml.device('qiskit.aer', wires=n_qubits,
                             noise_model=noise_model)
    else:
        if args.noise:
            prob_1 = noise_cfg["prob_1q"]
            prob_2 = noise_cfg["prob_2q"]

            if run is not None:
                run["noise"] = {"prob_1q": prob_1, "prob_2q": prob_2}

            error_1 = noise.depolarizing_error(prob_1, 1)
            error_2 = noise.depolarizing_error(prob_2, 2)

            noise_model = noise.NoiseModel()
            noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
            noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

            dev = qml.device('qiskit.aer', wires=n_qubits, noise_model=noise_model)
        else:
            dev = qml.device("default.qubit", wires=n_qubits)

    # ======================================================
    # Circuit build
    # ======================================================
    singles, doubles = None, None

    if args.architecture == 'single_double':
        circuit, cost, param = build_single_double_circuit(
            dev, hamiltonian, hf_state, n_qubits
        )
        print("\n=== Using Single DoubleExcitation Architecture ===")
    else:
        circuit, cost, param, singles, doubles = build_uccsd_circuit(
            dev, hamiltonian, hf_state, n_qubits, n_electrons
        )
        print("\n=== Using Full UCCSD Architecture ===")

    # ======================================================
    # Training Loop (QNGOptimizer)
    # ======================================================
    step_size = args.lr
    opt = qml.QNGOptimizer(stepsize=step_size, lam=0.001, approx='block-diag')

    exact_value = mol_cfg["exact_energy"]
    energies = []

    print("\n=== Training Start (QNGOptimizer) ===")
    start_time = time.time()

    for epoch in range(args.epochs):
        param = opt.step(circuit, param)
        # Natural gradient update
        energy = float(cost(param))
        deviation = float(abs(energy - exact_value))
        energies.append(energy)

        if run is not None:
            try:
                run.track(energy, name="finetune_energy", step=epoch)
                run.track(deviation, name="finetune_deviation", step=epoch)
            except Exception as e:
                print(f"⚠️ Failed to log metrics at epoch {epoch}: {e}")

        if epoch % 5 == 0:
            print(f"Step {epoch+1:3d}: Energy = {energy:.8f} Ha, ΔE = {deviation:.8f}")


    duration = time.time() - start_time
    print("\n=== Training Complete ===")
    print(f"Final energy = {energies[-1]:.8f} Ha")
    print(f"Deviation from FCI = {abs(energies[-1] - exact_value):.8f} Ha")
    print(f"Training time = {duration:.2f}s")

    # ======================================================
    # Save Results
    # ======================================================
    results = {
        "energies": energies,
        "final_energy": energies[-1],
        "final_param": param.tolist() if hasattr(param, 'tolist') else float(param),
        "deviation": float(abs(energies[-1] - exact_value)),
        "architecture": args.architecture,
        "training_time": duration
    }

    records_path = os.path.join(args.save, "records.json")
    with open(records_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {records_path}")

    if run is not None:
        try:
            run["final_results"] = results
            
            # Log the JSON content as text, not the path
            with open(records_path, "r") as f:
                records_content = f.read()
            run.track(AimText(records_content), name="records_json", context={"type": "artifact"})
            
        except Exception as e:
            print(f"⚠️ Failed to log final results: {e}")

    # ======================================================
    # Visualization & Artifacts
    # ======================================================
    try:
        # Convergence plot
        energy_plot_path = plot_energy_convergence(energies, exact_value, args)

        # High-level circuit
        highlevel_path = save_circuit_diagram(
            circuit, param,
            os.path.join(args.save, "architecture_highlevel.png"),
            title="High-Level Circuit"
        )

        # Build decomposed circuit
        decomposed_circuit = build_decomposed_circuit(
            dev, hamiltonian, hf_state, n_qubits,
            args.architecture, param, singles, doubles
        )

        decomposed_path = save_circuit_diagram(
            decomposed_circuit, param,
            os.path.join(args.save, "architecture_decomposed.png"),
            title="Decomposed Circuit"
        )

        # Specs
        num_wires, num_gates, depth, specs_path = extract_and_save_specs(
            decomposed_circuit, param, args.save
        )

        # Log artifacts to Aim with proper types
        if run is not None:
            try:
                # Log images
                if os.path.exists(energy_plot_path):
                    run.track(AimImage(energy_plot_path), name="energy_convergence", 
                             context={"type": "visualization"})
                
                if os.path.exists(highlevel_path):
                    run.track(AimImage(highlevel_path), name="circuit_highlevel", 
                             context={"type": "visualization"})
                
                if os.path.exists(decomposed_path):
                    run.track(AimImage(decomposed_path), name="circuit_decomposed", 
                             context={"type": "visualization"})
                
                # Log circuit specs as text
                if os.path.exists(specs_path):
                    with open(specs_path, "r") as f:
                        specs_content = f.read()
                    run.track(AimText(specs_content), name="circuit_specs_txt", 
                             context={"type": "artifact"})
                
                # Log circuit specs as metrics
                run["circuit_specs"] = {
                    "num_wires": int(num_wires),
                    "num_gates": int(num_gates),
                    "depth": int(depth)
                }
                
                print("✅ Successfully logged all artifacts to Aim")
                
            except Exception as e:
                print(f"⚠️ Failed to log artifacts to Aim: {e}")

    except Exception as e:
        print(f"⚠️ Visualization skipped due to error: {e}")

    # Close Aim run
    if run is not None:
        try:
            run.close()
            print("✅ Aim run closed successfully")
        except Exception as e:
            print(f"⚠️ Error closing Aim run: {e}")

    print("\n=== Finished ===")


if __name__ == "__main__":
    main()
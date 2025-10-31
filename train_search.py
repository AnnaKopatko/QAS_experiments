import argparse
import json
import yaml
import os

import time
import pennylane as qml
from pennylane import numpy as np
from model import CircuitSearchModel, NAS_search_space
from evolution.evolution_sampler import EvolutionSampler
import mlflow
from utils.molecule_utils import load_molecule_and_hf
from utils.utils import parse_architecture_key, expert_evaluator


# ======================================================
# Argument Parsing
# ======================================================
def get_args():
    parser = argparse.ArgumentParser("Quantum Architecture Search (QAS)")
    parser.add_argument('--epochs', type=int, default=500, help='training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=100, help='warm-up epochs')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers per subnet')
    parser.add_argument('--n_experts', type=int, default=3, help='number of experts')
    parser.add_argument('--n_search', type=int, default=100, help='number of search iterations')
    parser.add_argument('--ea_pop_size', type=int, default=20, help='population size (evolution)')
    parser.add_argument('--ea_gens', type=int, default=10, help='number of generations (evolution)')
    parser.add_argument('--searcher', type=str, default='evolution', choices=['random', 'evolution'])
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument(
        '--mol_name',
        type=str,
        default='H2',
        choices=['H2', 'LiH'],
        help='Select molecule to simulate (H2 or LiH)'
    )
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--log_experiment', action='store_true', default=False,
                    help='enable MLflow experiment logging')


    # ---- Added noise/device args ----
    parser.add_argument('--noise', action='store_true', default=False, help='use noise model')
    parser.add_argument('--device', type=str, default='default', choices=['default', 'ibmq-sim', 'ibmq'],
                        help='which backend device to use')

    args = parser.parse_args()

    experiments_root = os.path.join("experiments")
    os.makedirs(experiments_root, exist_ok=True)
    args.save = os.path.join(experiments_root, f'search-{args.save}-{time.strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(args.save, exist_ok=True)
    with open(os.path.join(args.save, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    return args





# ======================================================
# Main QAS logic
# ======================================================
def main():
    args = get_args()
    np.random.seed(args.seed)

    # ---- Optional noise import ----
    if args.noise or args.device in ['ibmq-sim', 'ibmq']:
        import qiskit
        import qiskit_aer.noise as noise     

    # ---- Define H₂ molecule ----
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Select molecule (H2 or Li3)
    mol_name = args.mol_name
    mol_cfg = cfg["Molecules"][mol_name]
    
    if args.log_experiment:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("QAS_Search_Experiments")
        noise_id = "noise" if args.noise else "no_noise"
        run_name = f"{mol_name}_{noise_id}_{args.searcher}_{time.strftime('%Y%m%d-%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        # Log experiment parameters
        mlflow.log_params({
            "epochs": args.epochs,
            "n_layers": args.n_layers,
            "n_experts": args.n_experts,
            "n_search": args.n_search,
            "ea_pop_size": args.ea_pop_size,
            "ea_gens": args.ea_gens,
            "searcher": args.searcher,
            "device": args.device,
            "noise": args.noise,
            "seed": args.seed,
        })

# ---- Load molecule and HF state ----
    hamiltonian, qubits, hf_state = load_molecule_and_hf(mol_cfg)
    noise_cfg = cfg["Noise"]
    # ---- Device ----
    if args.device in ['ibmq-sim', 'ibmq']:
        from qiskit import IBMQ
        account_key = ''
        assert account_key != '', 'Please fill in your IBMQ account key.'
        IBMQ.save_account(account_key, overwrite=True)
        provider = IBMQ.enable_account(account_key)
        if args.device == 'ibmq':
            dev = qml.device('qiskit.ibmq', wires=qubits, backend='ibmq_ourense', provider=provider)
        else:
            backend = provider.get_backend('ibmq_ourense')
            noise_model = noise.NoiseModel().from_backend(backend)
            dev = qml.device('qiskit.aer', wires=qubits, noise_model=noise_model)
    else:
        if args.noise:
            prob_1 = noise_cfg["prob_1q"]
            prob_2 = noise_cfg["prob_2q"]
            error_1 = noise.depolarizing_error(prob_1, 1)
            error_2 = noise.depolarizing_error(prob_2, 2)
            noise_model = noise.NoiseModel()
            noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
            noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
            print(noise_model)
            dev = qml.device('qiskit.aer', wires=qubits, noise_model=noise_model)
        else:
            dev = qml.device("default.qubit", wires=qubits)

    args.n_qubits = qubits

    # --- Hartree–Fock reference state ---
    basis_state = hf_state
    print("HF basis state:", basis_state)
    
    # ---- Model ----
    model = CircuitSearchModel(dev, args.n_qubits, args.n_layers, args.n_experts, basis_state=basis_state)

    @qml.qnode(dev)
    def circuit(params):
        model(params, wires=range(args.n_qubits))
        return qml.expval(hamiltonian)

    def cost(params):
        return circuit(params)

    opt = qml.AdamOptimizer(stepsize=0.2)
    exact_value = mol_cfg["exact_energy"]

    # ======================================================
    # Warm-up training
    # ======================================================
    print("\n=== Warm-up Training ===")
    for epoch in range(args.epochs):
        subnet = np.random.randint(0, len(NAS_search_space), args.n_layers).tolist()
        expert_idx = np.random.randint(args.n_experts) if epoch < args.warmup_epochs \
                     else expert_evaluator(model, subnet, args.n_experts, cost)
        model.params = model.get_params(subnet, expert_idx)
        model.params = opt.step(cost, model.params)
        model.set_params(model.params)
        
        if epoch % 50 == 0:
            energy = cost(model.params)
            print(f"Epoch {epoch}/{args.epochs}: Energy = {energy:.8f}")
            if args.log_experiment:
                mlflow.log_metric("warmup_energy", float(energy), step=epoch)

    # ======================================================
    # Architecture search (with expert tracking)
    # ======================================================
    print("\n=== Start Architecture Search ===")
    result = {}  # Maps subnet_key -> (energy, expert_idx)

    if args.searcher == 'random':
        for i in range(args.n_search):
            subnet = np.random.randint(0, len(NAS_search_space), args.n_layers).tolist()
            expert_idx = expert_evaluator(model, subnet, args.n_experts, cost)
            model.params = model.get_params(subnet, expert_idx)
            energy = cost(model.params)
            subnet_key = '-'.join(map(str, subnet))
            result[subnet_key] = (float(energy), int(expert_idx))
            print(f"{i+1}/{args.n_search}: subnet={subnet}, expert={expert_idx}, energy={energy:.6f}")

            # MLflow logging for each architecture
            if args.log_experiment:
                mlflow.log_metric("search_energy", float(energy), step=i)
                mlflow.log_text(
                    f"Iter {i+1}: subnet={subnet}, expert={expert_idx}, energy={energy:.6f}\n",
                    artifact_file="search_progress.log"
                )

    else:
        sampler = EvolutionSampler(
            pop_size=args.ea_pop_size,
            n_gens=args.ea_gens,
            n_layers=args.n_layers,
            n_blocks=len(NAS_search_space)
        )

        # Track expert indices during evolution
        expert_map = {}  # subnet_key -> expert_idx
        
        def eval_func(subnet):
            expert_idx = expert_evaluator(model, subnet, args.n_experts, cost)
            model.params = model.get_params(subnet, expert_idx)
            energy = cost(model.params)
            
            subnet_key = '-'.join(map(str, subnet))
            expert_map[subnet_key] = int(expert_idx)

            # MLflow logging per architecture evaluation
            if args.log_experiment:
                mlflow.log_metric("evolution_energy", float(energy))
            return float(energy)

        sampler.sample(eval_func)
        
        # Combine energy with expert info
        for subnet_key, energy in sampler.subnet_eval_dict.items():
            expert_idx = expert_map.get(subnet_key, 0)
            result[subnet_key] = (float(energy), expert_idx)

    # Save sorted results to file
    sorted_result = sorted(result.items(), key=lambda x: x[1][0])  # Sort by energy
    with open(os.path.join(args.save, 'nas_result_sorted.txt'), 'w') as f:
        for k, (energy, expert_idx) in sorted_result:
            f.write(f"{k} energy={energy:.12f} expert={expert_idx}\n")

    # Log the results file as an artifact
    if args.log_experiment:
        mlflow.log_artifact(os.path.join(args.save, 'nas_result_sorted.txt'))

    if not sorted_result:
        print("No architectures found.")
        if args.log_experiment:
            mlflow.end_run()
        return

    # ======================================================
    # Process candidate architectures (NO re-evaluation)
    # ======================================================
    search_space_size = len(NAS_search_space)
    evaluated_entries = []
    best_energy_entry = None

    def safe_parse_subnet(raw_subnet):
        if isinstance(raw_subnet, str):
            try:
                return parse_architecture_key(raw_subnet, search_space_size)
            except Exception:
                pass
        elif isinstance(raw_subnet, (list, np.ndarray)):
            raw_subnet = np.asarray(raw_subnet, dtype=float)
            clipped = np.clip(np.rint(raw_subnet), 0, search_space_size - 1)
            return clipped.astype(int).tolist()
        raise ValueError(f"Cannot parse subnet: {raw_subnet}")

    for arch_key, (stored_energy, expert_idx) in sorted_result:
        try:
            subnet = safe_parse_subnet(arch_key)
        except ValueError as exc:
            print(f"Skipping architecture {arch_key}: {exc}")
            continue
        
        if len(subnet) != args.n_layers:
            print(f"Skipping architecture {arch_key}: expected {args.n_layers} layers, got {len(subnet)}")
            continue

        # Use stored results - no re-evaluation needed!
        deviation = abs(stored_energy - exact_value)
        subnet_str = '-'.join(map(str, subnet))
        
        entry = {
            "arch_key": arch_key,
            "subnet": subnet,
            "subnet_str": subnet_str,
            "energy": stored_energy,
            "deviation": float(deviation),
            "expert_idx": expert_idx,
        }
        evaluated_entries.append(entry)

        if best_energy_entry is None or stored_energy < best_energy_entry["energy"]:
            best_energy_entry = entry

    if not evaluated_entries:
        print("No architectures could be evaluated; aborting fine-tuning.")
        if args.log_experiment:
            mlflow.end_run()
        return

    # Save detailed results
    detailed_path = os.path.join(args.save, 'nas_result_with_energy.txt')
    with open(detailed_path, 'w') as f:
        for item in evaluated_entries:
            f.write(
                "{} energy={:.12f} deviation={:.12f} expert={}\n".format(
                    item["subnet_str"],
                    item["energy"],
                    item["deviation"],
                    item["expert_idx"],
                )
            )

    print("\n=== Best Architecture (Lowest Energy) ===")
    print(f"Original key: {best_energy_entry['arch_key']}")
    print("Subnet:", best_energy_entry["subnet"])
    print("Expert:", best_energy_entry["expert_idx"])
    print(f"Energy: {best_energy_entry['energy']:.8f} Ha")
    print(f"Deviation: {best_energy_entry['deviation']:.8f} Ha")
    
    if args.log_experiment:
        mlflow.log_metrics({
            "energy_before_finetune": float(best_energy_entry['energy']),
            "deviation_before_finetune": float(best_energy_entry['deviation'])
        })

    best_subnet = best_energy_entry["subnet"]
    best_expert = best_energy_entry["expert_idx"]

    # ======================================================
    # Fine-tune best architecture
    # ======================================================
    fine_tune_epochs = 20
    model.params = model.get_params(best_subnet, best_expert)
    
    print("\n=== Fine-tuning Best Architecture ===")
    fine_tune_energies = []
    for epoch in range(fine_tune_epochs):
        model.params = opt.step(cost, model.params)
        model.set_params(model.params)
        energy = cost(model.params)
        fine_tune_energies.append(energy)
        deviation = abs(energy - exact_value)
        
        if args.log_experiment:
            mlflow.log_metric("finetune_energy", float(energy), step=epoch)
        print(f"Epoch {epoch+1}/{fine_tune_epochs}: Energy = {energy:.8f}, ΔE = {deviation:.8f}")
        
    final_energy = cost(model.params)
    print(f"\nFinal Energy after fine-tuning: {final_energy:.8f} Ha")
    print(f"Deviation from FCI: {abs(final_energy - exact_value):.8f} Ha")
    
    if args.log_experiment:
        mlflow.log_metrics({
            "final_energy": float(final_energy),
            "final_deviation": float(abs(final_energy - exact_value))
        })

    np.save(os.path.join(args.save, 'best_finetuned_params.npy'), model.params)

    # ======================================================
    # Visualization: Fine-tuning convergence
    # ======================================================
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(range(fine_tune_epochs), fine_tune_energies, marker='o')
        plt.axhline(y=exact_value, color='r', linestyle='--', label='FCI Reference')
        plt.title("Fine-tuning Energy Convergence (H₂)")
        plt.xlabel("Epoch")
        plt.ylabel("Energy (Ha)")
        plt.legend()
        plt.grid(True)

        fine_tune_plot_path = os.path.join(args.save, 'fine_tune_plot.png')
        plt.savefig(fine_tune_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved fine-tuning plot to {fine_tune_plot_path}")

        if args.log_experiment:
            mlflow.log_artifact(fine_tune_plot_path)

    except Exception as e:
        print(f"Skipping visualization: {e}")

    # ======================================================
    # Draw and save the final circuit diagram
    # ======================================================
    try:
        import matplotlib.pyplot as plt

        fig, ax = qml.draw_mpl(circuit)(model.params)
        fig.suptitle("Final Optimized Circuit (H₂)", fontsize=12)
        circuit_path = os.path.join(args.save, "final_circuit.png")
        fig.savefig(circuit_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Circuit diagram saved to {circuit_path}")

        if args.log_experiment:
            mlflow.log_artifact(circuit_path)

    except Exception as e:
        print(f"Could not draw circuit diagram: {e}")
        
    if args.log_experiment:
        mlflow.end_run()


if __name__ == "__main__":
    main()